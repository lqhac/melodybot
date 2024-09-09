from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.conformer.conformer import ConformerDecoder, ConformerEncoder
from modules.fastspeech.fast_tacotron import TacotronEncoder, DecoderRNN, Tacotron2Encoder
from modules.fastspeech.speedy_speech.speedy_speech import ConvBlocks
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    EnergyPredictor, RefEncoder, ConvEmbedding, ConvEmbedding2, ConvEmbedding3, FastspeechEncoder, VQVAEVarianceEncoder, \
    mel2ph_to_dur
from modules.commons.mixture import sample_from_mixture, sample_from_discretized_mix_logistic
from utils.cwt import cwt2f0
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0

FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, embed_tokens, d: TacotronEncoder(
        hp['hidden_size'], len(d), hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, embed_tokens, d: Tacotron2Encoder(len(d), hp['hidden_size']),
    'conformer': lambda hp, embed_tokens, d: ConformerEncoder(embed_tokens, len(d)),
}

FS_DECODERS = {
    'fft': lambda hp: FastspeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations']),
    'conformer': lambda hp: ConformerDecoder(hp['hidden_size']),
}


class FastSpeech2(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins'] if hparams['mel_mix_loss'] == 'none' else \
                80 * hparams['mix_num'] * 3
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)

        if hparams['use_spk_id']:  # use_spk_id=True与fs2一起训练 False就是用voice encoder的embedding
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        elif hparams['use_spk_embed']:
            self.spk_embed_proj = Linear(
                256, self.hidden_size if not hparams['spk_embed_affine'] else self.hidden_size * 2, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            if hparams['pitch_embed_type'] == 1:
                self.pitch_embed = ConvEmbedding(300, self.hidden_size)
            elif hparams['pitch_embed_type'] == 2:
                self.pitch_embed = ConvEmbedding2(300, self.hidden_size)
            elif hparams['pitch_embed_type'] == 3:
                self.pitch_embed = ConvEmbedding3(300, self.hidden_size)
            elif hparams['pitch_embed_type'] == 4:
                self.pitch_embed = nn.Conv1d(1, self.hidden_size, 9, padding=4)
            else:
                self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            if hparams['pitch_type'] == 'cwt':
                h = hparams['cwt_hidden_size']
                cwt_out_dims = 10
                if hparams['use_uv']:
                    cwt_out_dims = cwt_out_dims + 1
                self.cwt_predictor = nn.Sequential(
                    nn.Linear(self.hidden_size, h),
                    PitchPredictor(
                        h,
                        n_chans=predictor_hidden,
                        n_layers=hparams['predictor_layers'],
                        dropout_rate=hparams['predictor_dropout'], odim=cwt_out_dims,
                        padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel']))
                self.cwt_stats_layers = nn.Sequential(
                    nn.Linear(self.hidden_size, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2)
                )
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size,
                    n_chans=predictor_hidden,
                    n_layers=hparams['predictor_layers'],
                    dropout_rate=hparams['predictor_dropout'],
                    odim=2 if hparams['pitch_type'] == 'frame' else 1,
                    padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(256, self.hidden_size, self.padding_idx)
            self.energy_predictor = EnergyPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['predictor_layers'],
                dropout_rate=hparams['predictor_dropout'], odim=1,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['use_ref_enc']:
            self.ref_encoder = RefEncoder(
                hparams['audio_num_mel_bins'],
                hparams['ref_hidden_stride_kernel'],
                ref_norm_layer=hparams['ref_norm_layer'])
        if hparams['use_var_enc']:
            self.variance_encoder = VQVAEVarianceEncoder(self.hidden_size, hparams['var_enc_vq_codes'])

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur=None, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add ref style embed
        if hparams['use_ref_enc']:
            ref_embed = self.ref_encoder(ref_mels)[:, None, :]
            encoder_out = encoder_out + ref_embed
        # variance encoder
        if hparams['use_var_enc']:
            var_embed = self.forward_var_enc(encoder_out, mel2ph, txt_tokens, f0, uv, ret)
        else:
            var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        encoder_out_dur = encoder_out
        if hparams['use_spk_embed'] or hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
            encoder_out = self.add_spk_embed(spk_embed, encoder_out) * src_nonpadding
            if spk_embed_dur is not None and hparams['use_spk_id']:
                spk_embed_dur = self.spk_embed_proj(spk_embed_dur)[:, None, :]
                encoder_out_dur = self.add_spk_embed(spk_embed_dur, encoder_out_dur) * src_nonpadding
            else:
                encoder_out_dur = encoder_out

        # add dur
        mel2ph = self.add_dur(encoder_out_dur + var_embed, mel2ph, txt_tokens, ret)
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        # add pitch embed
        if hparams['use_pitch_embed']:
            decoder_inp = decoder_inp + self.add_pitch(
                decoder_inp_origin + var_embed, f0, uv, mel2ph, ret,
                encoder_out=encoder_out + var_embed)
        # add energy embed
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin + var_embed, energy, ret)

        tgt_nonpadding = (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp = decoder_inp * tgt_nonpadding
        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, **kwargs)
        return ret

    def add_spk_embed(self, spk_embed, encoder_out):
        if hparams['spk_embed_affine']:
            scale = F.relu(spk_embed[:, :, :self.hidden_size])
            bias = spk_embed[:, :, self.hidden_size:]
            return scale * encoder_out + bias
        else:
            return encoder_out + spk_embed

    def forward_var_enc(self, encoder_out, mel2ph, txt_tokens, f0, uv, ret):
        if mel2ph is not None:
            durs = mel2ph_to_dur(mel2ph, txt_tokens.shape[1], 31)
            durs = durs.float()
            nonpadding = (txt_tokens > 0).float()
            dur_mean = durs.float().sum(-1) / nonpadding.sum(-1)
            dur_mean = dur_mean[:, None]
            durs = durs * nonpadding + dur_mean * (1 - nonpadding)
            durs = durs.long()
            f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=mel2ph == 0)
            pitch = f0_to_coarse(f0_denorm)  # start from 0
        else:
            durs = None
            pitch = None
        var_embed, ret['vq_loss'] = self.variance_encoder(encoder_out, pitch, durs)  # [B, 1, H]
        return var_embed

    def add_dur(self, encoder_out, mel2ph, txt_tokens, ret):
        """

        :param encoder_out: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        dur_input = encoder_out
        if mel2ph is not None:
            dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if mel2ph is None:
            dur, xs = self.dur_predictor.inference(dur_input, src_padding)
            ret['dur'] = xs
            ret['dur_choice'] = dur
            mel2ph = self.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_padding)
        ret['mel2ph'] = mel2ph
        return mel2ph

    def add_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        if energy is None:
            energy = energy_pred
        energy = torch.clamp(energy * 256 // 4, max=255).long()
        energy_embed = self.energy_embed(energy)
        return energy_embed

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if hparams['pitch_type'] == 'ph':
            pitch_pred_inp = encoder_out.detach() + hparams['predictor_grad'] * (encoder_out - encoder_out.detach())
            pitch_padding = encoder_out.sum().abs() == 0
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, None, hparams, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
            pitch = F.pad(pitch, [1, 0])
            pitch = torch.gather(pitch, 1, mel2ph)  # [B, T_mel]
            pitch_embed = self.pitch_embed(pitch)
            return pitch_embed
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        pitch_padding = mel2ph == 0
        if hparams['pitch_type'] == 'cwt':
            pitch_padding = None
            ret['cwt'] = cwt_out = self.cwt_predictor(decoder_inp)
            stats_out = self.cwt_stats_layers(encoder_out[:, 0, :])  # [B, 2]
            mean = ret['f0_mean'] = stats_out[:, 0]
            std = ret['f0_std'] = stats_out[:, 1]
            cwt_spec = cwt_out[:, :, :10]
            if f0 is None:
                f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
                if hparams['use_uv']:
                    assert cwt_out.shape[-1] == 11
                    uv = cwt_out[:, :, -1] > 0
        elif hparams['pitch_ar']:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp, f0 if self.training else None)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
                uv = pitch_pred[:, :, 1] > 0
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        if pitch_padding is not None:
            f0[pitch_padding] = 0
        if hparams['pitch_embed_type'] == 4:
            pitch_embed = self.pitch_embed(f0[:, None, :]).transpose(1, 2)
        else:
            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        f0 = cwt2f0(cwt_spec, mean, std, hparams['cwt_scales'])
        f0 = torch.cat(
            [f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = (f0 - hparams['f0_mean']) / hparams['f0_std']
        return f0_norm

    def out2mel(self, out):
        if hparams['mel_mix_loss'] == 'none':
            return out
        else:
            B, T, n_bins = out.shape
            n_bins = n_bins // (hparams['mix_num'] * 3)
            out = out.reshape(B, T * n_bins, -1).transpose(1, 2)  # [B, C, T*80]
            if hparams['mel_mix_loss'] == 'gaus':
                out = sample_from_mixture(out, greedy=hparams['greedy_sample'],
                                          dist_cls=torch.distributions.Normal)
            if hparams['mel_mix_loss'] == 'lap':
                out = sample_from_mixture(out, greedy=hparams['greedy_sample'],
                                          dist_cls=torch.distributions.Laplace)
            if hparams['mel_mix_loss'] == 'log':
                out = sample_from_discretized_mix_logistic(out, greedy=hparams['greedy_sample'])
            out = out.reshape(B, T, n_bins)  # [B, T*80, 1]
            return self.mel_denorm(out)

    @staticmethod
    def mel_norm(x):
        return (x + 5.5) / (6.3 / 2) - 1

    @staticmethod
    def mel_denorm(x):
        return (x + 1) * (6.3 / 2) - 5.5
