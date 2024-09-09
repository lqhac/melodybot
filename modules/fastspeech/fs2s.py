from modules.fastspeech.fs2 import FastSpeech2
from modules.commons.common_layers import *
from modules.parallel_wavegan.models.melgan import MelGANGenerator
from modules.parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator
from utils.hparams import hparams


class WavDecoder(nn.Module):
    def __init__(self, wav_decoder_type='pwg'):
        super().__init__()
        self.wav_decoder_type = wav_decoder_type
        if wav_decoder_type == 'pwg':
            self.hidden_size = hparams['hidden_size']
            self.dropout = hparams['dropout']
            self.mel_out_dim = hparams['audio_num_mel_bins']
            self.aux_context_window = hparams['generator_params']['aux_context_window']
            self.generator = ParallelWaveGANGenerator(**hparams['generator_params'])
        if wav_decoder_type == 'melgan':
            self.generator = MelGANGenerator(**hparams['generator_params'])

    def forward(self, c):
        """

        :param c: [B, T, C]
        :param pitch: [B, T]
        :return:
        """
        B, T, _ = c.shape
        c = c.permute(0, 2, 1)  # [B, C, T]
        if self.wav_decoder_type == 'pwg':
            # [B, 1, T_wav]
            z = c.new_zeros([B, 1, (T - 2 * self.aux_context_window) * hparams['hop_size']]).normal_()
            w = self.generator(z, c)
            w = w.view(B, -1)[:, None, :]  # [B, 1, T_wav]
        if self.wav_decoder_type == 'melgan':
            w = self.generator(c)  # [B, 4, T_wav]
        return w


class FastSpeech2s(nn.Module):
    def __init__(self, dictionary, wav_decoder_type='pwg'):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.fs_encoder = FastSpeech2(dictionary)
        self.mel_out = Linear(self.hidden_size, hparams['audio_num_mel_bins'])
        self.wav_decoder = WavDecoder(wav_decoder_type)

    def forward(self, src_tokens, mel2ph=None, m_start_ends=None,
                f0=None, uv=None, energy=None, infer=False,
                skip_wav_decoder=False):
        encoder_outputs = self.fs_encoder(src_tokens, mel2ph, f0=f0, uv=uv, energy=energy, skip_decoder=infer)
        mel_out = encoder_outputs['mel_out'] if not infer else None
        fs_out = self.mel_out(encoder_outputs['decoder_inp'])
        if m_start_ends is not None:
            fs_out = torch.cat([m[s_e[0]:s_e[1]][None, ...] for m, s_e in zip(fs_out, m_start_ends)], 0)
        if not skip_wav_decoder:
            wavs_output = self.wav_decoder(fs_out)
            encoder_outputs.update({
                'wav': wavs_output,
                'mel_out': mel_out,
            })
        return encoder_outputs
        # {
        #     'cwt': encoder_outputs.get('cwt'),
        #     'mel2ph': encoder_outputs.get('mel2ph'),
        #     'dur': encoder_outputs.get('dur'),
        #     'pitch_pred': encoder_outputs.get('pitch_pred'),
        #     'pitch': encoder_outputs.get('pitch'),
        #     'pitch_coarse': encoder_outputs.get('pitch_coarse'),
        #     'energy_pred': encoder_outputs.get('energy_pred'),
        # }
