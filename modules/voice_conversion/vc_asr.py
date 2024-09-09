from functools import partial

from torch import nn
from modules.asr.base import Prenet
from modules.asr.seq2seq import TransformerASRDecoder
from modules.voice_conversion.spec_ae import ConvStacks
from modules.commons.common_layers import Embedding, Permute
from modules.fastspeech.tts_modules import FFTBlocks, RefEncoder
from utils.hparams import hparams


class VCASR(nn.Module):
    def __init__(self, dictionary, enc_layers=None, asr_dec_layers=None, mel_dec_layers=None):
        super().__init__()
        self.dictionary = dictionary
        self.asr_enc_layers = hparams['asr_enc_layers'] if enc_layers is None else enc_layers
        self.mel_dec_layers = hparams['mel_dec_layers'] if mel_dec_layers is None else mel_dec_layers
        self.asr_dec_layers = hparams['asr_dec_layers'] if asr_dec_layers is None else asr_dec_layers
        self.num_heads = 2
        self.hidden_size = hparams['hidden_size']
        self.n_mel_bins = hparams['audio_num_mel_bins']
        self.token_embed = Embedding(len(dictionary), self.hidden_size, 0)
        self.mel_prenet = Prenet(self.n_mel_bins, self.hidden_size, strides=[2, 2, 2])
        self.ref_encoder = RefEncoder(hparams['audio_num_mel_bins'], hparams['ref_hidden_stride_kernel'],
                                      ref_norm_layer=hparams['ref_norm_layer'])
        self.upsample_layer = nn.Sequential(
            # uplayer1
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
            # uplayer2
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
            # uplayer3
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2),
        )
        self.content_encoder = FFTBlocks(num_layers=self.asr_enc_layers, hidden_size=self.hidden_size)
        self.asr_decoder = TransformerASRDecoder(
            self.hidden_size, self.asr_dec_layers, hparams['dropout'], len(self.dictionary),
            num_heads=self.num_heads)
        self.pitch_embed = Embedding(300, self.hidden_size, 0)
        self.mel_out = nn.Linear(self.hidden_size, self.n_mel_bins)

        if hparams['arch'] == 'fft':
            self.mel_decoder = FFTBlocks(num_layers=self.mel_dec_layers, hidden_size=self.hidden_size)
        elif hparams['arch'] == 'conv':
            self.mel_decoder = ConvStacks(idim=self.hidden_size, odim=self.hidden_size, norm=hparams['norm_type'])
        elif hparams['arch'] == 'wn':
            from modules.parallel_wavegan.models import ParallelWaveGANGenerator
            h = 128
            wn_builder = partial(
                ParallelWaveGANGenerator, upsample_params={"upsample_scales": [1, 1]},
                aux_channels=h, aux_context_window=0, residual_channels=h, gate_channels=h * 2, skip_channels=h
            )
            self.mel_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, h),
                Permute(0, 2, 1), wn_builder(h, h, layers=10, stacks=2),
                Permute(0, 2, 1),
                nn.Linear(h, self.hidden_size),
            )

    def forward(self, mel_content, mel_ref, pitch, prev_tokens=None):
        ret = {}
        T = pitch.shape[1]
        h_content = self.content_encoder(self.mel_prenet(mel_content)[1])
        if prev_tokens is not None:
            ret['tokens'], ret['asr_attn'] = self.asr_decoder(self.token_embed(prev_tokens), h_content)
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:, :T]
        h_ref = self.ref_encoder(mel_ref)[:, None, :]
        h_pitch = self.pitch_embed(pitch)
        dec_inputs = h_content.detach() + h_ref + h_pitch
        ret['mel_out'] = self.mel_out(self.mel_decoder(dec_inputs))
        return ret

    def out2mel(self, out):
        return out
