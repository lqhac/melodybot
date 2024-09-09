from modules.commons.common_layers import *
from modules.fastspeech.fs2 import FastSpeech2, FS_DECODERS
from modules.parallel_wavegan.layers import Stretch2d
from utils.hparams import hparams


class MsDecoder(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.dec_proj = nn.Linear(c_in + self.hidden_size, self.hidden_size)
        self.dec = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_proj = nn.Linear(self.hidden_size, c_out)

    def forward(self, x, c=None):
        input_masks = c.abs().sum(-1).ne(0).data.float()
        if c is not None:
            x = self.dec_proj(torch.cat([x, c], -1))  # [B, T, H]
        x = x * input_masks[:, :, None]
        x = self.dec(x)
        x = self.out_proj(x)
        return x


class Fs2Ms(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims=out_dims)
        del self.decoder
        del self.mel_out
        self.n_scale = hparams['n_scale']
        self.mel_pool1d = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=4, stride=2, padding=1)
        self.in_dims = self.out_dims = [80, 40, 20]
        if hparams['mel_mix_loss'] != 'none':
            self.out_dims = [x * hparams['mix_num'] * 3 for x in self.out_dims]
        self.mel_in_proj = Linear(self.hidden_size, self.hidden_size)
        self.mel_decoders = nn.ModuleList()
        for c_in, c_out in zip(self.in_dims[:self.n_scale], self.out_dims[:self.n_scale]):
            self.mel_decoders.append(MsDecoder(c_in, c_out))
        self.mel_stretch = Stretch2d(2, 2)

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret):
        T_mel = decoder_inp.shape[1]
        mod = 2 ** (self.n_scale - 1)
        decoder_inp = decoder_inp[:, :(T_mel // mod) * mod]
        ret['f0_denorm'] = ret['f0_denorm'][:, :(T_mel // mod) * mod]
        ret['mel2ph'] = ret['mel2ph'][:, :(T_mel // mod) * mod]
        nonpadding = (decoder_inp.abs().sum(-1) > 0).float().data
        decoder_aux = self.mel_in_proj(decoder_inp)  # [B, T, H]
        B, T, H = decoder_aux.shape
        decoder_aux = decoder_aux * nonpadding[:, :, None]
        decoder_auxs = [decoder_aux]
        for i in range(1, self.n_scale):
            decoder_aux_ = self.mel_pool1d(decoder_auxs[-1].transpose(1, 2)).transpose(1, 2)
            decoder_aux_ = decoder_aux_ * nonpadding[:, ::2 ** i, None]
            decoder_auxs.append(decoder_aux_)
        n_reduction = 2 ** (self.n_scale - 1)
        decoder_inp = decoder_aux.new_zeros(B, T // n_reduction, 80 // n_reduction).normal_(0, 1)
        for i in range(self.n_scale - 1, -1, -1):
            ret[f'mel_p{i}'] = mel = self.mel_decoders[i](decoder_inp, decoder_auxs[i])
            decoder_inp = self.mel_stretch(self.out2mel(mel)[:, None])[:, 0]
        return ret[f'mel_p0']
