from modules.fastspeech.fs2 import FastSpeech2
from modules.commons.common_layers import *
from utils.hparams import hparams


class Fs2FreqAr(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        rnn_size = hparams['rnn_size']
        super().__init__(dictionary, out_dims=rnn_size)
        self.prenet = Linear(1, rnn_size)
        self.rnn = nn.LSTMCell(rnn_size, rnn_size)
        self.out_proj = Linear(rnn_size, 1)

    def forward(self, src_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False, tgt_mels=None):
        ret = super(Fs2FreqAr, self).forward(
            src_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy, skip_decoder)
        h = ret['mel_out']
        B, T, _ = h.shape
        h = h.reshape(B * T, -1)
        empty_mel_const = -6
        if tgt_mels is None:
            prev_freqs = [h.new_ones([B * T, 1]) * empty_mel_const]  # [1, B*T, 1]
        else:
            prev_mels = F.pad(tgt_mels[:, :, :-1], [1, 0], value=empty_mel_const)
            prev_freqs = prev_mels.reshape(B * T, 80, 1).transpose(0, 1)  # [80, B*T, 1]

        outputs = []
        state = None
        for i in range(80):
            inp = self.prenet(prev_freqs[i]) + h  # [B*T, H]
            hx, cx = state = self.rnn(inp, state)
            out = self.out_proj(hx)  # [B*T, 1]
            if tgt_mels is None:
                prev_freqs.append(out)
            outputs.append(out)
        outputs = torch.cat(outputs, 1)  # [B*T, 80]
        ret['mel_out'] = outputs.reshape(B, T, -1)
        return ret
