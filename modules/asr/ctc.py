import torch
from torch import nn

from modules.asr.base import Prenet
from modules.commons.common_layers import SinusoidalPositionalEmbedding
from modules.fastspeech.tts_modules import FFTBlocks, DEFAULT_MAX_SOURCE_POSITIONS


class CTC(torch.nn.Module):
    def __init__(
            self, reduce=True, blank=0
    ):
        super().__init__()
        reduction_type = "mean" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type, blank=blank)

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        th_pred = th_pred.log_softmax(2)
        loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        # Batch-size average
        loss = loss / th_pred.size(1)
        return loss

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.
        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = hs_pad.transpose(0, 1)

        # (B, L) -> (BxL,)
        ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])
        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens)
        return loss


class CTCASR(nn.Module):
    def __init__(self, dict_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_prenet = Prenet(80, hidden_size)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size, 0, init_size=self.max_source_positions,
        )
        self.embed_scale = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.encoder = FFTBlocks(hidden_size=hidden_size, num_layers=4)
        self.out_proj = nn.Linear(hidden_size, dict_size + 1)

    def forward(self, mel):
        """

        :param mel: [B, T_mel, 80]
        :param mel2ph: [B, T_mel]
        :return: P_ph: [B, T_mel, W]
        """
        h_prenet, x = self.in_prenet(mel)  # [B, T_mel, H]
        pos_embed = self.embed_scale * self.embed_positions(x[:, :, 0])
        h = self.encoder(x + pos_embed, return_hiddens=True)  # [L, B, T_mel, H]
        cls_out = self.out_proj(h[-1])  # [B, T_mel, W]
        return torch.cat([h_prenet, h], 0), cls_out
