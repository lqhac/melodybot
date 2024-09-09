import torch
from torch import nn


class Prenet(nn.Module):
    def __init__(self, in_dim=80, out_dim=256, kernel=5, n_layers=3, strides=None):
        super(Prenet, self).__init__()
        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=padding, stride=self.strides[l]),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim)
            ))
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """

        :param x: [B, T, 80]
        :return: [L, B, T, H], [B, T, H]
        """
        padding_mask = x.abs().sum(-1).eq(0).data  # [B, T]
        nonpadding_mask_TB = 1 - padding_mask.float()[:, None, :]  # [B, 1, T]
        x = x.transpose(1, 2)
        hiddens = []
        for i, l in enumerate(self.layers):
            nonpadding_mask_TB = nonpadding_mask_TB[:, :, ::self.strides[i]]
            x = l(x) * nonpadding_mask_TB
        hiddens.append(x)
        hiddens = torch.stack(hiddens, 0)  # [L, B, H, T]
        hiddens = hiddens.transpose(2, 3)  # [L, B, T, H]
        x = self.out_proj(x.transpose(1, 2))  # [B, T, H]
        x = x * nonpadding_mask_TB.transpose(1, 2)
        return hiddens, x
