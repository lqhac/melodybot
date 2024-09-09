import torch.nn as nn
from utils.hparams import hparams
from .layers import ResidualBlock


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, channels, out_dims, dec_dilations=None, dec_kernel_size=None):
        super(ConvBlocks, self).__init__()
        dec_kernel_size = 5 if dec_kernel_size is None else dec_kernel_size
        # receptive field is max 32
        dec_dilations = 4 * [1, 2, 4, 8] + [1] if dec_dilations is None else dec_dilations

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, dec_kernel_size, d, n=2,
                            norm_type=hparams['norm_type'], dropout=hparams['dropout'])
              for d in dec_dilations],
        )

        self.post_net1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        self.post_net2 = nn.Sequential(
            ResidualBlock(channels, dec_kernel_size, 1, n=2,
                          norm_type=hparams['norm_type'], dropout=hparams['dropout']),
            nn.Conv1d(channels, out_dims, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        x = x.transpose(1, 2)
        xx = self.res_blocks(x)
        x = self.post_net1(xx) + x
        x = self.post_net2(x)
        return x.transpose(1, 2)
