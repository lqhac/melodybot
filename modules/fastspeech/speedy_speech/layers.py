import torch.nn as nn
import torch
from torch import tanh, sigmoid

from utils.hparams import hparams


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm = nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(16, channels)

        self.blocks = [
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation),
                nn.ReLU(),
                norm,
                nn.Dropout(dropout),
            )
            for i in range(n)
        ]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return x + self.blocks(x)


class Pad(nn.ZeroPad2d):
    def __init__(self, kernel_size, dilation):
        pad_total = dilation * (kernel_size - 1)
        begin = pad_total // 2
        end = pad_total - begin

        super(Pad, self).__init__((begin, end, begin, end))


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""

    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))

        if causal:
            super(ZeroTemporalPad, self).__init__((total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((begin, end))


class WaveResidualBlock(nn.Module):
    """A residual gated block based on WaveNet
                        |-------------------------------------------------------------|
                        |                                                             |
                        |                        |-- conv -- tanh --|                 |
          residual ->  -|--(pos_enc)--(dropout)--|                  * ---|--- 1x1 --- + --> residual
                                                 |-- conv -- sigm --|    |
                                                                        1x1
                                                                         |
          -------------------------------------------------------------> + ------------> skip
    """

    def __init__(self, residual_channels, block_channels, kernel_size, dilation_rate, causal=True, dropout=False,
                 skip_channels=False):
        """
        :param residual_channels: Num. of channels for resid. connections between wave blocks
        :param block_channels: Num. of channels used inside wave blocks
        :param kernel_size: Num. of branches for each convolution kernel
        :param dilation_rate: Hom much to dilate inputs before applying gate and filter
        :param causal: If causal, input is zero padded from the front, else both sides are zero padded equally
        :param dropout: If dropout>0, apply dropout on the input to the block gates (not the residual connection)
        :param skip_channels: If >0, return also skip (batch, time, skip_channels)
        """
        super(WaveResidualBlock, self).__init__()

        self.pad = ZeroTemporalPad(kernel_size, dilation_rate, causal=causal)
        self.causal = causal
        self.receptive_field = dilation_rate * (kernel_size - 1) + 1

        # tanh and sigmoid applied in forward
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None
        self.filter = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.gate = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)

        self.conv1x1_resid = Conv1d(block_channels, residual_channels, 1)
        self.conv1x1_skip = Conv1d(block_channels, skip_channels, 1) if skip_channels else None

        self.tensor_q = None
        self.generate = False

    def forward(self, residual):
        """Feed residual through the WaveBlock

        Allows layer-level caching for faster sequential inference.
        See https://github.com/tomlepaine/fast-wavenet for similar tensorflow implementation and original paper.

        Non - causal version does not support iterative generation for obvious reasons.
        WARNING: generating must be called before each generated sequence!
        Otherwise there will be an error due to stored queue from previous run.

        RuntimeError: Trying to backward through the graph a second time,
        but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

        :param residual: Residual from previous block or from input_conv, (batch_size, channels, time_dim)
        :return: residual, skip
        """

        if self.generate and self.causal:
            if self.tensor_q is None:
                x = self.pad(residual)
                self.tensor_q = x[:, -self.receptive_field:, :].detach()
            else:
                assert residual.shape[
                           1] == 1, f'Expected residual.shape[1] == 1 during generation, but got residual.shape[1]={residual.shape[1]}'

                x = torch.cat((self.tensor_q, residual), dim=1)[:, -self.receptive_field:, :]
                self.tensor_q = x.detach()
        else:
            x = self.pad(residual)

        if self.dropout is not None:
            x = self.dropout(x)
        filter = tanh(self.filter(x))
        gate = sigmoid(self.gate(x))
        out = filter * gate
        residual = self.conv1x1_resid(out) + residual

        if self.conv1x1_skip is not None:
            return residual, self.conv1x1_skip(out)
        else:
            return residual

    def generating(self, mode):
        """Call before and after generating"""
        self.generate = mode
        self.reset_queue()

    def reset_queue(self):
        self.tensor_q = None
