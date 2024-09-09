import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.parallel_wavegan.layers import Conv1d


class Discriminator2DFactory(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128, sn=False):
        super(Discriminator2DFactory, self).__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)

        def discriminator_block(in_filters, out_filters, bn=True):
            """
            Input: (B, in, 2H, 2W)
            Output:(B, out, H,  W)
            """
            conv = nn.Conv2d(in_filters, out_filters, kernel, (2, 2), padding)
            if sn:
                conv = nn.utils.spectral_norm(conv)
            block = [
                conv,  # padding = kernel//2
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block = nn.Sequential(*block)
            return block

        self.model = nn.ModuleList([
            discriminator_block(c_in, hidden_size, bn=False),
            discriminator_block(hidden_size, hidden_size),
            discriminator_block(hidden_size, hidden_size),
        ])

        # The height and width of downsampled image
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)

    def forward(self, x):
        """

        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.model:
            x = l(x)
            h.append(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        return validity, h


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, cond_size=0, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128, sn=False):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths

        self.conv_layers = nn.ModuleList()
        if cond_size > 0:
            self.cond_proj_layers = nn.ModuleList()
            self.mel_proj_layers = nn.ModuleList()
        for time_length in time_lengths:
            conv_layer = [
                Discriminator2DFactory(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size, sn=sn)
            ]
            self.conv_layers += conv_layer
            if cond_size > 0:
                self.cond_proj_layers.append(nn.Linear(cond_size, freq_length))
                self.mel_proj_layers.append(nn.Linear(freq_length, freq_length))

    def forward(self, x, x_len, cond=None, start_frames_wins=None, reduction='sum'):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.conv_layers)
        h = []
        for i, start_frames in zip(range(len(self.conv_layers)), start_frames_wins):
            x_clip, c_clip, start_frames = self.clip(
                x, cond, x_len, self.win_lengths[i], start_frames)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            if cond is not None:
                x_clip = self.mel_proj_layers[i](x_clip)  # (B, 1, win_length, C)
                c_clip = self.cond_proj_layers[i](c_clip)[:, None]  # (B, 1, win_length, C)
                x_clip = x_clip + c_clip
            x_clip, h_ = self.conv_layers[i](x_clip)
            h += h_
            validity.append(x_clip)
        if reduction == 'sum':
            validity = sum(validity)  # [B]
        elif reduction == 'stack':
            validity = torch.stack(validity, -1)  # [B, W_L]
        return validity, start_frames_wins, h

    def clip(self, x, cond, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        T_start = 0
        T_end = x_len.max() - win_length
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame: start_frame + win_length]
        c_batch = cond[:, start_frame: start_frame + win_length] if cond is not None else None
        return x_batch, c_batch, start_frames


class Discriminator(nn.Module):
    def __init__(self, time_lengths=[32, 64, 128], freq_length=80, cond_size=0, kernel=(3, 3), c_in=1,
                 hidden_size=128, sn=False, reduction='sum'):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.cond_size = cond_size
        self.reduction = reduction
        self.discriminator = MultiWindowDiscriminator(
            freq_length=freq_length,
            time_lengths=time_lengths,
            kernel=kernel,
            c_in=c_in, hidden_size=hidden_size, sn=sn
        )
        if cond_size > 0:
            self.cond_disc = MultiWindowDiscriminator(
                freq_length=freq_length,
                time_lengths=time_lengths,
                cond_size=cond_size,
                kernel=kernel,
                c_in=c_in, hidden_size=hidden_size, sn=sn
            )

    def forward(self, x, cond=None, return_y_only=True, start_frames_wins=None):
        """

        :param x: [B, T, 80]
        :param cond: [B, T, cond_size]
        :param return_y_only:
        :return:
        """
        reduction = self.reduction
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x_len = x.sum([1, -1]).ne(0).int().sum([1, -1])
        pad_len = max(self.time_lengths) - x_len.min().item()
        if pad_len >= 0:
            x = F.pad(x, [0, 0, math.ceil(pad_len) + 1, 0], value=-4)
            if cond is not None:
                cond = F.pad(cond, [0, 0, math.ceil(pad_len) + 1, 0], value=0)
            x_len = x.sum([1, -1]).ne(0).int().sum([1, -1])
        y, start_frames_wins, h = self.discriminator(
            x, x_len, start_frames_wins=start_frames_wins, reduction=reduction)
        if return_y_only:
            return y

        ret = {'y': y, 'h': h}
        if self.cond_size > 0:
            ret['y_c'], start_frames_wins, ret['h_c'] = self.cond_disc(
                x, x_len, cond, start_frames_wins=start_frames_wins, reduction=reduction)
        else:
            ret['y_c'], start_frames_wins, ret['h_c'] = None, None, []
        ret['start_frames_wins'] = start_frames_wins
        return ret


class JCU_Discriminator(nn.Module):
    def __init__(self):
        super(JCU_Discriminator, self).__init__()
        self.mel_conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(80, 128, kernel_size=2, stride=1)),
            nn.LeakyReLU(0.2, True),
        )
        x_conv = [nn.ReflectionPad1d(7),
                  nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=7, stride=1)),
                  nn.LeakyReLU(0.2, True),
                  ]
        x_conv += [
            nn.utils.weight_norm(nn.Conv1d(
                16,
                64,
                kernel_size=41,
                stride=4,
                padding=4 * 5,
                groups=16 // 4,
            )
            ),
            nn.LeakyReLU(0.2),
        ]
        x_conv += [
            nn.utils.weight_norm(nn.Conv1d(
                64,
                128,
                kernel_size=21,
                stride=2,
                padding=2 * 5,
                groups=64 // 4,
            )
            ),
            nn.LeakyReLU(0.2),
        ]
        self.x_conv = nn.Sequential(*x_conv)
        self.mel_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )
        self.mel_conv3 = nn.utils.weight_norm(nn.Conv1d(
            128, 1, kernel_size=3, stride=1, padding=1
        ))

        self.x_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )
        self.x_conv3 = nn.utils.weight_norm(nn.Conv1d(
            128, 1, kernel_size=3, stride=1, padding=1
        ))

    def forward(self, x, mel):
        out = self.mel_conv(mel)
        out1 = self.x_conv(x)
        out = torch.cat([out, out1], dim=2)
        out = self.mel_conv2(out)
        cond_out = self.mel_conv3(out)
        out1 = self.x_conv2(out1)
        uncond_out = self.x_conv3(out1)
        return uncond_out, cond_out


class WaveNetDiscriminator(torch.nn.Module):
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=3,
                 layers=8,
                 conv_channels=64,
                 dilation_factor=2,
                 bias=True,
                 use_weight_norm=True,
                 cond_size=0
                 ):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(WaveNetDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        self.n_share_layers = layers // 2
        self.n_split_layers = layers // 2
        if cond_size > 0:
            self.cond_proj_layer = nn.Sequential(
                nn.Conv1d(cond_size, conv_channels, 5, padding=2), nn.LeakyReLU(0.2))
        self.share_layers = self.get_conv_layers(
            self.n_share_layers, dilation_factor, conv_channels, kernel_size, bias, in_channels)
        self.cond_layers = self.get_conv_layers(
            self.n_split_layers, dilation_factor, conv_channels, kernel_size, bias, conv_channels,
            out_channels=out_channels)
        self.nocond_layers = self.get_conv_layers(
            self.n_split_layers, dilation_factor, conv_channels, kernel_size, bias, conv_channels,
            out_channels=out_channels)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def get_conv_layers(self, layers, dilation_factor,
                        conv_channels, kernel_size, bias, in_channels,
                        out_channels=None):
        conv_layers = []
        if out_channels is not None:
            layers = layers - 1
        for i in range(layers):
            dilation = i if dilation_factor == 1 else dilation_factor ** i
            conv_in_channels = in_channels if i == 0 else conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = nn.Sequential(
                Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                nn.LeakyReLU(0.2)
            )
            conv_layers.append(conv_layer)
        if out_channels is not None:
            padding = (kernel_size - 1) // 2
            last_conv_layer = Conv1d(
                conv_in_channels, out_channels,
                kernel_size=kernel_size, padding=padding, bias=bias)
            conv_layers += [last_conv_layer]
        return nn.ModuleList(conv_layers)

    def forward(self, x, cond=None, return_y_only=True, start_frames_wins=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: (B, T), (B, T)

        """
        x = x.transpose(1, 2)
        cond = self.cond_proj_layer(cond.transpose(1, 2))
        h = []
        h_c = []
        for i, f in enumerate(self.share_layers):
            x = f(x)
            h.append(x)
        x_cond = x + cond
        for i, f in enumerate(self.cond_layers):
            x_cond = f(x_cond)
            h_c.append(x_cond)
        for i, f in enumerate(self.nocond_layers):
            x = f(x)
            h.append(x)
        return {'y': x[:, 0], 'y_c': x_cond[:, 0], 'h': h, 'h_c': h_c, 'start_frames_wins': None}

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
