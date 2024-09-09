from modules.commons.common_layers import ConvNorm
from modules.commons.common_layers import *


class ConvBlock(nn.Module):
    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=1, norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        hiddens = []
        for f in self.conv:
            x = x + f(x)  # (B, C, Tmax)
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)  # (B, Tmax, H)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)  # [B, L, C, T]
            return x, hiddens
        return x


class ConvLSTMStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=3, norm='gn', dropout=0):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=1, norm=norm, dropout=dropout))
        self.lstm = nn.LSTM(n_chans, n_chans, 1, batch_first=True, bidirectional=True)
        self.out_proj = Linear(n_chans * 2, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x = x + f(x)  # (B, C, Tmax)
        x = x.transpose(1, -1)
        x, _ = self.lstm(x)  # (B, Tmax, H*2)
        x = self.out_proj(x)  # (B, Tmax, H)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualLayer, self).__init__()
        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        """

        :param input: [B, H, T]
        :return: input: [B, H, T]
        """
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class ConvGLUStacks(nn.Module):
    def __init__(self, idim=80, n_layers=3, n_chans=256, odim=32, kernel_size=5, dropout=0):
        super().__init__()
        self.convs = []
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.convs.append(
                nn.Sequential(ResidualLayer(
                    n_chans, n_chans, kernel_size, kernel_size // 2),
                    nn.Dropout(dropout)
                ))
        self.convs = nn.Sequential(*self.convs)
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        x = self.convs(x)  # (B, C, Tmax)
        x = x.transpose(1, -1)
        x = self.out_proj(x)  # (B, Tmax, H)
        return x
