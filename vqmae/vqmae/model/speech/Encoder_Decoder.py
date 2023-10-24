import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Tanh(),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.tanh(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self.norm1 = nn.BatchNorm1d(num_hiddens // 2)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self.norm2 = nn.BatchNorm1d(num_hiddens)
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self.norm3 = nn.BatchNorm1d(num_hiddens)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self.norm1(x)
        x = torch.tanh(x)

        x = self._conv_2(x)
        x = self.norm2(x)
        x = torch.tanh(x)

        x = self._conv_3(x)
        x = self.norm3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self.norm1_dec = nn.BatchNorm1d(num_hiddens)

        self._conv_trans_2_bis = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
        self.norm2_dec = nn.BatchNorm1d(num_hiddens // 2)

        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                out_channels=1,
                                                kernel_size=3,
                                                stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = self.norm1_dec(x)
        x = torch.tanh(x)

        x = self._conv_trans_2_bis(x)
        x = self.norm2_dec(x)
        x = torch.tanh(x)


        return torch.exp(self._conv_trans_2(x))
        # return torch.relu(self._conv_trans_2(x))


if __name__ == '__main__':
    encoder = Encoder(1, 8, 2, 16)
    decoder = Decoder(8, 32, 2, 16)
    x = torch.randn(1, 1, 513)
    x_en = encoder(x)
    print(x_en.shape)
    x_dec = decoder(x_en)
