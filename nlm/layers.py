
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CHighwayLayer(nn.Module):
    def __init__(self, idim, odim, param_init = 0.):
        super(CHighwayLayer, self).__init__()
        self.linear_g = nn.Linear(idim, odim)
        self.linear_t = nn.Linear(idim, odim)
        self.bias = -2.0

        if param_init > 0.:
            self.linear_g.weight.data.uniform_(-param_init, param_init)
            self.linear_g.bias.data.uniform_(-param_init, param_init)
            self.linear_t.weight.data.uniform_(-param_init, param_init)
            self.linear_t.bias.data.uniform_(-param_init, param_init)

    def forward(self, input):
        """
        G = relu(x, Wg)
        T = sigmoid(x, Wt)

                                   |x, T == 0
        y = G * T + x * (1. - T) = |
                                   |G, T == 1
        """
        g = F.relu(self.linear_g(input))
        t = F.sigmoid(self.linear_t(input) + self.bias)
        output = t * g + (1. - t) * input

        return output


class CConvLayer(nn.Module):
    def __init__(self, idim, kernel_size, kernel_feature, param_init = 0.):
        super(CConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels = idim, out_channels = kernel_feature, kernel_size = kernel_size)

        if param_init > 0.:
            self.conv.weight.data.uniform_(-param_init, param_init)
            self.conv.bias.data.uniform_(-param_init, param_init)

    def forward(self, input, reduce_len):
        output = F.tanh(self.conv(input))
        output = F.max_pool2d(output, kernel_size = [1, reduce_len], stride = [1, 1])

        output = torch.squeeze(output, dim = 3)
        return torch.squeeze(output, dim = 2)

