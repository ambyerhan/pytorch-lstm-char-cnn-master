
import torch
import torch.nn as nn
from torch.autograd import Variable
from .layers import CHighwayLayer, CConvLayer


class CHighway(nn.Module):
    def __init__(self, idim, odim, nlayers, param_init = 0.):
        super(CHighway, self).__init__()

        self.nlayers = nlayers
        for layer in range(nlayers):
            name = "highway_layer_%d" % layer
            layer_input_size = idim if layer == 0 else odim

            setattr(self, name, CHighwayLayer(layer_input_size, odim, param_init))

    def forward(self, input):
        output = input
        for layer in range(self.nlayers):
            output = self.__getattr__("highway_layer_%d" % layer)(output)
        return output


class CConvModel(nn.Module):
    def __init__(self, idim, kernel_sizes, kernel_features, param_init = 0.):
        super(CConvModel, self).__init__()

        assert len(kernel_sizes) == len(kernel_features)
        self.kernel_sizes = kernel_sizes
        self.kernel_features = kernel_features

        for i, (ker_size, ker_feat) in enumerate(zip(kernel_sizes, kernel_features)):
            name = "kernel_size_%d" % i
            ker_size, ker_feat = (ker_size, ker_feat) # todo

            setattr(self, name, CConvLayer(idim, kernel_size = (1, ker_size), kernel_feature = ker_feat, param_init = param_init))

    def forward(self, input):
        s = input.size()
        input_ = torch.unsqueeze(input, 2) # <batch * maxlen, wdlen, 1, cedim>
        input = torch.transpose(input_, 1, 3) # <batch * maxlen, cedim, 1, wdlen>

        outputs = []
        wdlen = input.size(3)
        for i, ker in enumerate(self.kernel_sizes):
            reduce_len = wdlen - ker + 1
            output = self.__getattr__("kernel_size_%d" % i)(input, reduce_len)
            outputs.append(output)

        return torch.cat(outputs, 1)


class CNLM(nn.Module):
    def __init__(self, cvsize, cedim, wvsize, wedim, cnn_size, hdim,
                 kernel_sizes, kernel_features, nhlayers, nrlayers, droprate, tie_weight = False):
        super(CNLM, self).__init__()
        self.cvsize = cvsize
        self.cedim = cedim
        self.wvsize = wvsize
        self.wedim = wedim
        self.cnn_size = cnn_size
        self.hdim = hdim
        self.kernel_sizes = kernel_sizes
        self.kernel_featrues = kernel_features
        self.nhlayers = nhlayers
        self.nrlayers = nrlayers
        self.drop = droprate

        assert nhlayers >= 1

        self.drop = nn.Dropout(droprate)
        self.char_embed = nn.Embedding(cvsize, cedim, padding_idx = 0)
        self.conv = CConvModel(cedim, kernel_sizes, kernel_features, param_init = 0.)
        self.highway = CHighway(cnn_size, cnn_size, nhlayers, param_init = 0.)
        self.rnn = nn.LSTM(input_size = cnn_size, hidden_size = hdim, num_layers = nrlayers, dropout = droprate)
        self.linear = nn.Linear(hdim, wvsize)

        if False and tie_weight:
            if hdim != cedim:
                raise ValueError("When using the tied flag, edim must be equal to hdim")
            self.linear.weight = self.word_embed.weight

        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        self.char_embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.fill_(0.)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nrlayers, bsz, self.hdim).zero_()),
                Variable(weight.new(self.nrlayers, bsz, self.hdim).zero_()))

    def forward(self, input, hidden): # input <maxlen, batch, wdlen>
        maxlen, batch, wdlen = input.shape
        input_ = input.view(maxlen * batch, wdlen)

        emb = self.char_embed(input_) # todo <maxlen * batch> --> <maxlen * batch, wembed>
        emb = self.drop(emb)

        cnn = self.conv(emb) # <maxlen * batch, cedim> --> <maxlen * batch, cnn_size>

        h_ = self.highway(cnn) # todo <maxlen * batch, cnn_size> --> <maxlen * batch, cnn_size>
        h_ = h_.view(maxlen, batch, h_.size(-1)) # <maxlen, batch, cnn_size>
        output, hidden = self.rnn(h_, hidden) # todo <maxlen, batch, hdim>, <nlayer, batch, hdim>
        output = self.drop(output)

        decoded = output.view(output.size(0) * output.size(1), output.size(2)) # todo <maxlen * batch, hdim>
        decoded = self.linear(decoded) # todo <maxlen * batch, vsize>
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # todo <maxlen, batch, vsize>

        return decoded, hidden

