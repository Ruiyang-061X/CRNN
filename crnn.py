from torch import nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nin, nh, nout):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(nin, nh, bidirectional=True)
        self.linear = nn.Linear(nh * 2, nout)

    def forward(self, x):
        x, _ = self.lstm(x)
        t, n, h = x.size()
        x = x.view(t * n, h)
        x = self.linear(x)
        x = x.view(t, n, -1)

        return x

class CRNN(nn.Module):

    def __init__(self, nc, nclass, nh):
        super(CRNN, self).__init__()
        nm = [64, 128, 256, 256, 512, 512, 512]
        ks = [3, 3, 3, 3, 3, 3, 2]
        ss = [1, 1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1, 0]
        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            nin = nc if i == 0 else nm[i - 1]
            nout = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=ks[i], stride=ss[i], padding=ps[i]))
            if batch_norm:
                cnn.add_module('batch_norm{0}'.format(i), nn.BatchNorm2d(num_features=nout))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(inplace=True))

        conv_relu(0)
        cnn.add_module('max_pool{0}'.format(0), nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module('max_pool{0}'.format(1), nn.MaxPool2d(2, 2))
        conv_relu(2, batch_norm=True)
        conv_relu(3)
        cnn.add_module('max_pool{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(4, batch_norm=True)
        conv_relu(5)
        cnn.add_module('max_pool{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(6, batch_norm=True)
        self.cnn = cnn
        rnn = nn.Sequential()
        rnn.add_module('bidirectional_lstm{0}'.format(0), BidirectionalLSTM(512, nh, nh))
        rnn.add_module('bidirectional_lstm{0}'.format(1), BidirectionalLSTM(nh, nh, nclass))
        self.rnn = rnn

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)

        return x