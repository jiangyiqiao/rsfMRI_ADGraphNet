import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net.utils.graph import Graph


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        # temporal kernel size
        padding = int((kernel_size - 1) / 2)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        x = self.tcn(x)
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size, coff_embedding=4, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        num_jpts = A.shape[-1]

        self.conv_d = nn.Conv2d(in_channels, out_channels, 1)

        if adaptive:
            self.PA = A
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.Conv2d(in_channels, inter_channels, 1)
            self.conv_b = nn.Conv2d(in_channels, inter_channels, 1)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            self.mask = nn.ParameterList([nn.Parameter(torch.ones(self.A.size()))])
        self.adaptive = adaptive

        if attention:
            # temporal attention
            padding = int((kernel_size - 1) / 2)
            self.conv_ta = nn.Conv1d(out_channels, 1, kernel_size, padding=padding)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv_d)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            if torch.cuda.is_available():
                dev = 0
                A = self.PA.cuda(dev)
            else:
                A = self.PA
            # self.inter_c=16
            # print(self.conv_a(x).shape) [32, 16, 50, 90]
            # print("A1.shape", A1.shape) [32, 90, 800]
            # print(self.conv_b(x).shape) [32, 16, 50, 90]
            # print("A2.shape", A2.shape) [32, 800, 90]
            # print("A1.shape", A1.shape) [32, 90, 90]
            # print("A1.shape", A1.shape)[32, 90, 90])
            # print("A2.shape", A2.shape) [32, 4500, 90]
            # print("z.shape", z.shape)[32, 64, 50, 90]
            A1 = self.conv_a(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b(x).view(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A + A1 * self.alpha
            A2 = x.view(N, C * T, V)
            z = self.conv_d(torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        else:
            A = self.A.cuda(x.get_device()) * self.mask
            A1 = A
            A2 = x.view(N, C * T, V)
            z = self.conv_d(torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn = unit_gcn(in_channels, out_channels, A, kernel_size=kernel_size,adaptive=adaptive, attention=attention)
        self.tcn = unit_tcn(out_channels, out_channels, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.attention:
            y = self.relu(self.tcn(self.gcn(x)) + self.residual(x))
        else:
            y = self.relu(self.tcn(self.gcn(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class, graph_args, in_channels=90,
                 drop_out=0.5, kernel_size=3, adaptive=True, attention=True):
        super(Model, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.l1 = TCN_GCN_unit(90, 64, A, kernel_size=kernel_size, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 128, A, kernel_size=kernel_size, stride=2, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(128, 256, A, kernel_size=kernel_size, stride=2, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(256, 256, A, kernel_size=kernel_size, adaptive=adaptive, attention=attention)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x