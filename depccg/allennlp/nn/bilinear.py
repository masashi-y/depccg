import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module


class BilinearWithBias(Module):
    def __init__(self, in1_features, in2_features, out_features):
        super(BilinearWithBias, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.W = Parameter(torch.Tensor(out_features, in1_features, in2_features))
        self.V1 = Parameter(torch.Tensor(out_features, in1_features))
        self.V2 = Parameter(torch.Tensor(out_features, in2_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        result = F.bilinear(input1, input2, self.W, self.bias)
        result += F.linear(input1, self.V1, None)
        result += F.linear(input2, self.V2, None)
        return result

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
