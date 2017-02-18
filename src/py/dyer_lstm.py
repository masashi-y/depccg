
from chainer import Variable
import chainer
import math
import chainer.functions as F
# import chainer

def _dyer_init(indim, outdim):
    return math.sqrt(6) / math.sqrt(float(indim) + float(outdim))

class DyerLSTM(chainer.Chain):
    def __init__(self, insize, outsize):
        self.insize = insize
        self.outsize = outsize
        self.h = None
        self.c = None
        self.peep_dim = insize + 2 * outsize
        super(DyerLSTM, self).__init__(
                linear_in=F.Linear(self.peep_dim, outsize, bias=0.25,
                    wscale=_dyer_init(self.peep_dim, outsize)),
                linear_c=F.Linear(insize + outsize, outsize,
                    wscale=_dyer_init(insize + outsize, outsize)),
                linear_out=F.Linear(self.peep_dim, outsize,
                    wscale=_dyer_init(self.peep_dim, outsize))
                )

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, xs):
        """
        xs: (batchsize, hidden_dim)
        """

        if self.h is not None:
            h = self.h
            c = self.c
        else:
            xp = chainer.cuda.get_array_module(xs.data)
            batchsize = xs.shape[0]
            h = Variable(xp.zeros((batchsize, self.outsize), 'f'), volatile='AUTO')
            c = Variable(xp.zeros((batchsize, self.outsize), 'f'), volatile='AUTO')

        in_gate = F.sigmoid(self.linear_in(F.concat([xs, h, c])))
        new_in = F.tanh(self.linear_c(F.concat([xs, h])))
        self.c = in_gate * new_in + (1. - in_gate) * c
        out_gate = F.sigmoid(self.linear_out(F.concat([xs, h, self.c])))
        self.h = F.tanh(self.c) * out_gate
        return self.h

