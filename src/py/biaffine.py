import math
import numpy as np

from chainer import cuda
from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import functions as F


class Biaffine(link.Link):


    def __init__(self, in_size, wscale=1,
                 initialW=None, initial_bias=None):
        super(Biaffine, self).__init__()

        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))

        self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.add_param('W', (in_size + 1, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x1, x2):
        return F.matmul(
                F.concat([x1, np.ones((x1.shape[0], 1), 'f')]),
                F.matmul(self.W, F.transpose(x2)))
