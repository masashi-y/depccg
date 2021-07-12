import math
import numpy as np

from chainer import cuda
from chainer import link
from chainer import functions as F
from chainer import initializer
from chainer.initializers.normal import HeNormal
from chainer.initializers.constant import Constant
from chainer.initializers.constant import Identity


def _get_initializer(initializer, scale=1.0):
    if initializer is None:
        return HeNormal(scale / np.sqrt(2))
    if np.isscalar(initializer):
        return Constant(initializer * scale)
    if isinstance(initializer, np.ndarray):
        return Constant(initializer * scale)

    assert callable(initializer)
    if scale == 1.0:
        return initializer
    return _ScaledInitializer(initializer, scale)


class _ScaledInitializer(initializer.Initializer):

    def __init__(self, initializer, scale=1.0):
        self.initializer = initializer
        self.scale = scale
        dtype = getattr(initializer, 'dtype', None)
        super(Identity, self).__init__(dtype)

    def __call__(self, array):
        self.initializer(array)
        array *= self.scale


class Biaffine(link.Link):

    def __init__(self, in_size, wscale=1,
                 initialW=None, initial_bias=None):
        super(Biaffine, self).__init__()

        self._W_initializer = _get_initializer(
            initialW, math.sqrt(wscale))

        self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.add_param('W', (in_size + 1, in_size),
                       initializer=self._W_initializer)

    def forward_one(self, x1, x2):
        xp = cuda.get_array_module(x1.data)
        return F.matmul(
            F.concat([x1, xp.ones((x1.shape[0], 1), 'f')]),  # (slen, hidden+1)
            F.matmul(self.W, x2, transb=True))  # (hidden+1, hidden) * (slen, hidden)^T

    def forward_batch(self, x1, x2):
        xp = cuda.get_array_module(x1.data)
        batch, slen, hidden = x2.shape
        return F.batch_matmul(
            # (batch, slen, hidden+1)
            F.concat([x1, xp.ones((batch, slen, 1), 'f')], 2),
            F.reshape(F.linear(F.reshape(x2, (batch * slen, -1)), self.W),
                      (batch, slen, -1)), transb=True)

    def __call__(self, x1, x2):
        dim = len(x1.shape)
        if dim == 3:
            return self.forward_batch(x1, x2)
        elif dim == 2:
            return self.forward_one(x1, x2)
        else:
            raise RuntimeError()


class Bilinear(link.Link):

    # chainer.links.Bilinear may have some problem with GPU
    # and results in nan with batches with big size

    def __init__(self, in_size1, in_size2, out_size, wscale=1,
                 initialW=None, initial_bias=None, bias=0):
        super(Bilinear, self).__init__()

        self._W_initializer = _get_initializer(
            initialW, math.sqrt(wscale))
        if initial_bias is None:
            initial_bias = bias
        self.bias_initializer = _get_initializer(initial_bias)

        # same parameters as chainer.links.Bilinear
        # so that both can use serialized parameters of the other
        self.add_param('W', (in_size1, in_size2, out_size),
                       initializer=self._W_initializer)
        self.add_param('V1', (in_size1, out_size),
                       initializer=self._W_initializer)
        self.add_param('V2', (in_size2, out_size),
                       initializer=self._W_initializer)
        self.add_param('b', out_size,
                       initializer=self.bias_initializer)
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.out_size = out_size

    def __call__(self, e1, e2):
        ele2 = F.reshape(
            F.batch_matmul(e1[:, :, None], e2[:, None, :]), (-1, self.in_size1 * self.in_size2))

        res = F.matmul(ele2,
                       F.reshape(self.W, (self.in_size1 * self.in_size2, self.out_size))) + \
            F.matmul(e1, self.V1) + \
            F.matmul(e2, self.V2)

        res, bias = F.broadcast(res, self.b)
        return res + bias
