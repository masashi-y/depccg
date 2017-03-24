# requirements:
# pip install -e git://github.com/jekbradbury/chainer.git@raw-kernel

from chainer import cuda, Function, Variable, Chain
import chainer.links as L
import chainer.functions as F
import numpy as np

THREADS_PER_BLOCK = 32

class STRNNFunction(Function):

    def forward_cpu(self, inputs):
        f, z, hinit = inputs
        b, t, c = f.shape
        self.h = np.zeros((b, t + 1, c), dtype=np.float32)
        self.h[:, 0, :] = hinit
        t_size = f.shape[1]
        for i in range(t_size):
            self.h[:, i+1, :] = self.h[:, i, :] * f[:, i, :] + z[:, i, :]
        return self.h[:, 1:, :],

    def forward_gpu(self, inputs):
        f, z, hinit = inputs
        b, t, c = f.shape
        assert c % THREADS_PER_BLOCK == 0
        self.h = cuda.cupy.zeros((b, t + 1, c), dtype=np.float32)
        self.h[:, 0, :] = hinit
        cuda.raw('''
            #define THREADS_PER_BLOCK 32
            extern "C" __global__ void strnn_fwd(
                    const CArray<float, 3> f, const CArray<float, 3> z,
                    CArray<float, 3> h) {
                int index[3];
                const int t_size = f.shape()[1];
                index[0] = blockIdx.x;
                index[1] = 0;
                index[2] = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
                float prev_h = h[index];
                for (int i = 0; i < t_size; i++){
                    index[1] = i;
                    const float ft = f[index];
                    const float zt = z[index];
                    index[1] = i + 1;
                    float &ht = h[index];
                    prev_h = prev_h * ft + zt;
                    ht = prev_h;
                }
            }''', 'strnn_fwd')(
                (b, c // THREADS_PER_BLOCK), (THREADS_PER_BLOCK,),
                (f, z, self.h))
        return self.h[:, 1:, :],

    def backward_cpu(self, inputs, grads):
        f, z = inputs[:2]
        gh, = grads
        b, t, c = f.shape
        gz = np.zeros_like(gh)
        t_size = f.shape[1]
        gz[:, -1, :] = gh[:, -1, :]
        for i in range(t_size - 1, 0, -1):
            gz[:,  i-1, :] = gz[:, i, :] * f[:, i, :] + gh[:, i-1, :]
        gf = self.h[:, :-1, :] * gz
        ghinit = f[:, 0, :] * gz[:, 0, :]
        return gf, gz, ghinit

    def backward_gpu(self, inputs, grads):
        f, z = inputs[:2]
        gh, = grads
        b, t, c = f.shape
        gz = cuda.cupy.zeros_like(gh)
        cuda.raw('''
            #define THREADS_PER_BLOCK 32
            extern "C" __global__ void strnn_back(
                const CArray<float, 3> f, const CArray<float, 3> gh,
                CArray<float, 3> gz) {
                int index[3];
                const int t_size = f.shape()[1];
                index[0] = blockIdx.x;
                index[2] = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
                index[1] = t_size - 1;
                float &gz_last = gz[index];
                gz_last = gh[index];
                float prev_gz = gz_last;
                for (int i = t_size - 1; i > 0; i--){
                    index[1] = i;
                    const float ft = f[index];
                    index[1] = i - 1;
                    const float ght = gh[index];
                    float &gzt = gz[index];
                    prev_gz = prev_gz * ft + ght;
                    gzt = prev_gz;
                }
            }''', 'strnn_back')(
                (b, c // THREADS_PER_BLOCK), (THREADS_PER_BLOCK,),
                (f, gh, gz))
        gf = self.h[:, :-1, :] * gz
        ghinit = f[:, 0, :] * gz[:, 0, :]
        return gf, gz, ghinit


def strnn(f, z, h0):
    return STRNNFunction()(f, z, h0)


def attention_sum(encoding, query):
    alpha = F.softmax(F.batch_matmul(encoding, query, transb=True))
    alpha, encoding = F.broadcast(alpha[:, :, :, None],
                                  encoding[:, :, None, :])
    return F.sum(alpha * encoding, axis=1)


class Linear(L.Linear):

    def __call__(self, x):
        shape = x.shape
        if len(shape) == 3:
            x = F.reshape(x, (-1, shape[2]))
        y = super(Linear, self).__call__(x)
        if len(shape) == 3:
            y = F.reshape(y, (shape[0], shape[1], -1))
        return y


class QRNNLayer(Chain):

    def __init__(self, in_size, out_size, kernel_size=2, attention=False,
                 decoder=False):
        if kernel_size == 1:
            super(QRNNLayer, self).__init__(W=Linear(in_size, 3 * out_size))
        elif kernel_size == 2:
            super(QRNNLayer, self).__init__(W=Linear(in_size, 3 * out_size, nobias=True),
                             V=Linear(in_size, 3 * out_size))
        else:
            super(QRNNLayer, self).__init__(
                conv=L.ConvolutionND(1, in_size, 3 * out_size, kernel_size,
                                     stride=1, pad=kernel_size - 1))
        if attention:
            self.add_link('U', Linear(out_size, 3 * in_size))
            self.add_link('o', Linear(2 * out_size, out_size))
        self.in_size, self.size, self.attention = in_size, out_size, attention
        self.kernel_size = kernel_size

    def pre(self, x):
        dims = len(x.shape) - 1

        if self.kernel_size == 1:
            ret = self.W(x)
        elif self.kernel_size == 2:
            if dims == 2:
                xprev = Variable(
                    self.xp.zeros((self.batch_size, 1, self.in_size),
                                  dtype=np.float32), volatile='AUTO')
                xtminus1 = F.concat((xprev, x[:, :-1, :]), axis=1)
            else:
                xtminus1 = self.x
            ret = self.W(x) + self.V(xtminus1)
        else:
            ret = F.swapaxes(self.conv(
                F.swapaxes(x, 1, 2))[:, :, :x.shape[2]], 1, 2)

        if not self.attention:
            return ret

        if dims == 1:
            enc = self.encoding[:, -1, :]
        else:
            enc = self.encoding[:, -1:, :]
        return sum(F.broadcast(self.U(enc), ret))

    def init(self, encoder_c=None, encoder_h=None):
        self.encoding = encoder_c
        self.c, self.x = None, None
        if self.encoding is not None:
            self.batch_size = self.encoding.shape[0]
            if not self.attention:
                self.c = self.encoding[:, -1, :]

        if self.c is None or self.c.shape[0] < self.batch_size:
            self.c = Variable(self.xp.zeros((self.batch_size, self.size),
                                            dtype=np.float32), volatile='AUTO')

        if self.x is None or self.x.shape[0] < self.batch_size:
            self.x = Variable(self.xp.zeros((self.batch_size, self.in_size),
                                            dtype=np.float32), volatile='AUTO')

    def __call__(self, x):
        if not hasattr(self, 'encoding') or self.encoding is None:
            self.batch_size = x.shape[0]
            self.init()
        dims = len(x.shape) - 1
        f, z, o = F.split_axis(self.pre(x), 3, axis=dims)
        f = F.sigmoid(f)
        z = (1 - f) * F.tanh(z)
        o = F.sigmoid(o)

        if dims == 2:
            self.c = strnn(f, z, self.c[:self.batch_size])
        else:
            self.c = f * self.c + z

        if self.attention:
            context = attention_sum(self.encoding, self.c)
            self.h = o * self.o(F.concat((self.c, context), axis=dims))
        else:
            self.h = self.c * o

        self.x = x
        return self.h

    def reset_state(self):
        self.encoding = None

    def get_state(self):
        return F.concat((self.x, self.c, self.h), axis=1)

    def set_state(self, state):
        self.x, self.c, self.h = F.split_axis(
            state, (self.in_size, self.in_size + self.size), axis=1)

    state = property(get_state, set_state)
