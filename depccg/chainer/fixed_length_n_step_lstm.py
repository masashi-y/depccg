import itertools

import numpy
import six
import binascii
import time
import os

from chainer.links import NStepLSTM
import chainer.functions as F

from chainer import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.noise import dropout
from chainer.functions.connection.n_step_lstm import NStepLSTM as NStepLSTMFunction


# if cuda.cudnn_enabled:
#     cudnn = cuda.cudnn
#     libcudnn = cuda.cudnn
#     _cudnn_version = libcudnn.getVersion()


class DropoutRandomStates(object):

    def __init__(self, seed):
        self._states = None

        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        self._seed = seed

    def create_dropout_states(self, dropout):
        handle = cudnn.get_handle()
        if self._states is None:
            self._states = cudnn.DropoutStates(handle, self._seed)
        # TODO(unno): Make a method to set dropout instead of calling API
        cudnn.set_dropout_descriptor(self._states._desc, handle, dropout)

        return self._states


_random_states = {}


def get_random_state():
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = DropoutRandomStates(os.getenv('CHAINER_SEED'))
        _random_states[dev.id] = rs
    return rs


class DropoutRandomStates(object):

    def __init__(self, seed):
        self._states = None

        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        self._seed = seed

    def create_dropout_states(self, dropout):
        handle = cudnn.get_handle()
        if self._states is None:
            self._states = cudnn.DropoutStates(handle, self._seed)
        # TODO(unno): Make a method to set dropout instead of calling API
        cudnn.set_dropout_descriptor(self._states._desc, handle, dropout)

        return self._states


_random_states = {}


def get_random_state():
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = DropoutRandomStates(os.getenv('CHAINER_SEED'))
        _random_states[dev.id] = rs
    return rs


class PointerArray(object):

    def __init__(self, lst, back_pointer):
        self._value = numpy.array(lst, dtype=numpy.intp)
        # Store back_pointer to prevent the GC removes the original variable
        self._back_pointer = back_pointer

    @property
    def data(self):
        return self._value.ctypes.data


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementaiton is shuffled
    w = stack.stack(ws, axis=1)
    shape = w.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


class FixedLengthNStepLSTM(NStepLSTM):

    def __call__(self, hx, cx, xs, train=True):
        # xs: (length, batch, dim)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, ys = fixed_length_n_step_lstm(
            self.n_layers, self.dropout, hx, cx, ws, bs, xs,
            train=train)

        return hy, cy, ys


def _make_tensor_descriptor_array(xs, length):
    """Make an array of pointers denoting pointers of tensor descriptors.

    """
    descs = []
    batch_size = xs.shape[0] // length
    for i in range(length):
        x = xs[i*batch_size:(i+1)*batch_size]
        if x.ndim < 3:
            shape = x.shape + (1,) * (3 - x.ndim)
            x = x.reshape(shape)
        desc = cudnn.create_tensor_nd_descriptor(x)
        descs.append(desc)
    return PointerArray([d.value for d in descs], descs)


class FixedLengthNStepLSTMFunction(NStepLSTMFunction):

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        (hx, cx), inputs = _split(inputs, 2)
        ws, inputs = _split(inputs, self.n_layers * 8)
        bs, inputs = _split(inputs, self.n_layers * 8)
        xs = inputs[0]

        hx = cuda.cupy.ascontiguousarray(hx)
        cx = cuda.cupy.ascontiguousarray(cx)
        xs = cuda.cupy.ascontiguousarray(xs)

        #x_desc = cudnn.create_tensor_nd_descriptor(x_list[0][..., None])
        x_desc = cudnn.create_tensor_nd_descriptor(cuda.cupy.copy(xs[0][..., None]))
        # NOTE: copy is necessary here, since xs in the original NStepLSTM implementation is a different array from xs_list[0]

        length = xs.shape[0]
        n_units = hx.shape[2]

        #xs = cuda.cupy.concatenate(x_list, axis=0)
        xs = xs.reshape(-1, xs.shape[2])    # (length * batch_size, dim)
        #ys = cuda.cupy.empty((len(xs), n_units), dtype=xs.dtype)
        ys = cuda.cupy.empty((xs.shape[0], n_units), dtype=xs.dtype)

        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT, libcudnn.CUDNN_UNIDIRECTIONAL,
            libcudnn.CUDNN_LSTM, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        #c_x_descs = _make_tensor_descriptor_array(x_list)
        c_x_descs = _make_tensor_descriptor_array(xs, length)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for lin_layer_id in six.moves.range(8):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                m = mat.reshape(mat.size)
                m[...] = ws[layer * 8 + lin_layer_id].ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                b = bias.reshape(bias.size)
                b[...] = bs[layer * 8 + lin_layer_id]
        self.w = w
        self.w_desc = w_desc

        #sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        #y_list = cuda.cupy.split(ys, sections)

        #c_y_descs = _make_tensor_descriptor_array(y_list)
        c_y_descs = _make_tensor_descriptor_array(ys, length)
        hy = cuda.cupy.empty_like(hx)
        cy = cuda.cupy.empty_like(cx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)
        cy_desc = cudnn.create_tensor_nd_descriptor(cy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace

        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs

        #return tuple([hy, cy] + y_list)
        return hy, cy, ys   # NOTE: ys has shape (-1, n_units)

    def backward(self, inputs, grads):
        (hx, cx), inputs = _split(inputs, 2)
        ws, inputs = _split(inputs, self.n_layers * 8)
        bs, inputs = _split(inputs, self.n_layers * 8)
        #x_list = inputs
        xs = inputs[0]

        hx = cuda.cupy.ascontiguousarray(hx)
        cx = cuda.cupy.ascontiguousarray(cx)
        xs = cuda.cupy.ascontiguousarray(xs)

        #dhy, dcy = grads[:2]
        #dy_list = list(grads[2:])
        dhy, dcy, dys = grads
        dys = cuda.cupy.ascontiguousarray(dys)
        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)
        if dcy is None:
            dcy = cuda.cupy.zeros_like(cx)
        #for i in six.moves.range(len(dys)):
        #    if dy_list[i] is None:
        #        dy_list[i] = cuda.cupy.zeros_like(x_list[i])

        #xs = cuda.cupy.concatenate(x_list, axis=0)
        length = xs.shape[0]
        batch_size = xs.shape[1]
        xs = xs.reshape(-1, xs.shape[2])
        #length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)
        dcx = cuda.cupy.empty_like(cx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)
        dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)

        #c_dy_descs = _make_tensor_descriptor_array(dy_list)
        c_dy_descs = _make_tensor_descriptor_array(dys, length)
        #dys = cuda.cupy.concatenate(dy_list, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)
        dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)

        dxs = cuda.cupy.empty_like(xs)
        #sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        #dx_list = cuda.cupy.split(dxs, sections, 0)
        #c_dx_descs = _make_tensor_descriptor_array(dx_list)
        c_dx_descs = _make_tensor_descriptor_array(dxs, length)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
            dcy_desc.value, dcy.data.ptr, self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr, cx_desc.value, cx.data.ptr,
            c_dx_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
            dcx_desc.value, dcx.data.ptr, workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.zeros_like(self.w)
        dw_desc = cudnn.create_tensor_nd_descriptor(dw)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc.value, length,
            self.c_x_descs.data, xs.data.ptr,
            hx_desc.value, hx.data.ptr, self.c_y_descs.data, self.ys.data.ptr,
            workspace.data.ptr, work_size, dw_desc.value, dw.data.ptr,
            self.reserve_space.data.ptr, self.reserve_space.size)

        #dx = dx_list[0]
        dx = dxs[:batch_size]
        dx = dx.reshape(dx.shape + (1,))
        dx_desc = cudnn.create_tensor_nd_descriptor(dx)
        dws = [cuda.cupy.empty_like(w) for w in ws]
        dbs = [cuda.cupy.empty_like(b) for b in bs]
        for layer in six.moves.range(self.n_layers):
            for lin_layer_id in six.moves.range(8):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dws[layer * 8 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = mat.ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dbs[layer * 8 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = bias.ravel()

        dxs = dxs.reshape((length, batch_size, xs.shape[1]))

        #return tuple([dhx, dcx] + dws + dbs + dx_list)
        return tuple([dhx, dcx] + dws + dbs + [dxs])


def fixed_length_n_step_lstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, train=True,
        ):

    xp = cuda.get_array_module(hx, hx.data)

    if False:
    # if xp is not numpy and cuda.cudnn_enabled and _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, cx),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            (xs,)))
        rnn = FixedLengthNStepLSTMFunction(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy, cy, ys = ret
        _, batch_size, dim = hy.shape
        ys_reshape = F.reshape(ys, (-1, batch_size, dim))   # (length, batch, dim)
        return hy, cy, ys_reshape

    else:
        hx = split_axis.split_axis(hx, n_layers, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]
        cx = split_axis.split_axis(cx, n_layers, axis=0, force_tuple=True)
        cx = [reshape.reshape(c, c.shape[1:]) for c in cx]

        xws = [_stack_weight([w[2], w[0], w[1], w[3]]) for w in ws]
        hws = [_stack_weight([w[6], w[4], w[5], w[7]]) for w in ws]
        xbs = [_stack_weight([b[2], b[0], b[1], b[3]]) for b in bs]
        hbs = [_stack_weight([b[6], b[4], b[5], b[7]]) for b in bs]

        ys = []
        for x in xs:
            batch = x.shape[0]
            h_next = []
            c_next = []
            for layer in six.moves.range(n_layers):
                h = hx[layer]
                c = cx[layer]
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                    c, c_rest = split_axis.split_axis(c, [batch], axis=0)
                else:
                    h_rest = None

                x = dropout.dropout(x, ratio=dropout_ratio)
                h = dropout.dropout(h, ratio=dropout_ratio)
                lstm_in = linear.linear(x, xws[layer], xbs[layer]) + \
                          linear.linear(h, hws[layer], hbs[layer])

                c_bar, h_bar = lstm.lstm(c, lstm_in)
                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                    c = concat.concat([c_bar, c_rest], axis=0)
                else:
                    h = h_bar
                    c = c_bar
                h_next.append(h)
                c_next.append(c)
                x = h_bar
            hx = h_next
            cx = c_next
            ys.append(x)

        hy = stack.stack(hx)
        cy = stack.stack(cx)
        #return hy, cy, tuple(ys)
        ys_concat = F.concat(ys, axis=0)
        ys_reshape = F.reshape(ys_concat, (-1, ys[0].shape[0], ys[0].shape[1]))     # (length, batch, dim)

        return hy, cy, ys_reshape
