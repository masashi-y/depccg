from __future__ import division

import numpy

import random
from chainer.dataset import iterator


class SerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset) - 1

        self._repeat = repeat
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = random.randint(0, self.N)
        i_end = i + self.batch_size

        batch = self.dataset[i:i_end]
        if i_end >= self.N:
            i_end = self.N
            i = self.N - self.batch_size

        self.current_position += self.batch_size
        if self.current_position >= self.N:
            self.epoch += 1
            self.current_position = 0
        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

