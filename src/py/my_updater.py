

import chainer
from chainer import training, Variable, ChainList
from chainer import reporter as reporter_module
from chainer.training import extensions
import copy

class MyUpdater(training.StandardUpdater):
    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        optimizer.update(loss_func, self.converter(batch, self.device))


class MyEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target
        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                eval_func(*self.converter(batch, self.device))
            summary.add(observation)
        return summary.compute_mean()


