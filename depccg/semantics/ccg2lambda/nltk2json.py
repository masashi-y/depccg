
from nltk.sem.logic import *
from .logic_parser import lexpr
from .nltk2normal import remove_true


def run(expression):
    if not isinstance(expression, Expression):
        expression = lexpr(expression)
    expression = remove_true(expression)
    return _run(expression)


def _run(expression):
    if isinstance(expression, ApplicationExpression):
        return {
            'type': 'app',
            'func': _run(expression.function),
            'arg': _run(expression.argument)
            }
    elif isinstance(expression, EqualityExpression):
        return {
            'type': 'equal',
            'first': _run(expression.first),
            'second': _run(expression.second)
        }
    elif isinstance(expression, AndExpression):
        return {
            'type': 'and',
            'first': _run(expression.first),
            'second': _run(expression.second)
        }
    elif isinstance(expression, OrExpression):
        return {
            'type': 'or',
            'first': _run(expression.first),
            'second': _run(expression.second)
        }
    elif isinstance(expression, ImpExpression):
        return {
            'type': 'imp',
            'first': _run(expression.first),
            'second': _run(expression.second)
        }
    elif isinstance(expression, NegatedExpression):
        return {
            'type': 'not',
            'term': _run(expression.term),
        }
    elif isinstance(expression, ExistsExpression):
        return {
            'type': 'exists',
            'variable': str(expression.variable),
            'term': _run(expression.term),
        }
    elif isinstance(expression, AllExpression):
        return {
            'type': 'all',
            'variable': str(expression.variable),
            'term': _run(expression.term),
        }
    elif isinstance(expression, LambdaExpression):
        return {
            'type': 'lambda',
            'variable': str(expression.variable),
            'term': _run(expression.term),
        }
    else:
        return str(expression)