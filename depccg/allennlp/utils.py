from functools import partial
from collections import defaultdict

from allennlp.common.params import Params
from depccg.lang import get_global_language
from depccg.instance_models import GRAMMARS
from depccg.cat import Category


def read_params(
    param_path: str,
    disable_category_dictionary: bool = False,
    disable_seen_rules: bool = False,
):
    lang = get_global_language()
    params = Params.from_file(param_path)

    unary_rules = defaultdict(list)
    for key, value in params.pop('unary_rules'):
        unary_rules[Category.parse(key)].append(Category.parse(value))

    if disable_category_dictionary:
        category_dict = None
    else:
        category_dict = {
            word: [Category.parse(cat) for cat in cats]
            for word, cats in params.pop('cat_dict').items()
        }

    if disable_seen_rules:
        seen_rules = None
    else:
        seen_rules = {
            (Category.parse(x).clear_features('X', 'nb'),
             Category.parse(y).clear_features('X', 'nb'))
            for x, y in params.pop('seen_rules')
        }
        if len(seen_rules) == 0:
            seen_rules = None
    try:
        apply_binary_rules = partial(
            GRAMMARS[lang].apply_binary_rules,
            seen_rules=seen_rules
        )
        apply_unary_rules = partial(
            GRAMMARS[lang].apply_unary_rules,
            unary_rules=unary_rules
        )
    except KeyError:
        raise KeyError('unsupported language: {args.lang}')

    root_categories = [
        Category.parse(category)
        for category in params.pop('targets')
    ]

    return (
        apply_binary_rules,
        apply_unary_rules,
        category_dict,
        root_categories
    )
