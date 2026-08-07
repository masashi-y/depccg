from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from pathlib import Path

from depccg.cat import Category
from depccg.grammar import en, ja
from depccg.lang import get_global_language

GRAMMARS = {"en": en, "ja": ja}


def read_params(config_path: Path, args):
    with Path(config_path).open(encoding="utf-8") as file:
        params = json.load(file)

    unary_rules = defaultdict(list)
    for source, target in params["unary_rules"]:
        unary_rules[Category.parse(source)].append(Category.parse(target))

    category_dict = None
    if not args.disable_category_dictionary:
        category_dict = {
            word: [Category.parse(category) for category in categories]
            for word, categories in params["cat_dict"].items()
        }

    seen_rules = None
    if not args.disable_seen_rules:
        seen_rules = {
            (
                Category.parse(left).clear_features("X", "nb"),
                Category.parse(right).clear_features("X", "nb"),
            )
            for left, right in params["seen_rules"]
        } or None

    grammar = GRAMMARS[get_global_language()]
    return (
        partial(grammar.apply_binary_rules, seen_rules=seen_rules),
        partial(grammar.apply_unary_rules, unary_rules=unary_rules),
        category_dict,
        [Category.parse(category) for category in params["targets"]],
    )
