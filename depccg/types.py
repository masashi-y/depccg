from typing import Optional, NamedTuple, Callable, List
from pathlib import Path
import re
import numpy

from depccg.cat import Category

dunder_pattern = re.compile("__.*__")
protected_pattern = re.compile("_.*")


class Token(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if dunder_pattern.match(item) or protected_pattern.match(item):
            return super().__getattr__(item)
        return self[item]

    def __repr__(self):
        res = super().__repr__()
        return f'Token({res})'

    @classmethod
    def of_piped(cls, string: str) -> 'Token':
        # WORD|POS|NER or WORD|LEMMA|POS|NER
        # or WORD|LEMMA|POS|NER|CHUCK
        items = string.split('|')
        if len(items) == 5:
            word, lemma, pos, entity, chunk = items
        elif len(items) == 4:
            word, lemma, pos, entity = items
            chunk = 'XX'
        else:
            assert len(items) == 3
            word, pos, entity = items
            lemma = 'XX'
            chunk = 'XX'

        return Token(
            word=word,
            lemma=lemma,
            pos=pos,
            entity=entity,
            chunk=chunk
        )

    @classmethod
    def of_word(cls, word: str) -> 'Token':
        return Token(
            word=word,
            lemma='XX',
            pos='XX',
            entity='XX',
            chunk='XX'
        )


class CombinatorResult(NamedTuple):
    cat: Category
    op_string: str
    op_symbol: str
    head_is_left: bool


class ScoringResult(NamedTuple):
    tag_scores: numpy.ndarray
    dep_scores: numpy.ndarray


Combinator = Callable[[Category, Category], Optional[CombinatorResult]]

ApplyBinaryRules = Callable[..., List[CombinatorResult]]
ApplyUnaryRules = Callable[..., List[CombinatorResult]]


class GrammarConfig(NamedTuple):
    apply_binary_rules: ApplyBinaryRules
    apply_unary_rules: ApplyUnaryRules


class ModelConfig(NamedTuple):
    framework: str
    name: str
    url: str
    config: Path
    semantic_templates: Path
