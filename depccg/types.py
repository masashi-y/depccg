from typing import Optional, NamedTuple, Callable
from depccg.cat import Category
import numpy


class Token(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        res = super().__repr__()
        return f'Token({res})'

    @classmethod
    def from_piped(cls, string: str) -> 'Token':
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
    def from_word(cls, word: str) -> 'Token':
        return Token(word=word,
                     lemma='XX',
                     pos='XX',
                     entity='XX',
                     chunk='XX')


class CombinatorResult(NamedTuple):
    cat: Category
    op_string: str
    op_symbol: str
    head_is_left: bool


class ScoringResult(NamedTuple):
    tag_scores: numpy.ndarray
    dep_scores: numpy.ndarray


Combinator = Callable[[Category, Category], Optional[CombinatorResult]]
