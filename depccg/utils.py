from typing import List, Tuple, Dict, NamedTuple, Optional
import json
import logging
import numpy
from depccg.types import ScoringResult

from depccg.cat import Category


logger = logging.getLogger(__name__)


def is_json(file_path: str) -> bool:
    try:
        with open(file_path, 'r') as data_file:
            json.load(data_file)
            return True
    except json.JSONDecodeError:
        return False


def normalize(word: str) -> str:
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "{"
    elif word == "-RCB-":
        return "}"
    elif word == "-LSB-":
        return "["
    elif word == "-RSB-":
        return "]"
    else:
        return word


def denormalize(word: str) -> str:
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "{":
        return "-LCB-"
    elif word == "}":
        return "-RCB-"
    elif word == "[":
        return "-LSB-"
    elif word == "]":
        return "-RSB-"
    word = word.replace(">", "-RAB-")
    word = word.replace("<", "-LAB-")
    return word


def read_pretrained_embeddings(filepath: str) -> numpy.ndarray:
    nvocab = 0
    io = open(filepath)
    dim = len(io.readline().split())
    io.seek(0)
    for _ in io:
        nvocab += 1
    io.seek(0)
    res = numpy.empty((nvocab, dim), dtype=numpy.float32)
    for i, line in enumerate(io):
        line = line.strip()
        if len(line) == 0:
            continue
        res[i] = line.split()
    io.close()
    return res


def read_model_defs(filepath: str) -> Dict[str, int]:
    return {
        line.strip().split(' ')[0]: i
        for i, line in enumerate(open(filepath, encoding='utf-8'))
    }


def remove_comment(line: str) -> str:
    comment = line.find('#')
    if comment != -1:
        line = line[:comment]
    return line.strip()


class SpanInfo(NamedTuple):
    cat: Category
    idx: int
    end_idx: Optional[int] = None


def read_partial_tree(string: str) -> Tuple[List[str], List[SpanInfo]]:
    stack = []
    spans = []
    words = []
    buf = list(reversed(string.split()))
    counter = 0
    while buf:
        item = buf.pop()
        if item.startswith('<'):
            cat = item[1:]
            cat = None if cat == 'X' else Category.parse(cat)
            stack.append(cat)
            stack.append(counter)
        elif item == '>':
            start = stack.pop()
            cat = stack.pop()
            spans.append(SpanInfo(cat, start, counter - start))
        else:
            items = item.split('|')
            if len(items) == 1:
                words.append(items[0])
            elif len(items) == 2:
                cat, word = items
                assert len(cat) > 0 and len(word) > 0, \
                    'failed to parse partially annotated sentence.'
                words.append(word)
                spans.append(SpanInfo(Category.parse(cat), counter))
            counter += 1
    assert len(stack) == 0, 'failed to parse partially annotated sentence.'
    return words, spans


def maybe_split_and_join(string):
    if isinstance(string, list):
        split = string
        join = ' '.join(string)
    else:
        assert isinstance(string, str)
        split = string.split(' ')
        join = string
    return split, join


def read_weights(filename, file_type='json'):
    assert file_type == 'json'
    categories = None
    scores = []
    for line in open(filename):
        json_dict = json.loads(line.strip())

        if categories is None:
            categories = [
                Category.parse(cat)
                for cat in json_dict['categories']
            ]

        dep_scores = numpy.array(json_dict['heads']) \
            .reshape(json_dict['heads_shape']) \
            .astype(numpy.float32)
        tag_scores = numpy.array(json_dict['head_tags']) \
            .reshape(json_dict['head_tags_shape']) \
            .astype(numpy.float32)

        scores.append(
            ScoringResult(
                tag_scores,
                dep_scores
            )
        )

    return scores, categories
