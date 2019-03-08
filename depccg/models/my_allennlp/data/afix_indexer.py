from typing import Dict, List

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


OOR2 = "OOR2"
OOR3 = "OOR3"
OOR4 = "OOR4"


def get_suffix(word):
    return [word[-1],
            word[-2:] if len(word) > 1 else OOR2,
            word[-3:] if len(word) > 2 else OOR3,
            word[-4:] if len(word) > 3 else OOR4]


def get_prefix(word):
    return [word[0],
            word[:2] if len(word) > 1 else OOR2,
            word[:3] if len(word) > 2 else OOR3,
            word[:4] if len(word) > 3 else OOR4]


@TokenIndexer.register("afix_ids")
class SingleIdTokenIndexer(TokenIndexer[List[int]]):
    def __init__(self,
                 afix_type: str,
                 namespace: str = "afixes") -> None:
        assert afix_type in ['prefix', 'suffix']
        self.afix_type = afix_type
        if afix_type == 'prefix':
            self._get_afix = get_prefix
        else:
            self._get_afix = get_suffix
        self.namespace = namespace

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        text = token.text
        for afix in self._get_afix(text):
            counter[self.namespace][afix] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        indices: List[List[int]] = []

        for token in tokens:
            text = token.text
            afixes = [vocabulary.get_token_index(afix, self.namespace)
                      for afix in self._get_afix(text)]
            indices.append(afixes)

        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> List[int]:
        return [0] * 4

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        return {key: pad_sequence_to_length(val,
                                            desired_num_tokens[key],
                                            default_value=self.get_padding_token)
                for key, val in tokens.items()}
