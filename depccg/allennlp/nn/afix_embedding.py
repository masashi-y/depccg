import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("afix_embedding")
class AfixEmbedding(TokenEmbedder):
    def __init__(self, embedding: Embedding, dropout: float = 0.0) -> None:
        super(AfixEmbedding, self).__init__()
        self._embedding = embedding
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return self._embedding.get_output_dim() * 4

    def forward(self, token_afixes: torch.Tensor) -> torch.Tensor:
        batchsize, sentence_length, _ = token_afixes.size()
        embedding = TimeDistributed(self._embedding)
        embedding = TimeDistributed(embedding)
        embedded = embedding(token_afixes.unsqueeze(-1))
        return self._dropout(embedded.view(batchsize, sentence_length, -1))

    # The setdefault requires a custom from_params
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AfixEmbedding':  # type: ignore
        # pylint: disable=arguments-differ
        embedding_params: Params = params.pop("embedding")
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "token_characters" by default.
        embedding_params.setdefault("vocab_namespace", "afixes")
        embedding = Embedding.from_params(vocab, embedding_params)
        dropout = params.pop_float("dropout", 0.0)
        params.assert_empty(cls.__name__)
        return cls(embedding, dropout)
