from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from depccg.types import ScoringResult
from depccg.utils import normalize, read_model_defs

UNK = "*UNKNOWN*"
START = "*START*"
END = "*END*"
OOR2 = "OOR2"
OOR3 = "OOR3"
OOR4 = "OOR4"
MODEL_FILENAME = "tagger_model.pt"


def _prefixes(word: str) -> list[str]:
    return [
        word[0],
        word[:2] if len(word) > 1 else OOR2,
        word[:3] if len(word) > 2 else OOR3,
        word[:4] if len(word) > 3 else OOR4,
    ]


def _suffixes(word: str) -> list[str]:
    return [
        word[-1],
        word[-2:] if len(word) > 1 else OOR2,
        word[-3:] if len(word) > 2 else OOR3,
        word[-4:] if len(word) > 3 else OOR4,
    ]


class Biaffine(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size + 1, size))

    def forward(self, dependent: Tensor, head: Tensor) -> Tensor:
        ones = torch.ones(*dependent.shape[:-1], 1, device=dependent.device)
        dependent = torch.cat((dependent, ones), dim=-1)
        return torch.matmul(
            dependent, torch.matmul(self.weight, head.transpose(-1, -2))
        )


class Bilinear(nn.Module):
    def __init__(self, size1: int, size2: int, output_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size1, size2, output_size))
        self.weight1 = nn.Parameter(torch.empty(size1, output_size))
        self.weight2 = nn.Parameter(torch.empty(size2, output_size))
        self.bias = nn.Parameter(torch.empty(output_size))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return (
            torch.einsum("...i,ijo,...j->...o", left, self.weight, right)
            + torch.matmul(left, self.weight1)
            + torch.matmul(right, self.weight2)
            + self.bias
        )


class LegacyLSTMCell(nn.Module):
    """LSTM cell preserving the gate layout used by the released models."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size, input_size)) for _ in range(4)]
        )
        self.hidden_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size, hidden_size)) for _ in range(4)]
        )
        self.input_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size)) for _ in range(4)]
        )
        self.hidden_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size)) for _ in range(4)]
        )

    @staticmethod
    def _sigmoid(value: Tensor) -> Tensor:
        # Chainer's CPU LSTM uses this tanh formulation rather than exp.
        return torch.tanh(value * 0.5) * 0.5 + 0.5

    def forward(
        self, value: Tensor, hidden: Tensor, cell: Tensor
    ) -> tuple[Tensor, Tensor]:
        gates = [
            F.linear(value, self.input_weights[index], self.input_biases[index])
            + F.linear(hidden, self.hidden_weights[index], self.hidden_biases[index])
            for index in range(4)
        ]
        # FixedLengthNStepLSTM's row-interleaved stack is consumed by
        # chainer.functions.lstm in the layout g, i, f, o.
        candidate = torch.tanh(gates[0])
        input_gate = self._sigmoid(gates[1])
        forget_gate = self._sigmoid(gates[2])
        output_gate = self._sigmoid(gates[3])
        cell = forget_gate * cell + input_gate * candidate
        return output_gate * torch.tanh(cell), cell


class LegacyStackedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                LegacyLSTMCell(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(layers)
            ]
        )

    def forward(self, values: Tensor) -> Tensor:
        batch = values.shape[1]
        hidden = [values.new_zeros(batch, self.hidden_size) for _ in self.layers]
        cells = [values.new_zeros(batch, self.hidden_size) for _ in self.layers]
        outputs = []
        for value in values.unbind(0):
            for index, layer in enumerate(self.layers):
                hidden[index], cells[index] = layer(value, hidden[index], cells[index])
                value = hidden[index]
            outputs.append(value)
        return torch.stack(outputs)


class EnglishModel(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        word_dim = int(config["word_dim"])
        affix_dim = int(config["afix_dim"])
        hidden_dim = int(config["hidden_dim"])
        dep_dim = int(config["dep_dim"])
        layers = int(config["nlayers"])
        targets = config["targets"]
        assert isinstance(targets, dict)

        self.word_embedding = nn.Embedding(int(config["n_words"]), word_dim)
        self.prefix_embedding = nn.Embedding(int(config["n_prefixes"]), affix_dim)
        self.suffix_embedding = nn.Embedding(int(config["n_suffixes"]), affix_dim)
        input_dim = word_dim + 8 * affix_dim
        self.forward_lstm = LegacyStackedLSTM(input_dim, hidden_dim, layers)
        self.backward_lstm = LegacyStackedLSTM(input_dim, hidden_dim, layers)
        self.arc_dep = nn.Linear(2 * hidden_dim, dep_dim)
        self.arc_head = nn.Linear(2 * hidden_dim, dep_dim)
        self.rel_dep = nn.Linear(2 * hidden_dim, dep_dim)
        self.rel_head = nn.Linear(2 * hidden_dim, dep_dim)
        self.biaffine_arc = Biaffine(dep_dim)
        self.biaffine_tag = Bilinear(dep_dim, dep_dim, len(targets))

    @staticmethod
    def _embed_with_ignored_ids(embedding: nn.Embedding, ids: Tensor) -> Tensor:
        mask = ids.ge(0).unsqueeze(-1)
        return embedding(ids.clamp_min(0)) * mask

    def forward(
        self, words: Tensor, suffixes: Tensor, prefixes: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, length = words.shape
        word_values = self._embed_with_ignored_ids(self.word_embedding, words)
        suffix_values = self._embed_with_ignored_ids(
            self.suffix_embedding, suffixes
        ).reshape(batch, length, -1)
        prefix_values = self._embed_with_ignored_ids(
            self.prefix_embedding, prefixes
        ).reshape(batch, length, -1)
        values = torch.cat(
            (word_values, suffix_values, prefix_values), dim=-1
        ).transpose(0, 1)
        forward_values = self.forward_lstm(values)
        backward_values = self.backward_lstm(values.flip(0))
        hidden = torch.cat((forward_values, backward_values.flip(0)), dim=-1).transpose(
            0, 1
        )

        arc_dep = F.elu(self.arc_dep(hidden))
        arc_head = F.elu(self.arc_head(hidden))
        dependency_scores = self.biaffine_arc(arc_dep, arc_head)
        heads = dependency_scores.argmax(dim=2)

        flat_hidden = hidden.reshape(batch * length, -1)
        flat_heads = (
            heads.flatten()
            + torch.arange(batch, device=hidden.device).repeat_interleave(length)
            * length
        )
        head_values = F.elu(self.rel_head(flat_hidden))[flat_heads]
        child_values = F.elu(self.rel_dep(flat_hidden))
        category_scores = self.biaffine_tag(child_values, head_values).reshape(
            batch, length, -1
        )
        return category_scores, dependency_scores


class EnglishFeatureExtractor:
    def __init__(self, model_dir: Path) -> None:
        self.words = read_model_defs(str(model_dir / "words.txt"))
        self.prefixes = read_model_defs(str(model_dir / "prefixes.txt"))
        self.suffixes = read_model_defs(str(model_dir / "suffixes.txt"))

    def __call__(
        self, sentence: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sentence = [normalize(word) for word in sentence]
        word_ids = np.asarray(
            [
                self.words[START],
                *(self.words.get(word.lower(), self.words[UNK]) for word in sentence),
                self.words[END],
            ],
            dtype=np.int64,
        )
        prefix_ids = [[self.prefixes[START], -1, -1, -1]]
        prefix_ids += [
            [self.prefixes.get(value, self.prefixes[UNK]) for value in _prefixes(word)]
            for word in sentence
        ]
        prefix_ids += [[self.prefixes[END], -1, -1, -1]]
        suffix_ids = [[self.suffixes[START], -1, -1, -1]]
        suffix_ids += [
            [self.suffixes.get(value, self.suffixes[UNK]) for value in _suffixes(word)]
            for word in sentence
        ]
        suffix_ids += [[self.suffixes[END], -1, -1, -1]]
        return word_ids, np.asarray(suffix_ids), np.asarray(prefix_ids)


class JapaneseModel(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        word_dim = int(config["word_dim"])
        char_dim = int(config["char_dim"])
        hidden_dim = int(config["hidden_dim"])
        dep_dim = int(config["dep_dim"])
        layers = int(config["nlayers"])
        targets = config["targets"]
        assert isinstance(targets, dict)
        self.word_embedding = nn.Embedding(int(config["n_words"]), word_dim)
        self.char_embedding = nn.Embedding(int(config["n_chars"]), 50)
        self.char_convolution = nn.Conv2d(1, char_dim, (3, 50), padding=(1, 0))
        self.forward_lstm = LegacyStackedLSTM(word_dim + char_dim, hidden_dim, layers)
        self.backward_lstm = LegacyStackedLSTM(word_dim + char_dim, hidden_dim, layers)
        self.arc_dep = nn.Linear(2 * hidden_dim, dep_dim)
        self.arc_head = nn.Linear(2 * hidden_dim, dep_dim)
        self.rel_dep = nn.Linear(2 * hidden_dim, dep_dim)
        self.rel_head = nn.Linear(2 * hidden_dim, dep_dim)
        self.biaffine_arc = Biaffine(dep_dim)
        self.biaffine_tag = Bilinear(dep_dim, dep_dim, len(targets))

    def forward(self, words: Tensor, chars: Tensor) -> tuple[Tensor, Tensor]:
        word_values = self.word_embedding(words)
        char_mask = chars.ge(0).unsqueeze(-1)
        char_values = self.char_embedding(chars.clamp_min(0)) * char_mask
        char_values = (
            self.char_convolution(char_values.unsqueeze(1)).amax(dim=2).squeeze(-1)
        )
        values = torch.cat((word_values, char_values), dim=-1).unsqueeze(1)
        forward_values = self.forward_lstm(values)
        backward_values = self.backward_lstm(values.flip(0))
        hidden = torch.cat((forward_values, backward_values.flip(0)), dim=-1).squeeze(1)
        dependency_scores = self.biaffine_arc(
            F.elu(self.arc_dep(hidden)), F.elu(self.arc_head(hidden))
        )
        heads = dependency_scores.argmax(dim=1)
        category_scores = self.biaffine_tag(
            F.elu(self.rel_dep(hidden)), F.elu(self.rel_head(hidden)[heads])
        )
        return category_scores, dependency_scores


class JapaneseFeatureExtractor:
    def __init__(self, model_dir: Path) -> None:
        self.words = read_model_defs(str(model_dir / "words.txt"))
        self.chars = read_model_defs(str(model_dir / "chars.txt"))

    def __call__(self, sentence: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        words = [START, *sentence, END]
        word_ids = np.asarray(
            [self.words.get(word, self.words[UNK]) for word in words], dtype=np.int64
        )
        max_chars = max(len(word) for word in sentence)
        char_ids = np.full((len(words), max_chars), -1, dtype=np.int64)
        char_ids[0, 0] = self.chars[START]
        char_ids[-1, 0] = self.chars[END]
        for row, word in enumerate(sentence, 1):
            char_ids[row, : len(word)] = [
                self.chars.get(char, self.chars[UNK]) for char in word
            ]
        return word_ids, char_ids


class TorchSupertagger:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        checkpoint = torch.load(
            model_dir / MODEL_FILENAME, map_location=device, weights_only=True
        )
        self.language = checkpoint["language"]
        self.config = checkpoint["config"]
        if self.language == "en":
            self.model = EnglishModel(self.config).to(device)
            self.extractor = EnglishFeatureExtractor(model_dir)
        elif self.language == "ja":
            self.model = JapaneseModel(self.config).to(device)
            self.extractor = JapaneseFeatureExtractor(model_dir)
        else:
            raise ValueError(f"unsupported model language: {self.language}")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.device = device

    @property
    def categories(self) -> list[str]:
        targets = self.config["targets"]
        return [key for key, _ in sorted(targets.items(), key=lambda item: item[1])]

    def predict_doc(
        self, doc: Sequence[Sequence[str]], batchsize: int = 32, **_: object
    ):
        if self.language == "ja":
            results = []
            with torch.inference_mode():
                for sentence in doc:
                    words, chars = self.extractor(sentence)
                    tags, deps = self.model(
                        torch.as_tensor(words, device=self.device),
                        torch.as_tensor(chars, device=self.device),
                    )
                    results.append(
                        ScoringResult(
                            F.log_softmax(tags[1:-1], dim=-1)
                            .cpu()
                            .numpy()
                            .astype(np.float32),
                            F.log_softmax(deps[1:-1, :-1], dim=-1)
                            .cpu()
                            .numpy()
                            .astype(np.float32),
                        )
                    )
            return results, self.categories

        indexed = sorted(enumerate(doc), key=lambda item: len(item[1]))
        results = []
        with torch.inference_mode():
            for offset in range(0, len(indexed), batchsize):
                indices, sentences = zip(*indexed[offset : offset + batchsize])
                features = [self.extractor(sentence) for sentence in sentences]
                max_length = max(len(item[0]) for item in features)
                arrays = []
                for position in range(3):
                    shape = (
                        (len(features), max_length)
                        if position == 0
                        else (len(features), max_length, 4)
                    )
                    array = np.full(shape, -1, dtype=np.int64)
                    for row, feature in enumerate(features):
                        array[row, : len(feature[position])] = feature[position]
                    arrays.append(torch.as_tensor(array, device=self.device))
                category_scores, dependency_scores = self.model(*arrays)
                for index, sentence in zip(indices, sentences):
                    row = indices.index(index)
                    length = len(sentence) + 2
                    tags = (
                        F.log_softmax(category_scores[row, 1 : length - 1], dim=-1)
                        .cpu()
                        .numpy()
                    )
                    deps = (
                        F.log_softmax(
                            dependency_scores[row, 1 : length - 1, : length - 1], dim=-1
                        )
                        .cpu()
                        .numpy()
                    )
                    results.append(
                        (
                            index,
                            ScoringResult(
                                tags.astype(np.float32), deps.astype(np.float32)
                            ),
                        )
                    )
        return [result for _, result in sorted(results)], self.categories


def load_torch_tagger(model_path: Path, device_id: int = -1) -> TorchSupertagger:
    if device_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")
    return TorchSupertagger(Path(model_path), device)
