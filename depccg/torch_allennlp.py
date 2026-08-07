"""PyTorch-only inference for depccg models originally trained with AllenNLP.

The ELMo cell equations and character mapping follow AllenNLP 0.9.0.  This
module intentionally implements only the inference path used by the released
depccg checkpoints, avoiding a runtime dependency on AllenNLP.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from depccg.types import ScoringResult


def _read_vocabulary(path: Path) -> dict[str, int]:
    # AllenNLP padded namespaces reserve 0 for padding and 1 for OOV.
    return {token.rstrip("\n"): index + 2 for index, token in enumerate(path.open())}


class Highway(nn.Module):
    def __init__(self, size: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(size, size * 2) for _ in range(layers)])

    def forward(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            nonlinear, gate = layer(value).chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            value = gate * value + (1 - gate) * F.relu(nonlinear)
        return value


class ProjectedLSTMCell(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_linearity = nn.Linear(input_size, 16384, bias=False)
        self.state_linearity = nn.Linear(512, 16384)
        self.state_projection = nn.Linear(4096, 512, bias=False)

    def forward(self, values: Tensor, reverse: bool = False) -> Tensor:
        state = values.new_zeros(1, 512)
        memory = values.new_zeros(1, 4096)
        outputs = values.new_zeros(values.shape[0], 512)
        indices = (
            range(values.shape[0] - 1, -1, -1) if reverse else range(values.shape[0])
        )
        for index in indices:
            gates = self.input_linearity(
                values[index : index + 1]
            ) + self.state_linearity(state)
            input_gate, forget_gate, memory_init, output_gate = gates.chunk(4, dim=-1)
            memory = (
                torch.sigmoid(input_gate) * torch.tanh(memory_init)
                + torch.sigmoid(forget_gate) * memory
            )
            memory = memory.clamp(-3.0, 3.0)
            state = self.state_projection(
                torch.sigmoid(output_gate) * torch.tanh(memory)
            )
            state = state.clamp(-3.0, 3.0)
            outputs[index] = state[0]
        return outputs


class ElmoEncoder(nn.Module):
    _MAX_WORD_LENGTH = 50
    _BOW = 258
    _EOW = 259
    _PADDING = 260

    def __init__(self) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(262, 16)
        widths = (1, 2, 3, 4, 5, 6, 7)
        filters = (32, 32, 64, 128, 256, 512, 1024)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(16, count, width) for width, count in zip(widths, filters)]
        )
        self.highway = Highway(2048, 2)
        self.projection = nn.Linear(2048, 512)
        self.forward_layers = nn.ModuleList(
            [ProjectedLSTMCell(512), ProjectedLSTMCell(512)]
        )
        self.backward_layers = nn.ModuleList(
            [ProjectedLSTMCell(512), ProjectedLSTMCell(512)]
        )
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.empty(1)) for _ in range(3)]
        )
        self.gamma = nn.Parameter(torch.empty(1))

    @classmethod
    def _special(cls, character: int) -> list[int]:
        result = [cls._PADDING] * cls._MAX_WORD_LENGTH
        result[:3] = [cls._BOW, character, cls._EOW]
        return [value + 1 for value in result]

    @classmethod
    def _characters(cls, word: str) -> list[int]:
        encoded = word.encode("utf-8", "ignore")[: cls._MAX_WORD_LENGTH - 2]
        result = [cls._PADDING] * cls._MAX_WORD_LENGTH
        result[0] = cls._BOW
        result[1 : len(encoded) + 1] = encoded
        result[len(encoded) + 1] = cls._EOW
        return [value + 1 for value in result]

    def forward(self, sentence: Sequence[str]) -> Tensor:
        character_ids = [self._special(256)]
        character_ids.extend(self._characters(word) for word in sentence)
        character_ids.append(self._special(257))
        ids = torch.tensor(character_ids, device=self.gamma.device)
        embedded = self.char_embedding(ids).transpose(1, 2)
        convolved = [
            F.relu(layer(embedded).amax(dim=-1)) for layer in self.convolutions
        ]
        token_embedding = self.projection(self.highway(torch.cat(convolved, dim=-1)))

        forward = token_embedding
        backward = token_embedding
        activations = [torch.cat((token_embedding, token_embedding), dim=-1)]
        for layer_index, (forward_layer, backward_layer) in enumerate(
            zip(self.forward_layers, self.backward_layers)
        ):
            forward_cache, backward_cache = forward, backward
            forward = forward_layer(forward)
            backward = backward_layer(backward, reverse=True)
            if layer_index:
                forward = forward + forward_cache
                backward = backward + backward_cache
            activations.append(torch.cat((forward, backward), dim=-1))
        weights = torch.softmax(torch.cat(list(self.scalar_parameters)), dim=0)
        mixed = self.gamma * sum(
            weight * value for weight, value in zip(weights, activations)
        )
        return mixed[1:-1]


class CharacterCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(82, 100)
        self.convolution = nn.Conv1d(100, 200, 5)

    def forward(self, characters: Tensor) -> Tensor:
        return F.relu(
            self.convolution(self.embedding(characters).transpose(1, 2)).amax(dim=-1)
        )


class AllenNLPParser(nn.Module):
    def __init__(self, input_size: int, categories: int, use_elmo: bool) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(400002, 100)
        self.elmo = ElmoEncoder() if use_elmo else None
        self.character_cnn = None if use_elmo else CharacterCNN()
        self.encoder = nn.LSTM(
            input_size,
            300,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.32,
        )
        self.head_sentinel = nn.Parameter(torch.empty(1, 1, 600))
        self.head_arc = nn.Linear(600, 300)
        self.child_arc = nn.Linear(600, 300)
        self.head_tag = nn.Linear(600, 300)
        self.child_tag = nn.Linear(600, 300)
        self.arc_weight = nn.Parameter(torch.empty(301, 301))
        self.arc_bias = nn.Parameter(torch.empty(1))
        self.tag_weight = nn.Parameter(torch.empty(categories, 300, 300))
        self.tag_weight1 = nn.Parameter(torch.empty(categories, 300))
        self.tag_weight2 = nn.Parameter(torch.empty(categories, 300))
        self.tag_bias = nn.Parameter(torch.empty(categories))

    def forward(
        self, words: Tensor, sentence: Sequence[str], characters: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        embedded = self.word_embedding(words)
        if self.elmo is not None:
            contextual = self.elmo(sentence)
        else:
            assert self.character_cnn is not None and characters is not None
            contextual = self.character_cnn(characters)
        encoded, _ = self.encoder(
            torch.cat((embedded, contextual), dim=-1).unsqueeze(0)
        )
        encoded = torch.cat((self.head_sentinel, encoded), dim=1)
        head_arc = F.elu(self.head_arc(encoded))
        child_arc = F.elu(self.child_arc(encoded))
        ones = encoded.new_ones(1, encoded.shape[1], 1)
        head_arc = torch.cat((head_arc, ones), dim=-1)
        child_arc = torch.cat((child_arc, ones), dim=-1)
        arcs = head_arc @ self.arc_weight @ child_arc.transpose(1, 2) + self.arc_bias
        diagonal = torch.eye(
            arcs.shape[1], dtype=torch.bool, device=arcs.device
        ).unsqueeze(0)
        arcs = arcs.masked_fill(diagonal, -torch.inf)
        heads = arcs.argmax(dim=-1)

        head_tags = F.elu(self.head_tag(encoded))[0, heads[0]]
        child_tags = F.elu(self.child_tag(encoded))[0]
        tags = (
            torch.einsum("ni,oij,nj->no", head_tags, self.tag_weight, child_tags)
            + head_tags @ self.tag_weight1.T
            + child_tags @ self.tag_weight2.T
            + self.tag_bias
        )
        normalized_tags = F.log_softmax(tags, dim=-1)
        # AllenNLP masks a token when its greedy label is the OOV label (1),
        # then depccg discards the padding and OOV columns.
        normalized_tags[tags.argmax(dim=-1) == 1] = 0
        return normalized_tags[1:, 2:], F.log_softmax(arcs[0, 1:], dim=-1)


class AllenNLPSupertagger:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        config = json.loads((model_dir / "config.json").read_text())
        model_config = config["model"]
        use_elmo = "elmo" in model_config["text_field_embedder"]["token_embedders"]
        self.words = _read_vocabulary(model_dir / "vocabulary" / "tokens.txt")
        self.characters = (
            None
            if use_elmo
            else _read_vocabulary(model_dir / "vocabulary" / "token_characters.txt")
        )
        category_vocabulary = (
            (model_dir / "vocabulary" / "head_tags.txt").read_text().splitlines()
        )
        # This namespace is non-padded: index 0 is OOV and the checkpoint adds
        # one padding slot.  depccg exposes only the actual categories.
        self.categories = category_vocabulary[1:]
        self.model = AllenNLPParser(
            int(model_config["encoder"]["input_size"]),
            len(category_vocabulary) + 1,
            use_elmo,
        ).to(device)
        legacy = torch.load(
            model_dir / "weights.th", map_location=device, weights_only=True
        )
        self._load_weights(legacy)
        self.model.eval()
        self.device = device

    def _load_weights(self, legacy: dict[str, Tensor]) -> None:
        model = self.model
        model.head_sentinel.data.copy_(legacy["_head_sentinel"])
        model.word_embedding.weight.data.copy_(
            legacy["text_field_embedder.token_embedder_tokens.weight"]
        )
        for name in ("head_arc", "child_arc", "head_tag", "child_tag"):
            old = (
                name.replace("head_arc", "head_arc_feedforward")
                .replace("child_arc", "child_arc_feedforward")
                .replace("head_tag", "head_tag_feedforward")
                .replace("child_tag", "child_tag_feedforward")
            )
            getattr(model, name).weight.data.copy_(
                legacy[f"{old}._linear_layers.0.weight"]
            )
            getattr(model, name).bias.data.copy_(legacy[f"{old}._linear_layers.0.bias"])
        model.arc_weight.data.copy_(legacy["arc_attention._weight_matrix"])
        model.arc_bias.data.copy_(legacy["arc_attention._bias"])
        model.tag_weight.data.copy_(legacy["tag_bilinear.W"])
        model.tag_weight1.data.copy_(legacy["tag_bilinear.V1"])
        model.tag_weight2.data.copy_(legacy["tag_bilinear.V2"])
        model.tag_bias.data.copy_(legacy["tag_bilinear.bias"])
        encoder_state = {
            key.removeprefix("encoder._module."): value
            for key, value in legacy.items()
            if key.startswith("encoder._module.")
        }
        model.encoder.load_state_dict(encoder_state)
        if model.elmo is not None:
            self._load_elmo(legacy)
        else:
            assert model.character_cnn is not None
            prefix = "text_field_embedder.token_embedder_token_characters."
            model.character_cnn.embedding.weight.data.copy_(
                legacy[prefix + "_embedding._module.weight"]
            )
            model.character_cnn.convolution.weight.data.copy_(
                legacy[prefix + "_encoder._module.conv_layer_0.weight"]
            )
            model.character_cnn.convolution.bias.data.copy_(
                legacy[prefix + "_encoder._module.conv_layer_0.bias"]
            )

    def _load_elmo(self, legacy: dict[str, Tensor]) -> None:
        elmo = self.model.elmo
        assert elmo is not None
        prefix = "text_field_embedder.token_embedder_elmo._elmo."
        token = prefix + "_elmo_lstm._token_embedder."
        elmo.char_embedding.weight.data.copy_(legacy[token + "_char_embedding_weights"])
        for index, convolution in enumerate(elmo.convolutions):
            convolution.weight.data.copy_(legacy[f"{token}char_conv_{index}.weight"])
            convolution.bias.data.copy_(legacy[f"{token}char_conv_{index}.bias"])
        for index, layer in enumerate(elmo.highway.layers):
            layer.weight.data.copy_(legacy[f"{token}_highways._layers.{index}.weight"])
            layer.bias.data.copy_(legacy[f"{token}_highways._layers.{index}.bias"])
        elmo.projection.weight.data.copy_(legacy[token + "_projection.weight"])
        elmo.projection.bias.data.copy_(legacy[token + "_projection.bias"])
        lstm = prefix + "_elmo_lstm._elmo_lstm."
        for index in range(2):
            for direction, layers in (
                ("forward", elmo.forward_layers),
                ("backward", elmo.backward_layers),
            ):
                old = f"{lstm}{direction}_layer_{index}."
                layer = layers[index]
                layer.input_linearity.weight.data.copy_(
                    legacy[old + "input_linearity.weight"]
                )
                layer.state_linearity.weight.data.copy_(
                    legacy[old + "state_linearity.weight"]
                )
                layer.state_linearity.bias.data.copy_(
                    legacy[old + "state_linearity.bias"]
                )
                layer.state_projection.weight.data.copy_(
                    legacy[old + "state_projection.weight"]
                )
        for index, parameter in enumerate(elmo.scalar_parameters):
            parameter.data.copy_(
                legacy[f"{prefix}scalar_mix_0.scalar_parameters.{index}"]
            )
        elmo.gamma.data.copy_(legacy[prefix + "scalar_mix_0.gamma"])

    def _features(self, sentence: Sequence[str]) -> tuple[Tensor, Tensor | None]:
        words = torch.tensor(
            [self.words.get(word.lower(), 1) for word in sentence], device=self.device
        )
        if self.characters is None:
            return words, None
        sequences = [
            [self.characters.get(character, 1) for character in word] + [0, 0, 0, 0]
            for word in sentence
        ]
        length = max(len(sequence) for sequence in sequences)
        characters = torch.zeros(
            len(sequences), length, dtype=torch.long, device=self.device
        )
        for row, sequence in enumerate(sequences):
            characters[row, : len(sequence)] = torch.tensor(
                sequence, device=self.device
            )
        return words, characters

    def predict_doc(self, doc: Sequence[Sequence[str]], **_: object):
        results = []
        with torch.inference_mode():
            for sentence in doc:
                words, characters = self._features(sentence)
                tags, deps = self.model(words, sentence, characters)
                results.append(
                    ScoringResult(
                        tags.cpu().numpy().astype("float32"),
                        deps.cpu().numpy().astype("float32"),
                    )
                )
        return results, self.categories


def load_allennlp_tagger(model_path: Path, device_id: int = -1) -> AllenNLPSupertagger:
    if device_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")
    return AllenNLPSupertagger(model_path, device)
