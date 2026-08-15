import torch

from depccg.torch_allennlp import (
    CharacterCNN,
    ElmoEncoder,
    Highway,
    _read_vocabulary,
)


def test_read_vocabulary_reserves_only_padding_index(tmp_path):
    vocabulary = tmp_path / "tokens.txt"
    vocabulary.write_text("@@UNKNOWN@@\nknown\n")

    assert _read_vocabulary(vocabulary) == {"@@UNKNOWN@@": 1, "known": 2}


def test_elmo_character_ids_match_allennlp_mapping():
    ids = ElmoEncoder._characters("A")

    assert ids[:3] == [259, 66, 260]
    assert ids[3:] == [261] * 47


def test_highway_uses_allennlp_gate_direction():
    highway = Highway(1, 1)
    layer = highway.layers[0]
    layer.weight.data.zero_()
    layer.bias.data.copy_(torch.tensor([4.0, 20.0]))

    result = highway(torch.tensor([[2.0]]))

    torch.testing.assert_close(result, torch.tensor([[2.0]]))


def test_character_cnn_masks_padding_embeddings():
    cnn = CharacterCNN()
    characters = torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]])

    before = cnn(characters)
    cnn.embedding.weight.data[0].fill_(1000)

    torch.testing.assert_close(cnn(characters), before)
