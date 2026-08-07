import torch

from depccg.torch_allennlp import ElmoEncoder, Highway


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
