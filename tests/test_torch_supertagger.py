import torch

from depccg.torch_supertagger import Biaffine, Bilinear, LegacyLSTMCell


def test_biaffine_matches_explicit_matrix_product():
    layer = Biaffine(2)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2))
    dependent = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    head = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    augmented = torch.cat((dependent, torch.ones(2, 1)), dim=1)
    assert torch.allclose(layer(dependent, head), augmented @ layer.weight @ head.T)


def test_bilinear_matches_component_formula():
    layer = Bilinear(2, 2, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
        layer.weight1.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        layer.weight2.copy_(torch.arange(6, 12, dtype=torch.float32).reshape(2, 3))
        layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    left = torch.tensor([[1.0, 2.0]])
    right = torch.tensor([[3.0, 4.0]])
    expected = (
        torch.einsum("bi,ijo,bj->bo", left, layer.weight, right)
        + left @ layer.weight1
        + right @ layer.weight2
        + layer.bias
    )
    assert torch.allclose(layer(left, right), expected)


def test_legacy_lstm_cell_gate_layout():
    layer = LegacyLSTMCell(1, 1)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        # Candidate, input, forget and output gates respectively.
        layer.input_biases[0].fill_(0.5)
        layer.input_biases[1].fill_(1.0)
        layer.input_biases[2].fill_(-1.0)
        layer.input_biases[3].fill_(2.0)
    hidden, cell = layer(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))
    sigmoid = lambda value: torch.tanh(value * 0.5) * 0.5 + 0.5
    expected_cell = torch.tanh(torch.tensor(0.5)) * sigmoid(torch.tensor(1.0))
    expected_hidden = torch.tanh(expected_cell) * sigmoid(torch.tensor(2.0))
    assert torch.allclose(cell.squeeze(), expected_cell)
    assert torch.allclose(hidden.squeeze(), expected_hidden)
