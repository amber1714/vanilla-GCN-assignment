import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from formulas import add_self_loops, symmetric_normalize_adjacency


def test_self_loops():
    a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(add_self_loops(a), expected)


def test_symmetric_normalization():
    a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert torch.allclose(symmetric_normalize_adjacency(a), expected, atol=1e-6)
