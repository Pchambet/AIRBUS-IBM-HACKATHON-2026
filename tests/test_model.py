import numpy as np
import pytest

from corrosion_risk.model import shrink


def test_shrink_endpoints_and_order():
    p = np.array([-0.2, 0.0, 0.3, 0.5, 0.9, 1.4])
    assert np.allclose(shrink(p, 1.0), np.clip(p, 0, 1))
    assert np.allclose(shrink(p, 0.0), 0.5)
    assert np.allclose(shrink(p, 0.7), [0.15, 0.15, 0.36, 0.5, 0.78, 0.85])
    assert np.all(np.diff(shrink(p, 0.7)) >= 0)  # never reorders


@pytest.mark.parametrize("alpha", [0.2, 0.7])
def test_shrink_bounds(alpha):
    out = shrink(np.linspace(-1, 2, 50), alpha)
    assert out.min() == pytest.approx(0.5 - alpha / 2)
    assert out.max() == pytest.approx(0.5 + alpha / 2)
