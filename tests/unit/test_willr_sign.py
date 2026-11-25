"""Williams %R must retain negative values between rolling high and low."""
import torch
from model_core.features import MT5FeatureEngineer


def test_willr_distinguishes_high_midpoint_and_low():
    high = torch.full((3, 20), 20.0)
    low = torch.full((3, 20), 10.0)
    close = torch.tensor([20.0, 15.0, 10.0]).reshape(3, 1).expand_as(high)
    result = MT5FeatureEngineer._willr(close, high, low)
    torch.testing.assert_close(result[:, -1], torch.tensor([0.0, -0.5, -1.0]))


def test_flat_willr_window_is_finite():
    price = torch.ones(2, 20)
    result = MT5FeatureEngineer._willr(price, price, price)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0
