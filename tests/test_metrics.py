import numpy as np

import torch

import pytest

from sr.metrics import METRICS, calculate_metrics, minshift_loss


def test_metrics():
    x = torch.ones(5, 4, 64, 64)
    factor = 0.7

    with pytest.raises(AssertionError):
        calculate_metrics(x, x+factor, metrics=('L2', 'MSE'))

    mae_loss = METRICS['MAE'](x, x + factor)
    mse_loss = METRICS['MSE'](x, x + factor)
    psnr_loss = METRICS['PSNR'](x, x + factor)
    ssim_loss = METRICS['SSIM'](x, x + factor)
    mixed_loss = METRICS['MIXED'](x, x + factor)

    np.testing.assert_almost_equal(torch.mean(mae_loss).numpy(), factor, decimal=5)
    np.testing.assert_almost_equal(torch.mean(mse_loss).numpy(), factor**2, decimal=5)
    np.testing.assert_almost_equal(torch.mean(psnr_loss).numpy(), -10*np.log10(factor**2), decimal=4)
    np.testing.assert_almost_equal(torch.mean(ssim_loss).numpy(), 0.87358, decimal=5)
    np.testing.assert_almost_equal(torch.mean(mixed_loss).numpy(),
                                   torch.mean(mse_loss + .1*(-1 * ssim_loss + 1)).numpy())


def test_calculate_metrics():
    x = torch.rand(5, 3, 64, 64)
    y = torch.rand(5, 3, 64, 64)

    factor = .3

    scores = calculate_metrics(x, y, list(METRICS.keys()))

    assert set(scores.keys()) == set(METRICS.keys())

    for metric in METRICS:
        np.testing.assert_equal(scores[metric].numpy(), METRICS[metric](x, y).numpy())

    scores = calculate_metrics(x, x+factor, list(METRICS.keys()), apply_correction=True)

    results = dict(MAE=0.0, MSE=0.0, PSNR=80.0, SSIM=1.0, MIXED=0.0)

    for metric in METRICS:
        np.testing.assert_almost_equal(np.mean(scores[metric].numpy()), results[metric], decimal=4)

    assert type(scores) == dict
    assert type(calculate_metrics(x, x+factor, metrics='MAE')) == torch.Tensor
    assert type(calculate_metrics(x, x+factor, metrics='MIXED')) == torch.Tensor


def test_minshift_loss():
    factor = .1
    batch = 2
    h, w = 64, 64

    x = torch.rand(batch, 1, h, w)
    y = x + factor

    shifts = 6

    y[:, :, :shifts//2, :] = -10
    y[:, :, :, :shifts//2] = -10

    scores, _ = minshift_loss(x, y, metric='MSE', shifts=shifts)

    np.testing.assert_almost_equal(scores.numpy(), factor**2)

    scores, indices = minshift_loss(x, y, metric='PSNR', shifts=shifts, apply_correction=True)

    assert scores.shape == (batch,)
    assert indices.shape == (batch, 4)
    np.testing.assert_almost_equal(scores.mean().numpy(), 80.0)
    np.testing.assert_equal(indices.numpy()[0, :], (shifts//2, h-shifts//2, shifts//2, w-shifts//2))
