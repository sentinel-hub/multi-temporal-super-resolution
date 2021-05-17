from typing import Tuple, Union
from functools import partial

import torch
import tensorflow as tf 

from piqa.ssim import SSIM
from piqa.psnr import mse, psnr
import numpy as np
import gc 

def mixed_metric(hrs: torch.Tensor, srs: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """ Computes a metric defined as MSE + weight * (-SSIM + 1) """
    return mse(hrs, srs) + weight * (-1 * ssim(hrs, srs) + 1)


def ssim(hrs: torch.Tensor, srs: torch.Tensor) -> torch.Tensor:
    _, n_channels, _, _ = hrs.shape
    return SSIM(n_channels=n_channels, reduction='none').to(hrs.device)(hrs, srs)


METRICS = dict(MSE=mse,
               PSNR=psnr,
               SSIM=ssim,
               MAE=lambda x, y: torch.mean(torch.abs(x - y), dim=(1, 2, 3)),
               MIXED=partial(mixed_metric, weight=.1))


def calculate_metrics(hrs: torch.Tensor, srs: torch.Tensor, metrics: Union[str, Tuple[str]] = ('SSIM', 'PSNR', 'MSE'),
                      apply_correction: bool = False) -> Union[torch.Tensor, dict]:
    """
    Computes L1/MSE/SSIM/PSNR loss for each instance in a batch.

    :param hrs: tensor (B, C, H, W), high-res images
    :param srs: tensor (B, C, H, W), super resolved images
    :param metrics: name of metrics to compute. If a str, a single tensor is returned, if a list a dictionary of
    metrics is returned
    :param apply_correction: whether to apply brightness correction or not
    :returns scores: tensor (B), metric for each super resolved image.
    """
    metrics_check = [metrics] if isinstance(metrics, str) else metrics
    assert set(metrics_check) == set(metrics_check).intersection(METRICS), \
        f'The only supported metrics are {list(METRICS.keys())}'

    batch, channels, _, _ = srs.shape

    if apply_correction:
        bias = torch.mean(hrs-srs, dim=(2, 3))
        srs = srs + bias.reshape(batch, channels, 1, 1)

    if isinstance(metrics, str):

        return METRICS[metrics](hrs, srs)

    scores = {}

    for metric in metrics:
        scores[metric] = METRICS[metric](hrs, srs)

    return scores


def minshift_loss(hrs: torch.Tensor, srs: torch.Tensor, metric: str,
                  shifts: int = 5, apply_correction: bool = False,) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Computes a metric over shifted versions on the high-resolution image. The minimum over the shifts is returned.

    :param hrs: tensor (B, C, H, W), high-res images
    :param srs: tensor (B, C, H, W), super resolved images
    :param metric: name of metric to compute
    :param shifts: size of shifts in x and y dimensions to consider. All possible (i,j) positions are considered
    :param apply_correction: whether to apply brightness correction or not
    :returns scores: tensor (B), metric for each super resolved image.
    :returns indices: tuple, indices with best alignment between HR and SR image
    """
    _, _, h, w = srs.shape

    border = shifts // 2
    h_, w_ = h - 2*border, w - 2*border

    srs_mid = srs[..., border:-border, border:-border]
    scores = []
    offsets = []

    for i in range(shifts):
        for j in range(shifts):
            hrs_shift = hrs[..., i:i + h_, j:j + w_]

            score = calculate_metrics(hrs_shift, srs_mid, metrics=metric, apply_correction=apply_correction)
            scores.append(score)
            offsets.append((i, i+h_, j, j+w_))

    scores = torch.stack(scores)
    scores, indices = torch.min(scores, dim=0) if metric in ('MAE', 'MSE') else torch.max(scores, dim=0)

    indices = torch.Tensor(offsets)[indices]

    return scores, indices


def compute_perceptual_loss(norm_de: torch.Tensor, norm_sr: torch.Tensor, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute perceptual losses """
    feat_loss, style_loss = [], []
    
    layer_names = {x.split('.')[0] for x in model.state_dict().keys() if 'conv' in x}

        
    for layer_name in layer_names:
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook        
        
        handle = getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
        
        _ = model(norm_de)
        interm_de = activation[layer_name]
        
        _ = model(norm_sr)
        interm_sr = activation[layer_name]
        
        handle.remove() 
        
        feat_loss.append(torch.mean((interm_sr-interm_de)**2))

        nbatch, nfeat, height, width  = interm_de.shape
        
        rav_de = interm_de.reshape(nbatch*height*width, nfeat)
        rav_sr = interm_sr.reshape(nbatch*height*width, nfeat)    

        gram_de = torch.matmul(rav_de.T, rav_de)/(nbatch*height*width)
        gram_sr = torch.matmul(rav_sr.T, rav_sr)/(nbatch*height*width) 
    
        style_loss.append(torch.linalg.norm(gram_sr-gram_de))
        del rav_de, rav_sr, gram_de, gram_sr, interm_de, interm_sr, get_activation, handle

    return torch.stack(feat_loss), torch.stack(style_loss)