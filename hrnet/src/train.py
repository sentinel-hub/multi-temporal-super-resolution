""" Python script to train HRNet + shiftNet for multi frame super resolution (MFSR) """

import os
import gc
import json
import argparse
import datetime
from functools import partial
from collections import defaultdict, deque

import numpy as np

import cv2 as cv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sr.metrics import calculate_metrics
from hrnet.src.DeepNetworks.HRNet import HRNet
from hrnet.src.DeepNetworks.ShiftNet import ShiftNet
from hrnet.src.utils import normalize_plotting
from hrnet.src.utils import distributions_plot
from sr.data_loader import ImagesetDataset, augment
from sr.metrics import compute_perceptual_loss

from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import wandb


def register_batch(shiftNet, lrs, reference):
    """
    Registers images against references.
    Args:
        shiftNet: torch.model
        lrs: tensor (batch size, views, C, W, H), images to shift
        reference: tensor (batch size, 1, C, W, H), reference images to shift
    Returns:
        thetas: tensor (batch size, views, 2)
    """

    n_views = lrs.size(1)
    thetas = []
    for i in range(n_views):
        # Add references as views
        concated = torch.cat([reference, lrs[:, i:i + 1]], 1)
        theta = shiftNet(concated)
        thetas.append(theta)

    thetas = torch.stack(thetas, 1)
    return thetas


def apply_shifts(shiftNet, images, thetas, device):
    """
    Applies sub-pixel translations to images with Lanczos interpolation.
    Args:
        shiftNet: torch.model
        images: tensor (batch size, views, C, W, H), images to shift
        thetas: tensor (batch size, views, 2), translation params
    Returns:
        new_images: tensor (batch size, views, C, W, H), warped images
    """

    batch_size, n_views, channels, height, width = images.shape
    images = images.view(-1, channels, height, width)
    thetas = thetas.view(-1, 2)

    new_images = shiftNet.transform(thetas, images, device=device)

    return new_images.view(-1, n_views, channels, images.size(2), images.size(3))


def resize_batch_images(batch, fx=3, fy=3, interpolation=cv.INTER_CUBIC):
    resized = torch.tensor([cv.resize(np.moveaxis(img.detach().cpu().numpy(), 0, 2), None,
                            fx=fx, fy=fy, interpolation=interpolation) for img in batch])

    # The channel dimension was 1 and was lost by opencv...
    if resized.ndim < batch.ndim:
        b, w, h = resized.shape
        return resized.view(b, 1, w, h)

    return resized.permute([0, 3, 1, 2])


def save_per_sample_scores(val_score_lists, baseline_val_score_lists, val_names, filename):
    out_dict = {}

    for metric, batch_scores in val_score_lists.items():
        out_dict[metric] = np.concatenate(batch_scores).tolist()

    for metric, batch_scores in baseline_val_score_lists.items():
        out_dict[metric] = np.concatenate(batch_scores).tolist()

    out_dict['name'] = val_names

    with open(filename, 'w') as out_file:
        json.dump(out_dict, out_file)


def trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, config, perceptual_loss_model=None):
    """
    Trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        fusion_model: torch.model, HRNet
        regis_model: torch.model, ShiftNet
        optimizer: torch.optim, optimizer to minimize loss
        dataloaders: dict, wraps train and validation dataloaders
        config: dict, configuration file
        perceptual_loss_model: model used for perceptual loss 
    """

    # Set params from config
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    loss_metric = config['training']['loss_metric']
    val_metrics = config['training']['validation_metrics']
    apply_correction = config['training']['apply_correction']
    use_reg_regularisation = config['training']['use_reg_regularization']
    lambda_ = config['training']['lambda']
    use_kl_div_loss = config['training']['use_kl_div_loss']
    eta_ = config['training']['eta']
    upscale_factor = config['network']['upscale_factor']

    reg_offset = config['training']['reg_offset']
    plot_chnls = config['visualization']['channels_to_plot']
    distribution_sampling_proba = config['visualization']['distribution_sampling_proba']

    assert loss_metric in ['MAE', 'MSE', 'SSIM', 'MIXED']

    # Logging
    subfolder_pattern = 'batch_{}_time_{}'.format(batch_size, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    if config['training']['wandb']:
        wandb.watch(fusion_model)
        wandb.watch(regis_model)

        out_fusion = os.path.join(wandb.run.dir, 'HRNet.pth')
        out_regis = os.path.join(wandb.run.dir, 'ShiftNet.pth')
        out_val_scores = os.path.join(wandb.run.dir, 'val_scores.json')
    else:
        checkpoint_dir_run = os.path.join(config['paths']['checkpoint_dir'], subfolder_pattern)
        scores_dir_run = os.path.join(config['paths']['scores_dir'], subfolder_pattern)

        os.makedirs(checkpoint_dir_run, exist_ok=True)
        os.makedirs(scores_dir_run, exist_ok=True)

        out_fusion = os.path.join(checkpoint_dir_run, 'HRNet.pth')
        out_regis = os.path.join(checkpoint_dir_run, 'ShiftNet.pth')
        out_val_scores = os.path.join(scores_dir_run, 'val_scores.json')

    tb_logging_dir = config['paths']['tb_log_file_dir']
    logging_dir = os.path.join(tb_logging_dir, subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    writer = SummaryWriter(logging_dir)

    # Set backend
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')

    fusion_model.to(device)
    regis_model.to(device)

    # Iterate
    best_score_loss = np.Inf
    val_names_saved = False
    val_names = deque()
    
    for epoch in tqdm(range(0, num_epochs), desc='Epochs'):
        # Set train mode
        fusion_model.train()
        regis_model.train()

        # Reset epoch loss
        train_loss = 0.
        train_loss_reg = 0.
        train_loss_kl = 0.
        train_loss_perceptual = 0. 

        for sample in tqdm(dataloaders['train'], desc='Training iter. %d' % epoch):
            # Reset parameter gradients
            optimizer.zero_grad()

            # Potentially transfer data to GPU
            lrs = sample['lr'].float().to(device, non_blocking=True)
            alphas = sample['alphas'].float().to(device, non_blocking=True)
            lrs_last = lrs[np.arange(len(alphas)), torch.sum(alphas, dim=1, dtype=torch.int64) - 1]
            hrs = sample['hr'].float().to(device, non_blocking=True)

            # Fuse multiple frames into (B, 1, upscale_factor*W, upscale_factor*H)
            srs = fusion_model(lrs, alphas)
            batch, c, w, h = srs.shape
            srs = srs.view(batch, 1, c, w, h)
            
            # Register batch wrt HR
            shifts = register_batch(
                regis_model,
                srs[:, :, :, reg_offset:-reg_offset, reg_offset:-reg_offset],
                reference=hrs[:, :, reg_offset:-reg_offset, reg_offset:-reg_offset].view(-1, 1,
                                                                                         c,
                                                                                         h-2*reg_offset,
                                                                                         w-2*reg_offset))

            srs_shifted = apply_shifts(regis_model, srs, shifts, device)[:, 0]

            # Training loss
            scores = calculate_metrics(hrs=hrs,
                                       srs=srs_shifted,
                                       metrics=loss_metric,
                                       apply_correction=apply_correction)
            loss = torch.mean(scores)
            
            if loss_metric == 'SSIM':
                loss = -1 * loss + 1

            loss_registration = torch.mean(torch.linalg.norm(shifts, ord=2, dim=1))
           
            if use_reg_regularisation:
                loss += lambda_ * loss_registration

            srs = srs.view(batch, c, w, h)

            kl_losses = 0
            if use_kl_div_loss:
                for nc in np.arange(c):
                    mean_diffs = srs[:, nc, ...].mean(dim=(1, 2)) - lrs_last[:, nc, ...].mean(dim=(1, 2))
                    tmp_kl_loss = torch.abs(mean_diffs).mean()
                    loss += eta_ * tmp_kl_loss
                    kl_losses += tmp_kl_loss
                    del tmp_kl_loss
                kl_losses /= c

            # adding perceptual loss 
            if perceptual_loss_model: 
                feat_perceptual_loss, style_perceptual_loss = compute_perceptual_loss(hrs, 
                                                                                      srs, 
                                                                                      perceptual_loss_model)
                perceptual_loss  = feat_perceptual_loss + style_perceptual_loss
                loss += config["perceptual_loss"]["weight"] * perceptual_loss
                del feat_perceptual_loss, style_perceptual_loss
                
            # Backprop
            loss.backward()
            optimizer.step()

            # Scale loss so that epoch loss is the average of batch losses
            num_batches = len(dataloaders['train'].dataset) / len(hrs)
            train_loss += loss.detach().item() / num_batches
            train_loss_reg += loss_registration.detach().item() / num_batches
            train_loss_kl += kl_losses / num_batches
            train_loss_perceptual += perceptual_loss.detach().item() / num_batches

            # Try releasing some memory
            del lrs, alphas, hrs, srs, srs_shifted, scores, loss, loss_registration, sample, kl_losses, perceptual_loss
            gc.collect()
            torch.cuda.empty_cache()

        # Set eval mode
        fusion_model.eval()

        val_scores = defaultdict(float)
        baseline_val_scores = defaultdict(float)

        val_score_lists = defaultdict(list)
        baseline_val_score_lists = defaultdict(list)

        lrs_ref = None
        hrs_ref = None
        srs_ref = None
        lin_interp_img = None
        distribution_s2, distribution_deimos, distribution_sr = [], [], [] 

        # Run validation
        with torch.no_grad():
            for sample in tqdm(dataloaders['val'], desc='Valid. iter. %d' % epoch):
                # Potentially transfer data to GPU
                lrs_cpu = sample['lr'].float()
                hrs_cpu = sample['hr'].float()

                lrs = lrs_cpu.to(device, non_blocking=True)
                hrs = hrs_cpu.to(device, non_blocking=True)
                alphas = sample['alphas'].float().to(device, non_blocking=True)

                # Inference
                srs = fusion_model(lrs, alphas)
                
                if np.random.random() < distribution_sampling_proba: # sampling....
                    for lr, hr, sr, a in zip(lrs_cpu, hrs_cpu, srs.cpu().numpy(), sample['alphas']):
                        num_valid = torch.sum(a, dim=0, dtype=torch.int64).int()
                        distribution_s2.append(lr[:num_valid, ...])
                        distribution_deimos.append(np.expand_dims(hr, 0))
                        distribution_sr.append(np.expand_dims(sr, 0))
                        
                
                # Update scores
                metrics = calculate_metrics(hrs, srs, val_metrics, apply_correction)
                for metric, batch_scores in metrics.items():
                    batch_scores = batch_scores.cpu()
                    val_scores[metric] += torch.sum(batch_scores).item()
                    val_score_lists[metric].append(batch_scores.numpy().flatten())

                # First val. iter.: add names, calculate baseline
                if not val_names_saved:
                    val_names.append(sample['name'])

                    latest_s2_images = lrs_cpu[np.arange(len(alphas)), torch.sum(alphas, dim=1, dtype=torch.int64) - 1]
                    lin_interp_imgs = resize_batch_images(latest_s2_images, fx=upscale_factor, fy=upscale_factor)
                    baseline_metrics = calculate_metrics(hrs_cpu, lin_interp_imgs, val_metrics, apply_correction)

                    for metric, batch_scores in baseline_metrics.items():
                        batch_scores = batch_scores.cpu()
                        baseline_val_scores[f'{metric}_baseline'] += torch.sum(batch_scores).item()
                        baseline_val_score_lists[f'{metric}_baseline'].append(batch_scores.numpy().flatten())

                    del baseline_metrics

                # Keep a reference for plotting
                if lrs_ref is None:
                    lrs_ref = lrs_cpu[0].numpy()
                    hrs_ref = hrs_cpu[0].numpy()
                    lin_interp_img = lin_interp_imgs[0].numpy()

                if srs_ref is None:
                    srs_ref = srs[0].cpu().numpy()

                # Try releasing some memory
                del lrs_cpu, hrs_cpu, lrs, alphas, hrs, srs, metrics, batch_scores, sample
                gc.collect()
                torch.cuda.empty_cache()

        s2 = np.concatenate(distribution_s2)
        deimos = np.concatenate(distribution_deimos)
        sresolved = np.concatenate(distribution_sr)
        
        # Compute the average scores per sample (note the sum instead of the mean above)
        n = len(dataloaders['val'].dataset)

        for metric in val_scores:
            val_scores[metric] /= n

        # Validation file identifiers
        if not val_names_saved:
            val_names = np.concatenate(val_names).tolist()
            val_names_saved = True

            for metric in baseline_val_scores:
                baseline_val_scores[metric] /= n

        # Save improved model
        val_loss_metric = loss_metric if not use_kl_div_loss else 'SSIM'
        val_scores_loss = val_scores[val_loss_metric]
        if val_loss_metric == 'SSIM':
            val_scores_loss = -1 * val_scores_loss + 1

        if val_scores_loss < best_score_loss:
            print('Saving model (val. loss has improved).')

            torch.save(fusion_model.state_dict(), out_fusion)
            torch.save(regis_model.state_dict(), out_regis)

            save_per_sample_scores(val_score_lists, baseline_val_score_lists, val_names, out_val_scores)
            best_score_loss = val_scores_loss

        # Plotting
        lrs = lrs_ref
        srs = srs_ref
        hrs = hrs_ref

        normalized_srs = (srs - np.min(srs)) / np.max(srs)
        normalized_plot = normalized_srs[plot_chnls, ...]
        lrs_plot = np.array([normalize_plotting(x[plot_chnls, ...]) for x in lrs if np.any(x)])
        error_map = hrs - srs
        writer.add_image('SR Image', normalize_plotting(normalized_plot), epoch, dataformats='HWC')
        writer.add_image('Error Map', normalize_plotting(error_map[plot_chnls, ...]), epoch, dataformats='HWC')
        writer.add_image('HR GT', normalize_plotting(hrs[plot_chnls, ...]), epoch, dataformats='HWC')
        writer.add_images('S2', np.moveaxis(lrs_plot, 3, 1), epoch, dataformats='NCHW')
        writer.add_scalar('train/loss', train_loss, epoch)

        for metric in val_metrics:
            writer.add_scalar('val/%s' % metric.lower(), val_scores[metric], epoch)

        # wandb
        if config['training']['wandb']:            
            wandb.log({'Train loss': train_loss,
                       'Train loss registration': train_loss_reg,
                       'Train KL loss': train_loss_kl, 
                       'Train loss perceptual': train_loss_perceptual}, step=epoch)
            wandb.log({'sr': [wandb.Image(normalize_plotting(normalized_plot), caption='SR Image')]}, step=epoch)
            wandb.log({'gt': [wandb.Image(normalize_plotting(hrs[plot_chnls, ...]), caption='HR GT')]}, step=epoch)
            wandb.log({'S2': [wandb.Image(x, caption='S2 GT') for x in lrs_plot]}, step=epoch)
            wandb.log({'S2 bilinear interpolation baseline': [wandb.Image(normalize_plotting(lin_interp_img[plot_chnls, ...]))]}, step=epoch)
            wandb.log({'Distributions': [wandb.Image(normalize_plotting(hrs[plot_chnls, ...]), caption='HR GT')]}, step=epoch)
            wandb.log({'Distribution Blue': [wandb.Image(distributions_plot(s2, deimos, sresolved, 0), caption='Band Blue')],
                       'Distribution Green': [wandb.Image(distributions_plot(s2, deimos, sresolved, 1), caption='Band Green')],
                       'Distributions Red':  [wandb.Image(distributions_plot(s2, deimos, sresolved, 2), caption='Band Red')],
                       'DIstributions NIR': [wandb.Image(distributions_plot(s2, deimos, sresolved, 3), caption='Band NIR')]
                      }, step=epoch)
            wandb.log(val_scores, step=epoch)
            wandb.log(baseline_val_scores, step=epoch)

        del lrs, srs, hrs, lrs_ref, srs_ref, hrs_ref
        del val_scores, baseline_val_scores, val_score_lists, baseline_val_score_lists
        gc.collect()

    writer.close()


def main(
    config, data_df, filesystem=None, country_norm_df=None, normalize=True, norm_deimos_npz=None, norm_s2_npz=None, perceptual_loss_model=None, 
    fusion_model = None, regis_model = None
):
    """
    Given a configuration, trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """

    # Reproducibility options
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    channels_labels = config['training']['channels_labels']
    channels_features = config['training']['channels_features']

    lr_patch_size = config['training']['patch_size']
    upscale_factor = config['network']['upscale_factor']
    reg_offset = config['training']['reg_offset']
    histogram_matching = config['training']['histogram_matching']

    # Initialize the network based on the network configuration
    
    if fusion_model is None:
        fusion_model = HRNet(config["network"])
    if regis_model is None: 
        regis_model = ShiftNet(in_channel=len(channels_labels),
                           patch_size=lr_patch_size*upscale_factor - 2*reg_offset)

    optimizer = optim.Adam(list(fusion_model.parameters()) + list(regis_model.parameters()),
                           lr=config['training']['lr'])

    data_directory = config['paths']['prefix']

    # Dataloaders
    batch_size = config['training']['batch_size']
    n_workers = config['training']['n_workers']
    n_views = config['training']['n_views']
    use_augment = config['training']['augment']


    aug_fn = partial(augment, permute_timestamps=False) if use_augment else None

    # Train data loader
    train_samples = data_df[data_df.train_test_validation == 'train'].singleton_npz_filename.values

    train_dataset = ImagesetDataset(
        imset_dir=data_directory,
        imset_npz_files=train_samples,
        time_first=True,
        filesystem=filesystem,
        country_norm_df=country_norm_df,
        normalize=normalize,
        norm_deimos_npz=norm_deimos_npz,
        norm_s2_npz=norm_s2_npz,
        channels_labels=channels_labels,
        channels_feats=channels_features,
        n_views=n_views,
        padding='zeros',
        transform=aug_fn,
        histogram_matching=histogram_matching)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True)

    # Validation data loader
    validation_samples = data_df[data_df.train_test_validation == 'validation'].singleton_npz_filename.values

    val_dataset = ImagesetDataset(
        imset_dir=data_directory,
        imset_npz_files=validation_samples,
        time_first=True,
        filesystem=filesystem,
        country_norm_df=country_norm_df,
        normalize=normalize,
        norm_deimos_npz=norm_deimos_npz,
        norm_s2_npz=norm_s2_npz,
        channels_labels=channels_labels,
        channels_feats=channels_features,
        n_views=n_views,
        padding='zeros',
        transform=None, 
        histogram_matching=False)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # Train model
    torch.cuda.empty_cache()
    trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, config, perceptual_loss_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path of the config file', default='config/config.json')

    args = parser.parse_args()
    assert os.path.isfile(args.config)

    with open(args.config, 'r') as read_file:
        config = json.load(read_file)

    main(config)
