import os
from collections import OrderedDict
from typing import Tuple, List, Callable

from fs_s3fs import S3FS

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from skimage.exposure import match_histograms
from datetime import datetime
from eolearn.core import EOPatch


def augment(
    lr: np.ndarray,
    hr: np.ndarray,
    flip: bool = True,
    rotate: bool = True,
    distribution_shift: bool = False,
    distribution_scale: bool = False,
    permute_timestamps: bool = True,
    max_distribution_shift: float = 0.25,
    max_distribution_scale_diff: float = 0.25,
    proba_of_original: float = 0.67
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a series of image augmentations with specified probability.

    :param lr: array of low-resolution images, shape is `CxTxHxW`
    :param hr: array of high-resolution images, shape is `CxHxW`
    :param flip: whether to randomly flip height or width of arrays
    :param rotate: whether to randomly rotate the arrays
    :param distribution_shift: add an offset to the distribution
    :param distribution_scale: scale the channels distribution
    :param permute_timestamps: permute timestamps (not desired for HRN)
    :param max_distribution_shift: set max distribution shift used in distribution shift augmentation
    :param max_distribution_scale_diff: set max distribution scale used in distribution scale augmentation
    :param proba_of_original: set probability of not modifying original patch, e.g. 1 means no augmetnations
    :returns: augmented lr and hr arrays
    """

    # Base probability which, after `n_aug_conditions`, reduces to `proba_of_original`
    n_aug_conditions = sum(1. for aug_op in (flip, rotate, distribution_shift, distribution_scale, permute_timestamps)
                           if aug_op)
    rng_threshold = proba_of_original ** (1. / n_aug_conditions)

    if flip and np.random.random() > rng_threshold:
        flip_axis = np.random.choice([-2, -1])
        lr = np.flip(lr, axis=flip_axis)
        hr = np.flip(hr, axis=flip_axis)

    if rotate and np.random.random() > rng_threshold:
        k = np.random.choice(np.arange(-2, 3))

        lr = np.rot90(lr, k=k, axes=(-2, -1))
        hr = np.rot90(hr, k=k, axes=(-2, -1))

    if distribution_shift and np.random.random() > rng_threshold:
        d_shift = (np.random.random() - 0.5) * max_distribution_shift

        lr = lr + d_shift
        hr = hr + d_shift

    if distribution_scale and np.random.random() > rng_threshold:
        d_scale = 1. + (np.random.random() - 0.5) * max_distribution_scale_diff

        lr_mean = np.mean(lr, axis=(-2, -1))[..., None, None]
        hr_mean = np.mean(hr, axis=(-2, -1))[..., None, None]

        lr = (lr - lr_mean) * d_scale + lr_mean
        hr = (hr - hr_mean) * d_scale + hr_mean

    if permute_timestamps and np.random.random() > rng_threshold:
        # expects lr in `CxTxHxW` shape
        indices = np.random.permutation(lr.shape[1])
        lr = lr[:, indices]

    return lr, hr


def pad_to_k(feat: np.ndarray, k: int = 16, pad_to_front: bool = True) -> np.ndarray:
    """ Create an array with first dimension equal to k, filling with 0s in front or at back """
    n_pad = k - len(feat)

    if n_pad < 0:
        raise ValueError(f'Can not pad when length of features: {len(feat)} is longer than k: {k}')

    (_, h, w, c) = feat.shape
    if pad_to_front:
        feat = np.concatenate((np.zeros(shape=(n_pad, h, w, c)), feat))
    else:
        feat = np.concatenate((feat, np.zeros(shape=(n_pad, h, w, c))))

    return feat


class ImageSet(OrderedDict):
    """
    An OrderedDict derived class to group the assets of an imageset, with a pretty-print functionality.
    """

    def __init__(self, *args, **kwargs):
        super(ImageSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"

        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        return dict_info


def read_imageset(imset_file: str,
                  filesystem: S3FS = None,
                  normalize: bool = True,
                  country_norm_df: pd.DataFrame = None,
                  norm_deimos_npz: np.lib.npyio.NpzFile = None,
                  norm_s2_npz: np.lib.npyio.NpzFile = None,
                  n_views: int = 16,
                  padding: str = 'zeros',
                  histogram_matching: bool = False) -> ImageSet:
    """
    Retrieves all assets from the given directory.

    :param imset_file: name of npz file with sample imageset
    :param filesystem: S3 filesystem to read files directly from bucket. Default reads from local disk
    :param normalize: whether to normalize data or not
    :param country_norm_df: S2 median/std normalization factors stored per country
    :param norm_deimos_npz: 1st and 99th percentile normalization factors for DEIMOS
    :param norm_s2_npz: 1st and 99th percentile normalization factors for S2
    :param n_views: number of time frames to consider in lrs sequence. If n_views is smaller than the available time
                    frames, `n_views` timeframes from the lrs sequence are taken in reverted order, i.e. last is first
    :param padding: strategy used to fill lrs sequence if n_views is greater than available timestamps. Supported
                    options are `zeros`, where 0 frames are prepended to features, or `repeat` where random repeats of
                    timeframes are taken
    :param histogram_matching: whether to match the histogram between the HR and the corresponding LR image
    """
    assert padding in ['zeros', 'repeat']

    # Read asset names
    npz = np.load(filesystem.openbin(imset_file), allow_pickle=True) if filesystem else np.load(imset_file,
                                                                                                allow_pickle=True)
    
    features = npz['features']
    hr = npz['labels']

    if normalize:
        country = npz['countries']
        country_stats = country_norm_df[country_norm_df.country == str(country)]
        norm_median = country_stats[['median_0', 'median_1', 'median_2', 'median_3']].values

        norm_std = country_stats[['std_0', 'std_1', 'std_2', 'std_3']].values
        features = (features - norm_median) / norm_std

        deimos_p1 = norm_deimos_npz['p1']
        deimos_p99 = norm_deimos_npz['p99']

        s2_p1 = norm_s2_npz['p1']
        s2_p99 = norm_s2_npz['p99']

        hr = (hr - deimos_p1) / (deimos_p99 - deimos_p1)
        features = (features - s2_p1) / (s2_p99 - s2_p1)

    alphas = np.ones(n_views)

    if histogram_matching:
        hr = match_histograms(hr, features[-1], multichannel=True)

    n_feature_timestamps = len(features)
    if n_feature_timestamps < n_views:
        if padding == 'zeros':
            features = pad_to_k(features, n_views, pad_to_front=False)
            alphas[n_feature_timestamps:] = 0
        elif padding == 'repeat':
            n_pad = n_views - n_feature_timestamps
            padded = features[-1:].repeat(n_pad, axis=0)
            features = np.concatenate((features, padded))
    else:
        features = features[-n_views:, ...]

    # Tensor is `CxTxHxW`
    features = np.moveaxis(features, -1, 0)
    hr = np.moveaxis(hr, 2, 0)
    
    imageset = ImageSet(name=os.path.basename(imset_file),
                        timestamp_deimos=str(npz['timetamps_deimos'].item()),
                        lr=features,
                        hr=hr,
                        alphas=alphas)
    return imageset


class ImagesetDataset(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories.

    :param imset_dir: name of directory containing files
    :param imset_npz_files: list of filenames that constitute the dataset
    :param time_first: whether returned lrs sequence should have time dimension first or channels. Use `time_first=True`
                        if you are training HRN model (`BxTxCxHxW`), `time_first=False` if you are training RAMS
                        (`BxTxCxHxW`)
    :param filesystem: S3 filesystem to read files directly from bucket. Default reads from local disk
    :param normalize: whether to normalize data or not
    :param country_norm_df: S2 median/std normalization factors stored per country
    :param norm_deimos_npz: 1st and 99th percentile normalization factors for DEIMOS
    :param norm_s2_npz: 1st and 99th percentile normalization factors for S2
    :param channels_feats: which channels (i.e. indices) are extracted from lrs sequence
    :param channels_labels: which channels (i.e. indices) are extracted from hr image
    :param n_views: number of time frames to consider in lrs sequence. If n_views is smaller than the available time
                    frames, `n_views` timeframes from the lrs sequence are taken in reverted order, i.e. last is first
    :param padding: strategy used to fill lrs sequence if n_views is greater than available timestamps. Supported
                    options are `zeros`, where 0 frames are appended to features, or `repeat` where random repeats of
                    timeframes are taken
    :param transform: function executed on lr and hr arrays as augmentation
    :param histogram_matching: whether to match the histogram between the HR and the corresponding LR image
    """

    def __init__(
            self,
            imset_dir: str,
            imset_npz_files: list,
            time_first: bool,
            filesystem: object = None,
            normalize: bool = True,
            country_norm_df: object = None,
            norm_deimos_npz: np.ndarray = None,
            norm_s2_npz: np.ndarray = None,
            channels_feats: List[int] = [0, 1, 2, 3],
            channels_labels: List[int] = [0, 1, 2, 3],
            n_views: int = 16,
            padding: str = 'zeros',
            transform: Callable = None,
            histogram_matching: bool = False
    ):

        super().__init__()
        self.imset_dir = imset_dir
        self.filesystem = filesystem
        self.imset_npz_files = imset_npz_files
        self.time_first = time_first
        self.normalize = normalize
        self.country_norm_df = country_norm_df
        self.norm_deimos_npz = norm_deimos_npz
        self.norm_s2_npz = norm_s2_npz
        self.channels_feats = channels_feats
        self.channels_labels = channels_labels
        self.n_views = n_views
        self.padding = padding
        self.transform = transform
        self.histogram_matching = histogram_matching

    def __len__(self):
        return len(self.imset_npz_files)

    def __getitem__(self, index: int) -> ImageSet:
        """ Returns an ImageSet dict of all assets in the directory of the given index."""

        if isinstance(index, int):
            imset_file = os.path.join(self.imset_dir, self.imset_npz_files[index])
        else:
            raise KeyError('Index must be of type `int`.')

        imset = read_imageset(
            imset_file=imset_file,
            filesystem=self.filesystem,
            normalize=self.normalize,
            country_norm_df=self.country_norm_df,
            norm_deimos_npz=self.norm_deimos_npz,
            norm_s2_npz=self.norm_s2_npz,
            n_views=self.n_views,
            padding=self.padding,
            histogram_matching=self.histogram_matching
        )

        lr = imset['lr'][self.channels_feats]
        hr = imset['hr'][self.channels_labels]

        if self.transform is not None:
            lr, hr = self.transform(lr, hr)

        if self.time_first:
            lr = np.swapaxes(lr, 0, 1)

        imset['lr'] = torch.from_numpy(lr.copy())
        imset['hr'] = torch.from_numpy(hr.copy())
        imset['alphas'] = torch.from_numpy(imset['alphas'])

        return imset


def filter_cloudy_s2(eop, max_cc):
    idxs = []  
    for i, _ in enumerate(eop.timestamp): 
        if (eop.mask['CLM'][i, ...].mean() <= max_cc) and (eop.mask['IS_DATA'].mean() == 1): 
            idxs.append(i)
    eop.data['BANDS'] = eop.data['BANDS'][idxs, ...]
    eop.data['CLP'] = eop.data['CLP'][idxs, ...]
    eop.mask['CLM'] = eop.mask['CLM'][idxs, ...]
    eop.mask['IS_DATA'] = eop.mask['IS_DATA'][idxs, ...]
    eop.timestamp = list(np.array(eop.timestamp)[idxs])
    return eop 


def timestamps_within_date(timestamps, start_date, end_date): 
    timestamps = [ts.replace(tzinfo=None) for ts in timestamps] # Remove TZINfo that is present in batch
    return [i for i, ts in enumerate(timestamps) if ts >= start_date and ts < end_date]


def read_imageset_eopatch(imset_file: str,
                  start_date: datetime, 
                  end_date: datetime,
                  country: str,
                  filesystem: S3FS = None,
                  normalize: bool = True,
                  country_norm_df: pd.DataFrame = None,
                  norm_s2_npz: np.lib.npyio.NpzFile = None,
                  n_views: int = 16,
                  padding: str = 'zeros', histogram_matching: bool = False) -> ImageSet:
    """
    Retrieves all assets from the given directory.

    :param imset_file: name of npz file with sample imageset
    :param filesystem: S3 filesystem to read files directly from bucket. Default reads from local disk
    :param start_date: specifies the start of the temporal range of the stack of images used for prediction
    :param end_date: specifies the end of the temporal range of the stack of images used for prediction
    :param country: specifies the name of the country so it can be matched with the country_norm_df  
    :param normalize: whether to normalize data or not
    :param country_norm_df: S2 median/std normalization factors stored per country
    :param norm_s2_npz: 1st and 99th percentile normalization factors for S2
    :param n_views: number of time frames to consider in lrs sequence. If n_views is smaller than the available time
                    frames, `n_views` timeframes from the lrs sequence are taken in reverted order, i.e. last is first
    :param padding: strategy used to fill lrs sequence if n_views is greater than available timestamps. Supported
                    options are `zeros`, where 0 frames are prepended to features, or `repeat` where random repeats of
                    timeframes are taken
    """
    assert padding in ['zeros', 'repeat']

    eopatch = EOPatch.load(imset_file, filesystem=filesystem, lazy_loading=True)
    noncloudy = filter_cloudy_s2(eopatch, max_cc=0.1)
    ts_idxs = timestamps_within_date(noncloudy.timestamp, start_date, end_date)
    features = noncloudy.data['BANDS'][ts_idxs, ...] / 10000
    filtered_ts = [eopatch.timestamp[tsi] for tsi in ts_idxs]


    if normalize:
        country_stats = country_norm_df[country_norm_df.country == str(country)]
        norm_median = country_stats[['median_0', 'median_1', 'median_2', 'median_3']].values
        norm_std = country_stats[['std_0', 'std_1', 'std_2', 'std_3']].values
        features = (features - norm_median) / norm_std

        s2_p1 = norm_s2_npz['p1']
        s2_p99 = norm_s2_npz['p99']
        features = (features - s2_p1) / (s2_p99 - s2_p1)

    alphas = np.ones(n_views)
    if histogram_matching:
        hr = match_histograms(hr, features[-1], multichannel=True)


    n_feature_timestamps = len(features)
    if n_feature_timestamps < n_views:
        if padding == 'zeros':
            features = pad_to_k(features, n_views, pad_to_front=False)
            alphas[n_feature_timestamps:] = 0
        elif padding == 'repeat':
            n_pad = n_views - n_feature_timestamps
            padded = features[-1:].repeat(n_pad, axis=0)
            features = np.concatenate((features, padded))
    else:
        features = features[-n_views:, ...]

    # Tensor is `CxTxHxW`
    features = np.moveaxis(features, -1, 0)

    imageset = ImageSet(name=os.path.basename(imset_file),
                        lr=features,
                        alphas=alphas,
			ts=filtered_ts[::-1])
    return imageset


class EopatchPredictionDataset(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories.

    :param imset_dir: name of directory containing files
    :param imset_npz_files: list of filenames that constitute the dataset
    :param time_first: whether returned lrs sequence should have time dimension first or channels. Use `time_first=True`
                        if you are training HRN model (`BxTxCxHxW`), `time_first=False` if you are training RAMS
                        (`BxTxCxHxW`)
    :param filesystem: S3 filesystem to read files directly from bucket. Default reads from local disk
    :param start_date: specifies the start of the temporal range of the stack of images used for prediction
    :param end_date: specifies the end of the temporal range of the stack of images used for prediction
    :param country: specifies the name of the country so it can be matched with the country_norm_df  
    :param normalize: whether to normalize data or not
    :param country_norm_df: S2 median/std normalization factors stored per country
    :param norm_deimos_npz: 1st and 99th percentile normalization factors for DEIMOS
    :param norm_s2_npz: 1st and 99th percentile normalization factors for S2
    :param channels_feats: which channels (i.e. indices) are extracted from lrs sequence
    :param channels_labels: which channels (i.e. indices) are extracted from hr image
    :param n_views: number of time frames to consider in lrs sequence. If n_views is smaller than the available time
                    frames, `n_views` timeframes from the lrs sequence are taken in reverted order, i.e. last is first
    :param padding: strategy used to fill lrs sequence if n_views is greater than available timestamps. Supported
                    options are `zeros`, where 0 frames are appended to features, or `repeat` where random repeats of
                    timeframes are taken
    :param transform: function executed on lr and hr arrays as augmentation
    """

    def __init__(
            self,
            imset_dir: str,
            imset_npz_files: list,
            time_first: bool,
            start_date: datetime,
            end_date: datetime,
            country: str,
            filesystem: object = None,
            normalize: bool = True,
            country_norm_df: object = None,
            norm_deimos_npz: np.ndarray = None,
            norm_s2_npz: np.ndarray = None,
            channels_feats: List[int] = [0, 1, 2, 3],
            n_views: int = 16,
            padding: str = 'zeros',
            histogram_matching: bool = False
    ):

        super().__init__()
        self.imset_dir = imset_dir
        self.filesystem = filesystem
        self.imset_npz_files = imset_npz_files
        self.time_first = time_first
        self.normalize = normalize
        self.country_norm_df = country_norm_df
        self.norm_deimos_npz = norm_deimos_npz
        self.norm_s2_npz = norm_s2_npz
        self.channels_feats = channels_feats
        self.n_views = n_views
        self.padding = padding
        self.start_date = start_date
        self.end_date = end_date
        self.histogram_matching = histogram_matching
        self.country = country

    def __len__(self):
        return len(self.imset_npz_files)

    def __getitem__(self, index: int) -> ImageSet:
        """ Returns an ImageSet dict of all assets in the directory of the given index."""

        if isinstance(index, int):
            imset_file = os.path.join(self.imset_dir, self.imset_npz_files[index])
        else:
            raise KeyError('Index must be of type `int`.')            
            
        imset = read_imageset_eopatch(
            imset_file=imset_file,
            filesystem=self.filesystem,
            normalize=self.normalize,
            country_norm_df=self.country_norm_df,
            norm_deimos_npz=self.norm_deimos_npz,
            norm_s2_npz=self.norm_s2_npz,
            n_views=self.n_views,
            padding=self.padding,
            start_date=self.start_date,
            end_date=self.end_date,
            country=self.country,
            histogram_matching=self.histogram_matching, 
        )

        lr = imset['lr'][self.channels_feats]
        
        if self.time_first:
            lr = np.swapaxes(lr, 0, 1)

        imset['lr'] = torch.from_numpy(lr.copy())
        imset['alphas'] = torch.from_numpy(imset['alphas'])

        return imset

