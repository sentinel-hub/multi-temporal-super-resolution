import os
from functools import partial

import numpy as np
import pandas as pd

from sr.data_loader import ImagesetDataset, augment


def test_data_loader(input_folder):
    df = pd.read_parquet(os.path.join(input_folder, 'npz_info.pq'))

    min_max_de = np.load(os.path.join(input_folder, 'deimos_min_max_norm.npz'))
    min_max_s2 = np.load(os.path.join(input_folder, 's2_min_max_norm.npz'))

    norm_per_country = pd.read_parquet(os.path.join(input_folder, 's2_norm_per_country.pq'))

    imset_npz_files = df.singleton_npz_filename.values

    aug_fn = partial(augment, permute_timestamps=False)

    hrn_dataset = ImagesetDataset(imset_dir=input_folder,
                                  filesystem=None,
                                  imset_npz_files=imset_npz_files,
                                  time_first=True,
                                  normalize=True,
                                  country_norm_df=norm_per_country,
                                  norm_deimos_npz=min_max_de,
                                  norm_s2_npz=min_max_s2,
                                  channels_feats=[0, 1, 2, 3],
                                  channels_labels=[0, 1, 2, 3],
                                  n_views=8,
                                  padding='zeros',
                                  transform=aug_fn)

    sample = hrn_dataset[0]
    assert len(hrn_dataset) == 2
    assert sample.keys() == {'name', 'lr', 'hr', 'alphas'}
    assert sample['lr'].numpy().shape == (8, 4, 32, 32)
    assert sample['hr'].numpy().shape == (4, 128, 128)
    assert np.sum(sample['lr'].numpy()[-1]) == 0.
    assert sum(sample['alphas'].numpy()) == df.iloc[0].num_tstamps

    fra_dataset = ImagesetDataset(imset_dir=input_folder,
                                  filesystem=None,
                                  imset_npz_files=imset_npz_files,
                                  time_first=False,
                                  normalize=True,
                                  country_norm_df=norm_per_country,
                                  norm_deimos_npz=min_max_de,
                                  norm_s2_npz=min_max_s2,
                                  channels_feats=[0, 1, 2, 3],
                                  channels_labels=[0, 1, 2, 3],
                                  n_views=9,
                                  padding='repeat',
                                  transform=aug_fn)

    sample = fra_dataset[1]
    assert len(fra_dataset) == 2
    assert sample.keys() == {'name', 'lr', 'hr', 'alphas'}
    assert sample['lr'].numpy().shape == (4, 9, 32, 32)
    assert sample['hr'].numpy().shape == (4, 128, 128)
    assert np.sum(sample['lr'].numpy()[-1]) != 0.
    assert sum(sample['alphas'].numpy()) == 9

    com_dataset = ImagesetDataset(imset_dir=input_folder,
                                  filesystem=None,
                                  imset_npz_files=imset_npz_files,
                                  time_first=True,
                                  normalize=False,
                                  country_norm_df=norm_per_country,
                                  norm_deimos_npz=min_max_de,
                                  norm_s2_npz=min_max_s2,
                                  channels_feats=[0, 1, 2, 3],
                                  channels_labels=[0, 1, 2, 3],
                                  n_views=4,
                                  padding='zeros',
                                  transform=None)

    sample_npz = np.load(os.path.join(input_folder, imset_npz_files[0]), allow_pickle=True)
    sample = com_dataset[0]

    np.testing.assert_equal(sample_npz['features'][-1][..., 0], sample['lr'][-1][0].numpy())
    np.testing.assert_equal(sample_npz['labels'][..., 0], sample['hr'][0].numpy())

