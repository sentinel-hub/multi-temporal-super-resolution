import numpy as np 
import pandas as pd 
from typing import List 
from datetime import datetime
from dateutil.parser import parse
import torch

def denorm_s2(img: np.ndarray, denom_perc: np.lib.npyio.NpzFile, denorm_country: pd.DataFrame) -> np.ndarray: 
    """ Denormalize normalized Sentinel-2 image """
    norm_median = denorm_country[['median_0', 'median_1', 'median_2', 'median_3']].values
    norm_std = denorm_country[['std_0', 'std_1', 'std_2', 'std_3']].values

    # Denorm percentiles
    img  = np.moveaxis(img.squeeze(), 0, 2)*(denom_perc['p99'] - denom_perc['p1']) + denom_perc['p1']
    img = (img*norm_std)+norm_median 
    
    img = np.expand_dims(np.moveaxis(img, 2, 0), 0)
    return img

def get_closest_timestamp(timestamps: List[datetime], ref_timestamp: datetime) -> datetime:
    """ Get the timestamo closest to the reference timestamp """
    closest_idx = 0
    for i, ts in enumerate(timestamps):
        if abs((ts - ref_timestamp).days) < abs((timestamps[closest_idx] - ref_timestamp).days):
            closest_idx = i
    return timestamps[closest_idx]

def normalise_bands_perceptual(srs, hrs, timestamps, s2_p1, s2_p99, norm_std, norm_median, fd_means, fd_stds):
    
    srs = (srs.permute((0, 2, 3, 1)) * (s2_p99 - s2_p1) + s2_p1) * norm_std + norm_median
    srs = srs.permute(0, 3, 1, 2)
    
    hrs = (hrs.permute((0, 2, 3, 1)) * (s2_p99 - s2_p1) + s2_p1) * norm_std + norm_median
    hrs = hrs.permute(0, 3, 1, 2)
    
    srs_fd_normed = [] 
    hrs_fd_normed = [] 
    for hr, sr, ts in zip(hrs, srs, timestamps):
        
        month = parse(ts).month 
        
        means = fd_means[month]
        stds = fd_stds[month] 

        sr_fd_normalized = ((sr.permute(1, 2, 0)*10000 - means)/stds).permute(2, 0, 1)
        hr_fd_normalized = ((hr.permute(1, 2, 0)*10000 - means)/stds).permute(2, 0, 1)
        
        srs_fd_normed.append(sr_fd_normalized)
        hrs_fd_normed.append(hr_fd_normalized)


    srs = torch.stack(srs_fd_normed).float()
    hrs = torch.stack(hrs_fd_normed).float()
    
    return srs, hrs


