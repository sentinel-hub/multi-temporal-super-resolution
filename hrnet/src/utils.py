""" Python utilities """
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def distributions_plot(s2_values: np.ndarray, deimos_values: np.ndarray, sr_values: np.ndarray, band: int ) -> np.ndarray:
    """ 
    Return plot image histogram of S2, Deimos and SR distributions for a particular band. 
    
    s2_values: np.ndarray: NxCxWxH array of S2 values
    deimos_values: np.ndarray: NxCxWxH array of deimos values
    sr_values: np.ndarray: NxCxWxH array of predicted super resolved images  
    
    return: np.ndarray: matplotlib plot of the distributions converted to numpy array image (so it can be passed to WANDB).
    """ 
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(15, 7), dpi=300)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(s2_values[:, band, ...].flatten(), alpha=.33, bins=100, range=(-2, 2), label='S2', density=True)
    ax.hist(deimos_values[:, band, ...].flatten(), alpha=.33, bins=100, range=(-2, 2), label='Deimos', density=True)
    ax.hist(sr_values[:, band, ...].flatten(), alpha=.33, bins=100, label='SR', range=(-2, 2), density=True, histtype='step')
    ax.set_title(f'Band {band}')
    ax.legend()

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    return X



def normalize_plotting(rgb): 
    """
    Rescales the data between 0 and 1. 
    
    rgb: np.ndarray:  CxHxW array of bands.  
    """
    n_channels =  len(rgb)
    rgb = np.moveaxis(rgb, 0, 2)
    min_rgb = np.ones(n_channels)*(-1)
    max_rgb = np.ones(n_channels)
    rgb_0_1 = (rgb - min_rgb) / (np.abs(min_rgb) + max_rgb)  
    return rgb_0_1
