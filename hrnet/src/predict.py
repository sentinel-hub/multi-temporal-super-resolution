import torch
from hrnet.src.DeepNetworks.HRNet import HRNet

def get_sr(sample, model):
    """
    Super resolves an imset with a given model.
    Args:
        sample: imset sample
        model: HRNet, pytorch model
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
    """
    
    lrs, alphas, names = sample['lr'], sample['alphas'], sample['name']
    
    if lrs.ndim == 4: 
        nviews, c, h, w = lrs.shape
        lrs = lrs.view(1, nviews, c, h, w) 
        alphas = alphas.view(1, nviews)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)
    sr = model(lrs, alphas)
    sr = sr.detach().cpu().numpy()

    return sr


def load_model(config, checkpoint_file):
    """
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet(config["network"]).to(device)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    return model

class Model(object):
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def load_checkpoint(self, checkpoint_file):
        self.model = load_model(self.config, checkpoint_file)
        
    def __call__(self, sample):
        sr = get_sr(sample, self.model)
        return sr