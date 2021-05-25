''' Pytorch implementation of HomographyNet.
    Reference: https://arxiv.org/pdf/1606.03798.pdf and https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Currently supports translations (2 params)
    The network reads pair of images (tensor x: [B,2*C,W,H])
    and outputs parametric transformations (tensor out: [B,n_params]).

Credits:
This code is adapted from ElementAI's HighRes-Net: https://github.com/ElementAI/HighRes-net
'''

import torch
import torch.nn as nn
from hrnet.src import lanczos


class ShiftNet(nn.Module):
    ''' ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel. '''
    
    def __init__(self, in_channel=4, patch_size=128, num_filters=64, size_linear=1024):
        '''
        Args:
            in_channel : int, number of input channels
        '''
       
        dim_after_conv = patch_size // (2**3) # 2**number_of_maxpools (because you divide dimension by two after each maxpool)
        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, num_filters, 3, padding=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(num_filters, num_filters*2, 3, padding=1),
                                    nn.BatchNorm2d(num_filters*2),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1),
                                    nn.BatchNorm2d(num_filters*2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1),
                                    nn.BatchNorm2d(num_filters*2),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1),
                                    nn.BatchNorm2d(num_filters*2),
                                    nn.ReLU())
        
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(num_filters*2 * dim_after_conv * dim_after_conv, size_linear)
        
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(size_linear, 2, bias=False)
        self.fc2.weight.data.zero_() # init the weights with the identity transformation

    def forward(self, x):
        '''
        Registers pairs of images with sub-pixel shifts.
        Args:
            x : tensor (B, 2, C_in, H, W), input pairs of images
        Returns:
            out: tensor (B, 2), translation params
        '''
        
        #print(f'shift net forward input shape {x.shape}')
        
        batch, nviews, c, h, w = x.shape
     
        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(2, 3)).view(-1, c, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(2, 3)).view(-1, c, 1, 1)
                
        x = x.view(batch, nviews*c, h, w)
        out = self.layer1(x)
        
        out = self.layer2(out)        
        out = self.layer3(out)        
        out = self.layer4(out)        
        out = self.layer5(out)        
        out = self.layer6(out)
        out = self.layer7(out)

        out = self.layer8(out)
        _, feats, dim, _ = out.shape
        
        out = out.view(-1, feats * dim * dim)
        
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    def transform(self, theta, I, device="cpu"):
        '''
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        '''
        
        self.theta = theta
        
        new_I = lanczos.lanczos_shift(img=I,
                                      shift=self.theta.flip(-1),  # (dx, dy) from register_batch -> flip
                                      a=3, p=5)[:, None]
        #print(f'new I shejp: {new_I.shape}')
        return new_I