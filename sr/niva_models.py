import torch.nn as nn
import torch


class TorchUnetv2(nn.Module):
    
    def __init__(self, in_channels, config):
        """
        Args:
            config : dict, configuration file
        """

        super(TorchUnetv2, self).__init__()
        #TODO: do this recursively as done for TF
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels,
                                         config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                    nn.Conv2d(config.features_root,
                                         config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        # acts as pool
        self.conv_pool_1 = nn.Sequential(nn.Conv2d(config.features_root, 
                                         config.features_root,
                                         config.pool_size, 
                                         stride=config.pool_stride,
                                         padding=0),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
    
        self.conv_2 = nn.Sequential(nn.Conv2d(config.features_root,
                                         2*config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                    nn.Conv2d(2*config.features_root,
                                         2*config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        # acts as pool
        self.conv_pool_2 = nn.Sequential(nn.Conv2d(2*config.features_root, 
                                         2*config.features_root,
                                         config.pool_size, 
                                         stride=config.pool_stride,
                                         padding=0),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        self.conv_3 = nn.Sequential(nn.Conv2d(2*config.features_root,
                                         4*config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                    nn.Conv2d(4*config.features_root,
                                         4*config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                    nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        # acts as pool
        self.conv_pool_3 = nn.Sequential(nn.Conv2d(4*config.features_root, 
                                         4*config.features_root,
                                         config.pool_size, 
                                         stride=config.pool_stride,
                                         padding=0),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob))

        self.conv_4 = nn.Sequential(nn.Conv2d(4*config.features_root,
                                         8*config.features_root,
                                         config.conv_size, 
                                         stride=config.conv_stride,
                                         padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                nn.Conv2d(8*config.features_root,
                                          8*config.features_root,
                                          config.conv_size, 
                                          stride=config.conv_stride,
                                          padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(8*config.features_root,
                                                         4*config.features_root,
                                                         config.deconv_size,
                                                         stride=config.deconv_size),
                                      nn.ReLU())
        
        self.conv_5 = nn.Sequential(nn.Conv2d(8*config.features_root,
                                          4*config.features_root,
                                          config.conv_size, 
                                          stride=config.conv_stride,
                                          padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                nn.Conv2d(4*config.features_root,
                                          4*config.features_root,
                                          config.conv_size, 
                                          stride=config.conv_stride,
                                          padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(4*config.features_root,
                                                         2*config.features_root,
                                                         config.deconv_size,
                                                         stride=config.deconv_size), 
                               nn.ReLU())
        
        self.conv_6 = nn.Sequential(nn.Conv2d(4*config.features_root,
                                          2*config.features_root,
                                          config.conv_size, 
                                          stride=config.conv_stride,
                                          padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                nn.Conv2d(2*config.features_root,
                                          2*config.features_root,
                                          config.conv_size, 
                                          stride=config.conv_stride,
                                          padding=config.conv_size//2),
                                  nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(2*config.features_root,
                                                         config.features_root,
                                                         config.deconv_size,
                                                         stride=config.deconv_size), 
                               nn.ReLU())
        
        
        conv_dist_1 = nn.Sequential(nn.Conv2d(2*config.features_root,
                                              config.features_root,
                                              config.conv_size, 
                                              stride=config.conv_stride,
                                              padding=config.conv_size//2),
                                     nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                    nn.Conv2d(config.features_root,
                                              config.features_root,
                                              config.conv_size, 
                                              stride=config.conv_stride,
                                              padding=config.conv_size//2),
                                     nn.ReLU(), nn.Dropout(p=1-config.keep_prob))

        conv_dist_2 = nn.Sequential(nn.Conv2d(config.features_root,
                                              config.n_classes,
                                              1),
                                    nn.Softmax(dim=1))
        self.distance = nn.Sequential(conv_dist_1, conv_dist_2)
        
        conv_bound_1 =  nn.Sequential(nn.Conv2d(2*config.features_root+config.n_classes,
                                                config.features_root,
                                                config.conv_size, 
                                                stride=config.conv_stride,
                                                padding=config.conv_size//2),
                                     nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        conv_bound_2 = nn.Sequential(nn.Conv2d(config.features_root,
                                               config.n_classes,
                                               1),
                                     nn.Softmax(dim=1))
        self.boundary = nn.Sequential(conv_bound_1, conv_bound_2)
        
        conv_extent_1 = nn.Sequential(nn.Conv2d(2*config.features_root+2*config.n_classes,
                                                config.features_root,
                                                config.conv_size, 
                                                stride=config.conv_stride,
                                                padding=config.conv_size//2),
                                     nn.ReLU(), nn.Dropout(p=1-config.keep_prob),
                                      nn.Conv2d(config.features_root,
                                                config.features_root,
                                                config.conv_size, 
                                                stride=config.conv_stride,
                                                padding=config.conv_size//2),
                                     nn.ReLU(), nn.Dropout(p=1-config.keep_prob))
        
        conv_extent_2 = nn.Sequential(nn.Conv2d(config.features_root,
                                                config.n_classes,
                                                1),
                                    nn.Softmax(dim=1))
        self.extent = nn.Sequential(conv_extent_1, conv_extent_2)

    def forward(self, x):
        """
        NIVA model v2 on input features
        Args:
            x : tensor (B, C, W, H)
        Returns:
            extent: tensor (B, C_out, W, H), extent pseudo-probas
            boundary: tensor (B, C_out, W, H), boundary pseudo-probas
            distance: tensor (B, C_out, W, H), distance pseudo-probas
        """
        x_1 = self.conv_1(x)
        x_p1 = self.conv_pool_1(nn.functional.pad(x_1, (0, 2, 0, 2)))
        
        x_2 = self.conv_2(x_p1)
        x_p2 = self.conv_pool_2(nn.functional.pad(x_2, (0, 2, 0, 2)))
        
        x_3 = self.conv_3(x_p2)
        x_p3 = self.conv_pool_3(nn.functional.pad(x_3, (0, 2, 0, 2)))
        
        x_4 = self.conv_4(x_p3)
        x_d4 = self.deconv_1(x_4)
        x_c4 = torch.cat([x_3, x_d4], 1)
        
        x_5 = self.conv_5(x_c4)
        x_d5 = self.deconv_2(x_5)
        x_c5 = torch.cat([x_2, x_d5], 1)
        
        x_6 = self.conv_6(x_c5)
        x_d6 = self.deconv_3(x_6)
        x_c6 = torch.cat([x_1, x_d6], 1)
        
        distance = self.distance(x_c6)
        
        cat_bound = torch.cat([x_c6, distance], 1)
        boundary = self.boundary(cat_bound)
        
        cat_extent = torch.cat([cat_bound, boundary], 1)
        extent = self.extent(cat_extent)
        
        return extent, boundary, distance

