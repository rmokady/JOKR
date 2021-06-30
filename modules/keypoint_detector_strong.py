import torch.nn.functional as F
from torch import nn

import models.networks as nets
from modules.util import make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector_strong(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0, args=None):
        super(KPDetector_strong, self).__init__()

        #self.predictor = Hourglass(block_expansion, in_features=num_channels,
        #                           max_features=max_features, num_blocks=num_blocks)

        #self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
        #                    padding=pad)

        self.predictor = nets.define_G(num_channels,num_kp, args.ngf, args.netG, args.norm,
                                        not args.no_dropout, args.init_type, args.init_gain) #, not_mask=False, only_mask=True)

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        #kp = {'value': value}

        return value

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        #feature_map = self.predictor(x)
        #prediction = self.kp(feature_map)

        prediction = self.predictor(x)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        return out, heatmap
