import torch
import torch.nn as nn
import numpy as np
from math import ceil

"""
Implementation of EfficientNet, as proposed in https://arxiv.org/pdf/1905.11946.pdf
"""

# base model configuration list, only the backbone tho which are inverted res blocks (aka mobile blocks)
base_model = [
    # key: expand ratio (see MB Block def), output channels, #num repeat, stride, kernel size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

scaling_params = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                             bias=False)  # dont use bias since we use batchnorm
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()  # SiLU(x) = x*sigmoid(x)

    def forward(self, x):
        return self.activation(self.batchnorm(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # adaptive pool can transform ANY input dim to specified output dim
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InverseResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4,  # reduction factor for squeeze excitation
                 survival_prob=0.8  # for stochastic depth
                 ):
        super(InverseResBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1  # cannot use skip connection if dims change
        self.hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != self.hidden_dim
        self.reduced_dim = int(in_channels / reduction)

        if self.expand:  # expand to hidden dim if expand != 1
            self.expand_conv = CNNBlock(
                in_channels, self.hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(self.hidden_dim, self.hidden_dim, kernel_size, stride, padding,
                     groups=self.hidden_dim),  # depth-wise conv
            SqueezeExcitation(self.hidden_dim, self.reduced_dim),
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, bias=False),  # restore out channels
            nn.BatchNorm2d(out_channels)
        )

    def stochastic_depth(self, x):
        if not self.training:  # training defined in nn.Module
            return x
        # randomly drop the skip connection FOR SPECIFIC EXAMPLES!
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        # divide x by survival prob to maintain mean and std deviation in minibatch since thats used in batchnorm
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """
    implement efficient net according to paper, using the classes defined above
    """
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor) # final layer of baseline has 1280 channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):  # according to formula 3 in paper, resolution fixed so dont use gamma
        phi, resolution, drop_rate = scaling_params[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        firstMBLayerChannels = int(32 * width_factor)  # first MB layer in base model has 32 in channels
        features = [CNNBlock(3, firstMBLayerChannels, kernel_size=3, stride=2, padding=1)]
        in_channels = firstMBLayerChannels

        # add inverse Res Blocks according to the baseline specified above
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = int(channels * width_factor)
            # reduction factor for squeeze excitation is 4, so ensure out channels can be divided by 4:
            out_channels = 4 * ceil(out_channels / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InverseResBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1, # only downsample once
                        kernel_size=kernel_size,
                        padding=kernel_size//2,  # effectively makes it so that only stride defines downsampling, eg k=3: padding 1

                    )
                )

                in_channels = out_channels  # update channels after one iteration of the repetition

        features.append(CNNBlock(in_channels, last_channels, 1, 1, 0))  # final conv layer

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))  # flatten before lin layer, output shape = (B, num_classes)


def test(version):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    version = version
    _, res, _ = scaling_params[version]
    num_examples, num_classes = 4, 10
    x = torch.rand((num_examples, 3, res, res)).to(device)
    model = EfficientNet(version, num_classes).to(device)

    print(model(x).shape)


if __name__ == "__main__":
    test("b0")