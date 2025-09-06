import torch
import torch.nn as nn
from torchvision import models

class DeepLabV3Plus(nn.Module):
    def __init__(self, config):
        super(DeepLabV3Plus, self).__init__()
        self.config = config

        if config.backbone == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Use layer0 (64 channels) for skip connection
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 channels
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPP(2048, [6, 12, 18], 256)
        self.decoder = Decoder(256, 64, config.num_classes)  # Make sure skip_channels=64

        self._init_weights()

    def forward(self, x, adapt=False):
        # Debug input shape
        assert x.dim() == 4, f"Input must be 4D (got {x.dim()}D), shape: {x.shape}"

        # Example forward pass for a DeepLab-like model
        x0 = self.layer0(x)          # Low-level features
        x = self.layer1(x0)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_aspp = self.aspp(x)
        x_decoder = self.decoder(x_aspp, x0)  # Use skip connection

        if adapt:
            # Return extra features for adaptation if needed
            return x_decoder, x_aspp
        return x_decoder
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # Image-level features (use GroupNorm instead of BatchNorm2d)
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # <-- FIX: use GroupNorm here
            nn.ReLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            # Check if the first module is AdaptiveAvgPool2d
            first_module = next(conv.children())
            if isinstance(first_module, nn.AdaptiveAvgPool2d):
                h, w = x.size()[2:]
                pool = conv(x)
                res.append(nn.functional.interpolate(pool, size=(h, w), mode='bilinear', align_corners=True))
            else:
                res.append(conv(x))
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Decoder, self).__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(256, out_channels, 1)

    def forward(self, x, skip):
        # Save input size before downsampling
        input_size = skip.shape[2:]  # Get HxW from early layer (x0)

        skip = self.skip_conv(skip)

        x = nn.functional.interpolate(
            x,
            size=skip.size()[2:], mode='bilinear', align_corners=True
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Final upsample to match original input size (via skip or x0)
        output_size = skip.size()[2:]  # height, width from skip connection
        x = nn.functional.interpolate(
            x, size=output_size, mode='bilinear', align_corners=True
        )
        return x