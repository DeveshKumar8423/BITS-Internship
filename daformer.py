import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from Domain-Adversarial Training of Neural Networks
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)

class FeatureNormalization(nn.Module):
    """Normalize features to improve stability"""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)  # Back to (B, C, H, W)
        return x

class DaFormer(nn.Module):
    def __init__(self, config):
        super(DaFormer, self).__init__()
        self.num_classes = config.num_classes
        
        # Enhanced backbone with feature extraction
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels
        
        # Feature normalization for stability
        self.feature_norm = FeatureNormalization(2048)
        
        # Enhanced decoder with better stability
        self.decoder = EnhancedDecoder(2048, 256, self.num_classes)
        
        # Gradient reversal layer for domain adaptation
        self.grl = GradientReversalLayer()
        
        # Improved domain discriminator with batch normalization
        self.domain_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, adapt=False, alpha=0.0):
        # Encoder forward pass
        x0 = self.layer0(x)  # 64 channels
        x = self.maxpool(x0)
        x1 = self.layer1(x)  # 256 channels
        x2 = self.layer2(x1) # 512 channels
        x3 = self.layer3(x2) # 1024 channels
        x4 = self.layer4(x3) # 2048 channels - main features
        
        # Normalize features for stability
        x4_norm = self.feature_norm(x4)
        
        # Decode with skip connections
        logits = self.decoder(x4_norm, x1)  # Use skip connection from layer1
        
        if adapt:
            # Apply gradient reversal for domain adaptation
            reversed_features = self.grl(x4_norm, alpha)
            domain_pred = self.domain_discriminator(reversed_features)
            return logits, domain_pred
        else:
            return logits, None

class EnhancedDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, num_classes):
        super(EnhancedDecoder, self).__init__()
        
        # Multi-scale feature processing
        self.aspp = ASPP(in_channels, [6, 12, 18], 512)
        
        # Process skip connection
        self.conv_skip = nn.Sequential(
            nn.Conv2d(skip_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Enhanced fusion with attention
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + 128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        # Attention mechanism for better feature integration
        self.attention = ChannelAttention(256)
        
        # Refinement layers
        self.refine_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        self.refine_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(128, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, main_features, skip_features):
        # Multi-scale feature processing
        main = self.aspp(main_features)
        
        # Upsample main features to match skip features size
        main = F.interpolate(main, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Process skip features
        skip = self.conv_skip(skip_features)
        
        # Fuse features
        fused = torch.cat([main, skip], dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply attention
        fused = self.attention(fused)
        
        # Refinement
        refined = self.refine_conv1(fused)
        refined = self.refine_conv2(refined)
        
        # Final classification
        logits = self.classifier(refined)
        
        return logits

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels//4, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels//4),
                nn.ReLU(inplace=True)
            ))
        
        # Image-level features
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels//4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            # Check if the first module is AdaptiveAvgPool2d
            first_module = next(conv.children())
            if isinstance(first_module, nn.AdaptiveAvgPool2d):
                h, w = x.size()[2:]
                pool = conv(x)
                res.append(F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=False))
            else:
                res.append(conv(x))
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Attention weights
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)