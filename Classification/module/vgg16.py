from platform import release
from statistics import mode
from turtle import forward
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class VGG16(nn.Module):
    # references from https://dgschwend.github.io/netscope/#/preset/vgg-16
    def __init__(self, in_channels=3, ndf=64, stages=[2, 2, 3, 3, 3],
                 fc_layers=3, size=224, num_classes=1000, **ignore_kwargs) -> None:
        super().__init__()
        self.models = nn.ModuleList()
        for idx, stage in enumerate(stages):
            out_channels = ndf * (2 ** idx) if in_channels<512 else 512
            last_layer = True if idx == len(stages)-1 else False
            model = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                              num_layers=stage, last_layer=last_layer)
            in_channels = out_channels
            self.models.append(model)

        in_channels = in_channels * (size // (2 ** len(stages)))
        for i in range(fc_layers):
            out_channels = 4096
            if i == fc_layers-1:
                out_channels = num_classes
            model = FCBlcok(in_channels=in_channels, out_channels=out_channels)
            in_channels = out_channels
            self.models.append(model)
            
        self.out = Out(in_channels=in_channels)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        for model in self.models:
            x = model(x)
        x = self.out(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, last_layer=False) -> None:
        super().__init__()
        model = []
        for _ in range(num_layers):
            model += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm1d(out_channels),
                      nn.ReLU(True)]
            in_channels = out_channels
        model += [nn.MaxPool1d(kernel_size=2, stride=2, padding=0)]
        if last_layer:
            model += [nn.Flatten()]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class FCBlcok(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, outmost=False) -> None:
        super().__init__()
        model = [nn.Linear(in_channels, out_channels)]
        if not outmost:
            model += [nn.ReLU(True), nn.Dropout(dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class Out(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, 2),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
