import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, ch_in, hiddens, ch_out):
        super().__init__()
        
        layers = []
        for hid in hiddens:
            layers.append(nn.Conv2d(ch_in, hid, kernel_size=5, stride=1, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = hid
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(ch_in, ch_out))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
        