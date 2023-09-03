import torch
from torchvision.models import resnet50, resnet101, resnet34
import torch.nn as nn
from torchsummary import summary
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import numpy as np

def get_model(sources, arch, device, checkpoint=None, dropout=None, heteroscedastic=False): 
    """ Returns a model suitable for the given sources """

    if sources == "S2S5P": 
        if arch == "CNN":
            return get_S2S5P_ch4_cnn(device, checkpoint, dropout, heteroscedastic)

        elif arch == "UNet":
            return unet(in_channels=13, out_channels=1)

class Head(nn.Module):
    def __init__(self, input_dim, intermediate_dim, dropout_config, heteroscedastic):
        super(Head, self).__init__()
        self.dropout1_p = dropout_config["p_second_to_last_layer"]
        self.dropout2_p = dropout_config["p_last_layer"]
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if heteroscedastic:
            # split the output layer into [mean, sigma2]
            self.fc2 = nn.Linear(intermediate_dim, 2)
        else:
            self.fc2 = nn.Linear(intermediate_dim, 1)
        self.dropout_on = True

    def forward(self, x):
        x = nn.functional.dropout(x, p=self.dropout1_p, training=self.dropout_on)
        x = self.fc1(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, p=self.dropout2_p, training=self.dropout_on)
        x = self.fc2(x)

        return x

    def turn_dropout_on(self, use=True):
        self.dropout_on = use


def get_S2S5P_ch4_cnn(device, checkpoint=None, dropout=None, heteroscedastic=False):
    """ Returns a model with two input streams
    (one for S2, one for S5P) followed by a dense
    regression head """
    backbone_S2 = get_resnet_model(device, checkpoint)
    backbone_S2.fc = nn.Identity()
    backbone_S5P = nn.Sequential(nn.Conv2d(1, 10, 3),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(10, 15, 5),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Flatten(),
                              nn.Linear(544500, 128), 
                             )
    
    #summary(backbone_S5P, (1, 2000, 1500)) 
    if dropout is not None:
        # add dropout to linear layers of regression head
        head = Head(2048+128, 544, dropout, heteroscedastic)
        head.turn_dropout_on()
    else:
        head = nn.Sequential(nn.Linear(2048+128, 544), nn.ReLU(), nn.Linear(544, 750000))
    regression_model = MultiBackboneRegressionHead(backbone_S2, backbone_S5P, head)
    return regression_model

def get_resnet_model(device, checkpoint=None):
    """
    create a resnet50 model, optionally load pretrained checkpoint
    and pass it to the device
    """
    model = resnet50(pretrained=False, num_classes=19) 

    model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), bias=False)
    #summary(model, (12, 2000, 1500))

    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    return model

class MultiBackboneRegressionHead(nn.Module):
    """ Wrapper class that combines features extracted
    from two inputs (S2 and S5P) with a regression head """
    def __init__(self, backbone_S2, backbone_S5P, head):
        super(MultiBackboneRegressionHead, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head = head
        self.use_dropout = True

    def forward(self, x):
        
        s5p = x.get("s5p")
        x = x.get("s2")

        x = self.backbone_S2(x) 
        s5p = self.backbone_S5P(s5p)
        x = torch.cat((x, s5p), dim=1)
        x = self.head(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x) 

class unet(nn.Module): # adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
    def __init__(self, in_channels=13, out_channels=1, features=[64, 128, 256, 512]):
        super(unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        s5p = x.get("s5p")
        s2 = x.get("s2")
        s2s5p = torch.cat((s5p, s2), 1) # combine the S5P and S2 data to begin with 13 x 2000 x 1500 input image shape
    
        for down in self.downs:
            s2s5p = down(s2s5p)
            skip_connections.append(s2s5p)
            s2s5p = self.pool(s2s5p)

        s2s5p = self.bottleneck(s2s5p)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            s2s5p = self.ups[idx](s2s5p)
            skip_connection = skip_connections[idx//2]

            if s2s5p.shape != skip_connection.shape:
                s2s5p = functional.resize(s2s5p, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, s2s5p), dim=1)
            s2s5p = self.ups[idx+1](concat_skip)

        return self.final_conv(s2s5p)
