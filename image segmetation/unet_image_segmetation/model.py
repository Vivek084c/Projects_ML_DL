import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DoubleConv, self).__init__()
        ''' 
        bias=False as we are using batch norm
        convolution layer with kernal 3, stride 1 and padding 1--> gives the same size output,
        '''

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False) ,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False) ,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, X):
        return self.conv(X)
    

class UNET(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels =1
            features = [64, 128, 256, 512],
            ):
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #creating the down part of the unet 
        for feature in feature:
            self.downs.append(DoubleConv(in_channels, feature)) # we mapp the inchannel to the specific out channel
            in_channels = feature #updating to form the net for each layer set

        #creating the up part for the unet
        for features in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))


        #creating the bottel layers
        self.bottelLayer = DoubleConv(feature[-1], feature[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, X):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottelLayer(x)
        skip_connections = skip_connections[::-1]
        

        for _ in range(0, len(self.ups), 2):
            x = self.ups[_](x)
            skip_connection = skip_connections[_//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[_+1](concat_skip)

        return self.final_conv(x)
    

        
        