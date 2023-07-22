from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

class resnet_model(nn.Module):
    def __init__(self):
        super(resnet_model, self).__init__()
        # PrepLayer
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # output_size = 32/3/1

        # Layer1        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            
        )
        
        self.resnet1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        

        # Layer2
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),        
        )
        
        # Layer3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride = 1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )

        self.resnet2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        
        self.maxpool2   = nn.MaxPool2d(kernel_size = 4, stride = 2)
                
        self.fc_layer   = nn.Linear(512, 10)


    def forward(self, x):
        x  = self.convblock1(x)
        
        x  = self.convblock2(x)
        r1 = self.resnet1(x)
        x  = x + r1
        
        x  = self.convblock3(x)
        
        x  = self.convblock4(x)
        r2 = self.resnet2(x)
        x  = x + r2
        
        x  = self.maxpool2(x) 
        
        x = x.view(x.size(0), -1)
        x  = self.fc_layer(x) 
        

        #x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


