import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torchsummary import summary

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class ITrackerModel(nn.Module):


    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeleft1 = ItrackerImageModel()
        self.eyeleft2 = ItrackerImageModel()
        self.eyeright1 = ItrackerImageModel()
        self.eyeright2 = ItrackerImageModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(4*8*14*19, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )

    def forward(self, img0, img1,img3,img4):
        # Eye nets
        xEyeL1 = self.eyeleft1(img0)
        xEyeL2 = self.eyeleft2(img1)
        xEyeR1 = self.eyeright1(img3)
        xEyeR2 = self.eyeright2(img4)

        # Cat and FC
        # xEyes = torch.cat((xEyeL, xEyeR), 1)
        # xEyes = self.eyesFC(xEyes)
        x = torch.cat([xEyeL1, xEyeL2, xEyeR1, xEyeR2], dim=1)  
        x = self.eyesFC(x)
        x = self.fc(x)
        return x





if __name__ == "__main__":
    summary(ItrackerImageModel(), input_size=(1, 576, 720))