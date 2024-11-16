import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LeNet_CIFAR10(nn.Module): 
    def __init__(self, num_classes=10): 
        super(LeNet_CIFAR10, self).__init__() 
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1) 
        self.fc1 = nn.Linear(2048, 512) 
        self.fc2 = nn.Linear(512, num_classes) 
 
    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2), 2) 
        x = F.relu(self.conv3(x)) 
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2), 2) 
        x = x.view(-1, int(x.nelement() / x.shape[0])) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x
