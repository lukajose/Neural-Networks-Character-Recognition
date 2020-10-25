# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        input_size = 784 # 28x28 = 784 x 1
        self.linear = nn.Linear(input_size,10)
        

        

    def forward(self, x):
        x = x.view(x.shape[0], -1) #flatten image into 784x1
        x = self.linear(x)
        return F.log_softmax(x,dim=1)
        


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    #10 -- Acc = 66%
    #40 -- Acc = 80%
    #60 -- Acc = 82%
    #70 -- Acc = 83%
    #80 -- Acc = 84%
    #90 -- Acc = 84%


    def __str__(self):
        return "Netfull"
    def __init__(self):
        super(NetFull, self).__init__()
        self.input_layer = 784
        self.hid_nodes = 2000
        self.linear1 = nn.Linear(self.input_layer,self.hid_nodes) #10
        self.linear2 = nn.Linear(self.hid_nodes,10) #Try multiples of 10 for the hidden layer


    def forward(self, x):
        x = x.view(x.shape[0], -1)#flatten image into 784x1
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return torch.log_softmax(x,dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __str__(self):
        return "NetConv"
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5)
        self.conv2 = nn.Conv2d(30, 30, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 10)
        #self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 480) #flatten the image 
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        x = self.fc1(x)
        return F.log_softmax(x,dim=1)
