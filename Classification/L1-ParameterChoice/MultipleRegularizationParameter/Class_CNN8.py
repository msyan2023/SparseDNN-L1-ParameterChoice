# -*- coding: utf-8 -*-


import torch.nn as nn
from torchsummary import summary
    
class CNN8(nn.Module): 
    ## inherit from the class nn.Module
    
    ## eight layers: conv --> max_pooling --> conv --> max_pooling --> conv --> max_pooling --> fully connected --> fully connected --> output
    
    def __init__(self, num_classes=10):
        
        super().__init__()  ## apply the method "__init__" from the inherited class "nn.Module"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  ## by default, stride=1, padding=0
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
        
        self.fc1 = nn.Linear(128*3*3, 512)  ## fully connected layer
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # print(x.shape)
        x = self.pool(self.ReLU(self.conv1(x))) # convolution layer 1 --> relu activation --> max pooling
        # print(x.shape)
        x = self.pool(self.ReLU(self.conv2(x)))
        # print(x.shape)
        x = self.pool(self.ReLU(self.conv3(x)))
        
        '''!!! take case of this: Since the output of the third convolutional layer has dimensions [batch_size, 128, 4, 4]'''
        x = x.view(-1, 128*3*3)   
        x = self.ReLU(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
    
def BasicInfo_CNN8():
    
    ## instantiate a CNN8
    model = CNN8()
    
    ## number of weight parameters in convolutional and fully connected layers
    NUM_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            layer_name = name.rsplit('.')[0]
            NUM_dict[layer_name] = param.numel()
            
    # ## number of all parameters including weights and bias
    # NUM = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    ''' print the summary '''
    summary(model, (1, 28, 28))  # (1, 28, 28) is the input shape (channel, height, width)
    
    return NUM_dict



