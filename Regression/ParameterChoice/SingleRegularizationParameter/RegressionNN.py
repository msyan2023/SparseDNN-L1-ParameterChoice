# -*- coding: utf-8 -*-

import torch.nn as nn

# Define the neural network architecture
class RegressionNN(nn.Module):
    def __init__(self, NN_width):
        super(RegressionNN, self).__init__()
        
        input_size=6
        [size1, size2, size3] = NN_width
        
        self.fc1 = nn.Linear(input_size, size1)
        self.fc2 = nn.Linear(size1, size2)
        self.fc3 = nn.Linear(size2, size3)
        self.fc4 = nn.Linear(size3, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
    
def BasicInfo_RegressionNN(NN_width):
    
    ## instantiate a RegressionNN
    model = RegressionNN(NN_width)
    
    ## number of weight parameters in fully connected layers
    NUM_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            layer_name = name.rsplit('.')[0]
            NUM_dict[layer_name] = param.numel()
            
    # ## number of all parameters including weights and bias
    # NUM = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    # ''' print the summary '''
    # summary(model, (1,6))  # (1, 28, 28) is the input shape (channel, height, width)
    print(NUM_dict)
    return NUM_dict


# BasicInfo_RegressionNN()