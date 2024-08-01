# -*- coding: utf-8 -*-

## load the mat file 
from scipy.io import loadmat
data = loadmat('MgDataLabel.mat')

## extract data 
import torch 
Data = torch.tensor(data['MgData'].T, dtype=torch.float32)
Label = torch.tensor(data['MgLabel'], dtype=torch.float32)

NumTrain = 1000
TrainData = Data[:NumTrain,:]
TrainLabel = Label[:NumTrain,:]
TestData = Data[NumTrain+1:,:]
TestLabel = Label[NumTrain+1:,:]

from RegressionNN import RegressionNN
# Initialize the model 
torch.manual_seed(123)
model = RegressionNN()

import torch.nn as nn



learning_rate = 0.1
num_epochs = 50000

for lambd in [1e-5, 2e-5, 3e-5, 5e-5, 1e-4]:  ## regularization parameter

    SoftShrink = torch.nn.Softshrink(lambd=lambd*learning_rate)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        TrainOpts = model(TrainData)  ## training outputs
        TrainMSE = criterion(TrainOpts, TrainLabel)
        
        TestOpts = model(TestData)  ## testing outputs
        TestMSE = criterion(TestOpts, TestLabel)
        
        # Backward and optimize
        optimizer.zero_grad()
        TrainMSE.backward()
        
        ## update parameters
        optimizer.step()
        
        for name, param in model.named_parameters():
            if "weight" in name:            
                param.data = SoftShrink(param.data)
             
        
        # optimizer.step()
    
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], TrainMSE: {TrainMSE.item():.6f}, TestMSE: {TestMSE.item():.6f}')

    
    print('---------------------------Analysis--------------------------------------------------------------------------------------------')
    NNZs = []
    weights = [768,16384,8192,64]
    for name, param in model.named_parameters():
        if "weight" in name:
            layer_name = name.rsplit('.')[0]
            NNZs.append(torch.nonzero(param.data).size(0))
            
    print(sum(NNZs) / sum(weights))






