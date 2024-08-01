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

lambd = 0.0001  ## regularization parameter
learning_rate = 0.1
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
num_epochs = 50000
for epoch in range(num_epochs):
    # Forward pass
    TrainOpts = model(TrainData)  ## training outputs
    TrainMSE = criterion(TrainOpts, TrainLabel)
    
    TestOpts = model(TestData)  ## testing outputs
    TestMSE = criterion(TestOpts, TestLabel)
    
    # Backward and optimize
    optimizer.zero_grad()
    TrainMSE.backward()
    
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data -= learning_rate * (param.grad + lambd * param.data)
        else:
            param.data -= learning_rate * param.grad   ## we do not apply L2 regularization on bias
    
    
    # optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], TrainMSE: {TrainMSE.item():.6f}, TestMSE: {TestMSE.item():.6f}')
