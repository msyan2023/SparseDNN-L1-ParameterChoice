# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle
from Class_CNN8 import CNN8

def GetAccuracy(model, DataLoader):
    
    correct = 0

    total = len(DataLoader.dataset)

    with torch.no_grad():
    
        for data in DataLoader:
            
            images, labels = data
            
            outputs = model(images)
            ## outputs is of shape [Batch, NumClass]
            ## outputs[j,k] represents the probability of j-th sample in k-th class
            
            _, predicted = torch.max(torch.softmax(outputs.data, dim=1), axis=1)
            
            correct += (predicted==labels).sum().item()
            ## item() is a method in PyTorch that is used to extract a scalar value from a tensor of size 1. When applied to a tensor with a single element, 
            ## item() returns the value of that element as a Python float or integer.
    return correct / total      
 


    
def Training(train_dataset, test_dataset, lambd, Num_epoch=200):   
    
    torch.manual_seed(123)
    model = CNN8()  ## initialize the convolutional neural network model 
    
    
    print("Num_epoch: ",Num_epoch)
    
    # Set the random seed for reproducibility (fix the random intial parameters)
    torch.manual_seed(0)
    
    # set up mini-batch
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    
    ## set up loss function 
    criterion = nn.CrossEntropyLoss()
    
    ## set up optimier
    learning_rate = 0.1  ## 0.1 for sgd, 0.0001 for Adam
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    
    NumTrain = len(trainloader.dataset)
    NumTest = len(testloader.dataset)
    
    

    history = {'TrainAccuracy': [], 'TestAccuracy': [], 'lambd': lambd}
    
    Training_time_start = time.time()
    for epoch in range(Num_epoch):
    
        start_time = time.time()
        running_loss = 0.0
    
        # Set the random seed for reshuffling within each epoch for reproducibility
        torch.manual_seed(epoch)
        
        ## The training data is shuffled every time trainloader is called
        for mini_batch_index, mini_batch_data in enumerate(trainloader, 0):
    
            inputs, labels = mini_batch_data 
            
            ## every time before computing gradient during training, we need to refresh the gradient to be 0
            optimizer.zero_grad()
      
            ## forward propagation
            outputs = model(inputs)
            ## get loss
            loss = criterion(outputs, labels)
            ## back propagation (get gradient)
            loss.backward()
            
            ## update parameters
            # optimizer.step()
            
            # Update parameters with l2 regularization 
            for name, param in model.named_parameters():
                if "weight" in name:
                    param.data -= learning_rate * (param.grad + lambd * param.data)
                else:
                    param.data -= learning_rate * param.grad 
            
            # Forward propagate with updated parameters to compute the loss
            outputs = model(inputs)
    
            loss_updated = criterion(outputs, labels)
            running_loss += loss_updated.item()
            
            if epoch % 10 == 0 and mini_batch_index % 200 == 0:  # print every 10 epoch and 100 mini-batches per epoch
                print('[epoch: %d, mini_batch: %d] loss: %.3f' % (epoch+1, mini_batch_index+1, running_loss)) 
                running_loss = 0.0    
                
        end_time = time.time()

        TrainAccuracy = GetAccuracy(model, trainloader)
        TestAccuracy = GetAccuracy(model, testloader)
        
        history['TrainAccuracy'].append(TrainAccuracy)
        history['TestAccuracy'].append(TestAccuracy)
        
        if epoch % 10 == 0 or epoch == Num_epoch:
            print("Elipsed time for epoch {}: {:.2f} seconds".format(epoch, end_time-start_time))
            print("------------------")
            print('TrainAccuracy over %d training samples: %0.4f %%' % (
                    NumTrain, TrainAccuracy*100))
            print('TestAccuracy over %d testing samples: %0.4f %%\n' % (
                    NumTest, TestAccuracy*100))
            print("---------------------------------------------------------------------------------")
                
    Training_time_end = time.time()
    Training_time = Training_time_end - Training_time_start
    print(f"Total Training Time = {Training_time}")    
    
    
    result_Accuracy = {"TrainAccuracy": TrainAccuracy, 
                  "TestAccuracy": TestAccuracy}
    
        
    record = {"lambd": lambd, "model": model, 
              "result_Accuracy": result_Accuracy,
              "history": history}

    with open(f"Record_L2_lambd{lambd}.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
            
        pickle.dump(record, dbfile)
                
                


 


 


