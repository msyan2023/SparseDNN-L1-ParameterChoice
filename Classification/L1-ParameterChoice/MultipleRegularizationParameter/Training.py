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
    
def Training_print(Target_NNZ, NUM_dict, *args, **kwargs):
    model, NNZ_dict, meetStopSign = Training(*args, **kwargs)
    print(f"NNZ_dict: {NNZ_dict}")
    print(f"Target_NNZ: {Target_NNZ}")
    print(f"NUM_dict: {NUM_dict}")
    print(f"NNZ/NUM: {sum(NNZ_dict.values())/sum(NUM_dict.values())}")
    print("=============================================================================")
    return model, NNZ_dict, meetStopSign


    
def Training(train_dataset, test_dataset, lambd_dict, CheckStop, Exp, num_epoch=200):   
    
    torch.manual_seed(123)
    model = CNN8()  ## initialize the convolutional neural network model 
    
    
    print("num_epoch: ",num_epoch)
    
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
    
    

    history = {'TrainAccuracy': [], 'TestAccuracy': [], 'NNZ': [], 'lambd_dict': lambd_dict}
    
    Training_time_start = time.time()
    for epoch in range(num_epoch):
    
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
            optimizer.step()
            
            # Update parameters through soft thresholding
            NNZ_dict = {key: 0 for key in lambd_dict.keys()}  ## dictionary storing the number of nonzeros for each layer 
            for name, param in model.named_parameters():
                if "weight" in name:
                    layer_name = name.rsplit('.')[0]
                    lambd = lambd_dict[layer_name]
                    ## soft thresholding
                    SoftShrink = torch.nn.Softshrink(lambd=lambd*learning_rate)
                    param.data = SoftShrink(param.data)
                    NNZ_dict[layer_name] = torch.nonzero(param.data).size(0)
                    
            meetStopSign = CheckStop(NNZ_dict=NNZ_dict)
            if meetStopSign: 
                print(f"Meet stop sign at Exp={Exp}, epoch={epoch}, mini_batch_index={mini_batch_index}")
                break
            
            # Forward propagate with updated parameters to compute the loss
            outputs = model(inputs)
    
            loss_updated = criterion(outputs, labels)
            running_loss += loss_updated.item()
            
            if epoch % 10 == 0 and mini_batch_index % 200 == 0:  # print every 10 epoch and 100 mini-batches per epoch
                print('[epoch: %d, mini_batch: %d] loss: %.3f' % (epoch+1, mini_batch_index+1, running_loss)) 
                NNZ = sum(NNZ_dict.values())
                running_loss = 0.0    
                
        end_time = time.time()
        
        TrainAccuracy = GetAccuracy(model, trainloader)
        TestAccuracy = GetAccuracy(model, testloader)
        NNZ = sum(NNZ_dict.values())
        
        history['TrainAccuracy'].append(TrainAccuracy)
        history['TestAccuracy'].append(TestAccuracy)
        history['NNZ'].append(NNZ)
        
        if meetStopSign: 
            break
        
        if epoch % 10 == 0 or epoch == num_epoch-1 or meetStopSign:
            print("Elipsed time for epoch {}: {:.2f} seconds".format(epoch, end_time-start_time))
            print("------------------")
            print(f"lambd_dict: {lambd_dict}\n")
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
        
    result_NNZ = {"NNZ_dict": NNZ_dict,
                      "NNZ": NNZ}
        
    record = {"lambd_dict": lambd_dict, "model": model, 
              "result_Accuracy": result_Accuracy, "result_NNZ": result_NNZ, 
              "history": history}

    with open(f"Record_Exp{Exp}.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
            
        pickle.dump(record, dbfile)
                
                
    return model, NNZ_dict, meetStopSign

 


 


