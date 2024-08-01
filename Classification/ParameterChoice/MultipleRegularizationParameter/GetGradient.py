# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import numpy as np 


def GetGrad_afterTraining(model, train_dataset, lower_bound, upper_bound): 

    ## set up loss function 
    criterion = nn.CrossEntropyLoss()
    
    trainloader_full = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    inputs, labels = next(iter(trainloader_full))
    ## forward propagation
    outputs = model(inputs)
    ## get loss
    loss = criterion(outputs, labels)
    ## back propagation (get gradient)
    loss.backward()
    # access and concatenate gradients 
    param_grad_stat = {}  ## statistics of gradients of parameters in each layer
    num_stat = {}  ## number of entries between bounds in each layer
    
 
    for name, param in model.named_parameters():
        
        if "weight" in name:
            
            layer_name = name.rsplit('.')[0]
            
            param_grad_abs = torch.abs(param.grad)
            
            mask = (param_grad_abs > lower_bound[layer_name]) & (param_grad_abs < upper_bound[layer_name])
            
            param_grad_stat[layer_name] = param_grad_abs[mask].median().item()
 
            num_stat[layer_name] = torch.count_nonzero(mask).item()
    
            if np.isnan(param_grad_stat[layer_name]):
            
                param_grad_stat[layer_name] = (lower_bound[layer_name] + upper_bound[layer_name]) / 2.0    
     
    
    return param_grad_stat, num_stat





def GetSandwichIdx(numbers, number_to_find):
    
    # numbers = np.array([34, 1, 3, 44, 49])
    # number_to_find = 20
    
    numbers = np.array(numbers)
    
    ## find the maximum among entries less than number_to_find
    id_low = np.argmax(np.where([numbers>number_to_find], -np.inf, numbers))
    
    ## find the minimum among entries greater than given number
    id_up = np.argmin(np.where([numbers<number_to_find], np.inf, numbers))

    ## in case of some equal terms like 
    # numbers = np.array([1, 1, 20, 20])
    # number_to_find = 10
    ids_low = np.where(numbers == numbers[id_low])[0]
    ## remark: np.where returns (array,) so that why we need np.where()[0] to return the array 
    ids_up = np.where(numbers == numbers[id_up])[0]    
       

    ''' numbers[index_low] < number_to_find < numbers[index_up] '''
    return ids_low, ids_up 


