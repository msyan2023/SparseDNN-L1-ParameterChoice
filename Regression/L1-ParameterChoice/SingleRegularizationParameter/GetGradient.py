# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

import numpy as np 


def GetGrad_afterTraining(model, TrainData, TrainLabel, lower_bound, upper_bound): 

    ## set up loss function 
    criterion = nn.MSELoss()
    
    ## forward propagation
    outputs = model(TrainData)
    ## get loss
    loss = criterion(outputs, TrainLabel)
    ## back propagation (get gradient)
    loss.backward()
    
    # Initialize lists to store gradients
    all_gradients = []
 
    for name, param in model.named_parameters():
        
        if "weight" in name:
                        
            param_grad_abs = torch.abs(param.grad)
            
            mask = (param_grad_abs > lower_bound) & (param_grad_abs < upper_bound)
            
            # Append flattened masked gradients to the list
            all_gradients.append(param_grad_abs[mask].flatten())

                        
    # Concatenate all gradients into a single tensor
    all_gradients_concat = torch.cat(all_gradients)            
    
    # Print the shape of the concatenated gradients
    print("Shape of all_gradients_concat:", all_gradients_concat.shape)
    
    # Calculate the median of the concatenated gradients
    median_grad = all_gradients_concat.median().item()
    
    if np.isnan(median_grad):
        
        print("**********We compromise to the average strategy for selecting lambda**********")
    
        median_grad = (lower_bound + upper_bound) / 2.0
        
    else: 
    
        print("**********We use gradient strategy for selecting lambda**********")
        
    return median_grad





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


