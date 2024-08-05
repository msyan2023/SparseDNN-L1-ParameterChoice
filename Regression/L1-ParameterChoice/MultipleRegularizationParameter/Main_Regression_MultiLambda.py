# -*- coding: utf-8 -*-

import numpy as np
import pickle

 
NN_width = [128,128,64]   ## Neural network width parameter

TargetNNZ = [650, 1200, 1000, 45]

num_epoch = 50000

ratio = 0.05
TolStopSign = 3


print(TargetNNZ)
print(num_epoch)
print(ratio)
print(TolStopSign)

''' stopping criteria function '''
def CheckStop_general(NNZ_dict, TargetNNZ_dict, ratio, TolStopSign):
    ## NNZ_dict: current NNZ dictionary 
    ## TargetNNZ_dict: target NNZ dictionary 
    NumSign = 0
    for key in NNZ_dict.keys():
        if abs(NNZ_dict[key]-TargetNNZ_dict[key]) <= TargetNNZ_dict[key]*ratio:
            NumSign += 1
    meetStopSign = (NumSign >= TolStopSign)
    if meetStopSign == True:
        print(f"-----Stop sign (ratio={ratio}, TolStopSign={TolStopSign}) is reached !!!-----")
        print(f"-----{NumSign} layers (totally {len(TargetNNZ_dict)} layers) are satisfied.-----")
        
    return meetStopSign

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Target_NNZ = {'fc1': TargetNNZ[0], 'fc2': TargetNNZ[1], 'fc3': TargetNNZ[2], 'fc4': TargetNNZ[3]}

from functools import partial
CheckStop = partial(CheckStop_general, TargetNNZ_dict=Target_NNZ, ratio=ratio, TolStopSign=TolStopSign)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''   
 

''' load the dataset '''
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

from RegressionNN import BasicInfo_RegressionNN
NUM_dict = BasicInfo_RegressionNN(NN_width)
NUM = sum(NUM_dict.values())



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from Training import Training_print
eps = 1e-3
lambd_dict = {'fc1': eps, 'fc2': eps, 'fc3': eps, 'fc4': eps}  ## initial lambda values 

model, NNZ_dict, meetStopSign = Training_print(Target_NNZ, NUM_dict, NN_width, TrainData, TrainLabel, TestData, TestLabel, lambd_dict, CheckStop, Exp=0, num_epoch=num_epoch)

lambd_IterHistory = {key: [] for key in lambd_dict.keys()}  
NNZ_IterHistory = {key: [] for key in lambd_dict.keys()}
for key in lambd_dict.keys():
    lambd_IterHistory[key].append(lambd_dict[key])
    NNZ_IterHistory[key].append(NNZ_dict[key])

eps = 1e-7
lambd_dict = {'fc1': eps, 'fc2': eps, 'fc3': eps, 'fc4': eps}  ## initial lambda values 

lambd_low, lambd_up = {}, {} 

from GetGradient import GetGrad_afterTraining, GetSandwichIdx

for Exp in range(1,100):
    
    print(f"================================= Exp {Exp} =================================")

    model, NNZ_dict, meetStopSign = Training_print(Target_NNZ, NUM_dict, NN_width, TrainData, TrainLabel, TestData, TestLabel, lambd_dict, CheckStop, Exp, num_epoch=num_epoch)
    
    if meetStopSign:
        break
 
    
    for key in lambd_dict.keys():
        lambd_IterHistory[key].append(lambd_dict[key])
        NNZ_IterHistory[key].append(NNZ_dict[key])

        ''' get the range of lambda according to the previous NNZ '''
        ## NNZ_IterHistory[key][id_low] < Target_NNZ[key] < NNZ_IterHistory[key][id_up]
        ## lambd_IterHistory[key][id_up] < lambd < lambd_IterHistory[key][id_low]
        ids_low, ids_up = GetSandwichIdx(NNZ_IterHistory[key], Target_NNZ[key])
        
        lambd_low[key] = np.max(np.array(lambd_IterHistory[key])[ids_up])
        
        lambd_up[key] = np.min(np.array(lambd_IterHistory[key])[ids_low])  

    ''' update lambda '''
    lambd_dict, _ = GetGrad_afterTraining(model, TrainData, TrainLabel, lower_bound=lambd_low, upper_bound=lambd_up)
    
with open("IterationLambdaHistory.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
        
    pickle.dump((Target_NNZ, lambd_IterHistory, NNZ_IterHistory), dbfile) 
   