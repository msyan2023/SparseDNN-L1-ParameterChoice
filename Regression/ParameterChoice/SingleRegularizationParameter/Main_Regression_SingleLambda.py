# -*- coding: utf-8 -*-

import numpy as np
import pickle


NN_width = [128, 128, 64]


TargetNNZ = 2000

num_epoch = 50000

ratio = 0.001

print(f"TargetNNZ: {TargetNNZ}")
print(f"ratio: {ratio}")
print(f"num_epoch: {num_epoch}")


''' stopping criteria function '''
def CheckStop_general(NNZ_dict, TargetNNZ, ratio):
    ## NNZ_dict: current NNZ dictionary 
    ## TargetNNZ: integer
    
    NNZ = sum(NNZ_dict.values())
    
    if abs(NNZ - TargetNNZ) <= TargetNNZ*ratio:
        meetStopSign = True
        print(f"-----Stop sign (ratio={ratio}) is reached !!!-----")
    else:
        meetStopSign = False
    #     print(f"-----Stop sign (ratio={ratio}) is NOT reached !!!-----")
        
    return meetStopSign

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        

from functools import partial
CheckStop = partial(CheckStop_general, TargetNNZ=TargetNNZ, ratio=ratio)
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
lambd = 1e-3

print("================================= Exp 0 =================================")
model, NNZ_dict, meetStopSign = Training_print(TargetNNZ, NUM_dict, NN_width, TrainData, TrainLabel, TestData, TestLabel, lambd, CheckStop, Exp=0, num_epoch=num_epoch)

lambd_IterHistory = []
NNZ_IterHistory = []

lambd_IterHistory.append(lambd)
NNZ_IterHistory.append(sum(NNZ_dict.values()))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

lambd = 1e-7

from GetGradient import GetGrad_afterTraining, GetSandwichIdx

for Exp in range(1,100):
    
    print(f"================================= Exp {Exp} =================================")

    model, NNZ_dict, meetStopSign = Training_print(TargetNNZ, NUM_dict, NN_width, TrainData, TrainLabel, TestData, TestLabel, lambd, CheckStop, Exp, num_epoch=num_epoch)
    
    if meetStopSign:
        break
 
    
    lambd_IterHistory.append(lambd)
    NNZ_IterHistory.append(sum(NNZ_dict.values()))
    
    ''' get the range of lambda according to the previous NNZ '''
    ## NNZ_IterHistory[id_low] < TargetNNZ < NNZ_IterHistory[id_up]
    ## lambd_IterHistory[id_up] < lambd < lambd_IterHistory[id_low]
    ids_low, ids_up = GetSandwichIdx(NNZ_IterHistory, TargetNNZ)
        
    lambd_low = np.max(np.array(lambd_IterHistory)[ids_up])
        
    lambd_up = np.min(np.array(lambd_IterHistory)[ids_low])  

    ''' update lambda '''
    lambd = GetGrad_afterTraining(model, TrainData, TrainLabel, lower_bound=lambd_low, upper_bound=lambd_up)
    
with open("IterationLambdaHistory.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
        
    pickle.dump((TargetNNZ, lambd_IterHistory, NNZ_IterHistory), dbfile) 
   