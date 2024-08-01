# -*- coding: utf-8 -*-

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle 
import numpy as np 

TargetNNZ = [250, 8000, 13000, 20000, 4200]

num_epoch = 200

ratio = 0.1

TolStopSign = 5


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
Target_NNZ = {'conv1': TargetNNZ[0], 'conv2': TargetNNZ[1], 'conv3': TargetNNZ[2], 'fc1': TargetNNZ[3], 'fc2': TargetNNZ[4]}

from functools import partial
CheckStop = partial(CheckStop_general, TargetNNZ_dict=Target_NNZ, ratio=ratio, TolStopSign=TolStopSign)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''' load the dataset FashionMNIST, MNIST, cifar10(should apply different transform)'''
# Define the data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_dataset = datasets.MNIST(root='~/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='~/data', train=False, download=True, transform=transform)

from Class_CNN8 import BasicInfo_CNN8
NUM_dict = BasicInfo_CNN8()
NUM = sum(NUM_dict.values())

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
from Training import Training_print
eps = 1e-4
lambd_dict = {'conv1': eps, 'conv2': eps, 'conv3': eps, 'fc1': eps, 'fc2': eps}  ## initial lambda values 

model, NNZ_dict, meetStopSign = Training_print(Target_NNZ, NUM_dict, train_dataset, test_dataset, lambd_dict, CheckStop, Exp=0, num_epoch=num_epoch)

lambd_IterHistory = {key: [] for key in lambd_dict.keys()}  
NNZ_IterHistory = {key: [] for key in lambd_dict.keys()}
for key in lambd_dict.keys():
    lambd_IterHistory[key].append(lambd_dict[key])
    NNZ_IterHistory[key].append(NNZ_dict[key])

## lambd_dict: dictionary of regularization parameters for each layer
eps = 1e-7
lambd_dict = {'conv1': eps, 'conv2': eps, 'conv3': eps, 'fc1': eps, 'fc2': eps}  ## initial lambda values 

lambd_low, lambd_up = {}, {} 

from GetGradient import GetGrad_afterTraining, GetSandwichIdx

for Exp in range(1,30):
    
    print(f"================================= Exp {Exp} =================================")

    model, NNZ_dict, meetStopSign = Training_print(Target_NNZ, NUM_dict, train_dataset, test_dataset, lambd_dict, CheckStop, Exp, num_epoch=num_epoch)

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
    lambd_dict, _ = GetGrad_afterTraining(model, train_dataset, lower_bound=lambd_low, upper_bound=lambd_up)
    
with open("IterationLambdaHistory.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
        
    pickle.dump((Target_NNZ, lambd_IterHistory, NNZ_IterHistory), dbfile) 
   

