# -*- coding: utf-8 -*-

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle 
import numpy as np 
import sys

TargetNNZ = 50000

num_epoch = 200

ratio = 0.001



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

    return meetStopSign


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        

from functools import partial
CheckStop = partial(CheckStop_general, TargetNNZ=TargetNNZ, ratio=ratio)
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
lambd = 1e-2

print("================================= Exp 0 =================================")
model, NNZ_dict, meetStopSign = Training_print(TargetNNZ, NUM_dict, train_dataset, test_dataset, lambd, CheckStop, Exp=0, num_epoch=num_epoch)

if meetStopSign:
    # If the condition is true, print a message and exit with a completion code "0"
    print(f"In Exp=0, the stopping criteria (TargetNNZ={TargetNNZ},ratio={ratio}) is met.")
    sys.exit(0)  # Exit with a completion code (0)


lambd_IterHistory = [] 
NNZ_IterHistory = []

lambd_IterHistory.append(lambd)
NNZ_IterHistory.append(sum(NNZ_dict.values()))
    
## lambd: regularization parameters for all layers
lambd = 1e-7

from GetGradient import GetGrad_afterTraining, GetSandwichIdx

for Exp in range(1,30):
    
    print(f"================================= Exp {Exp} =================================")

    model, NNZ_dict, meetStopSign = Training_print(NUM_dict, train_dataset, test_dataset, lambd, CheckStop, Exp, num_epoch=num_epoch)
    
    
    if meetStopSign:
        break
    
    lambd_IterHistory.append(lambd)
    NNZ_IterHistory.append(sum(NNZ_dict.values()))
    
    ''' get the range of lambda according to the previous NNZ '''
    ## NNZ_IterHistory[id_low] < Target_NNZ < NNZ_IterHistory[id_up]
    ## lambd_IterHistory[id_up] < lambd < lambd_IterHistory[id_low]
    ids_low, ids_up = GetSandwichIdx(NNZ_IterHistory, TargetNNZ)
        
    lambd_low = np.max(np.array(lambd_IterHistory)[ids_up])
        
    lambd_up = np.min(np.array(lambd_IterHistory)[ids_low])  

    ''' update lambda '''
    lambd = GetGrad_afterTraining(model, train_dataset, lower_bound=lambd_low, upper_bound=lambd_up)
    
with open("IterationLambdaHistory.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
        
    pickle.dump((TargetNNZ, lambd_IterHistory, NNZ_IterHistory), dbfile) 

    
print(f"lambd_IterHistory: {lambd_IterHistory}")

print(f"NNZ_IterHistory: {NNZ_IterHistory}")