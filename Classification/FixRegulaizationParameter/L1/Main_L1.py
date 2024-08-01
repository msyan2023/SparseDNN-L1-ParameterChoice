# -*- coding: utf-8 -*-

import torchvision.datasets as datasets
import torchvision.transforms as transforms

lambd = 0.000001



''' load the dataset FashionMNIST, MNIST, cifar10(should apply different transform)'''
# Define the data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)


from Class_CNN8 import BasicInfo_CNN8
NUM_dict = BasicInfo_CNN8()
NUM = sum(NUM_dict.values())
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
from Training import Training
Training(train_dataset, test_dataset, lambd)


