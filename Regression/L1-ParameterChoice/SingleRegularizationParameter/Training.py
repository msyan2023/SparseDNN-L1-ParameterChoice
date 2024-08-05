# -*- coding: utf-8 -*-

from RegressionNN import RegressionNN
import torch 
import torch.nn as nn
import pickle 
 
def Training_print(TargetNNZ, NUM_dict, *args, **kwargs):
    model, NNZ_dict, meetStopSign = Training(*args, **kwargs)
    print(f"NNZ_dict: {NNZ_dict}")
    print(f"TargetNNZ: {TargetNNZ}")
    print(f"NUM_dict: {NUM_dict}")
    print(f"NNZ/NUM: {sum(NNZ_dict.values())/sum(NUM_dict.values())}")
    print("=============================================================================")
    return model, NNZ_dict, meetStopSign


def Training(NN_width, TrainData, TrainLabel, TestData, TestLabel, lambd, CheckStop, Exp, num_epoch=50000):
    
    '''
    Arguments:  
        lambd : a regularization parameter of l1 norm for all layers
        
    '''
    
    # Initialize the model 
    torch.manual_seed(123)
    model = RegressionNN(NN_width)
    
    criterion = nn.MSELoss()
    
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    history = {'TrainMSE': [], 'TestMSE': [], 'NNZ': [], 'lambd': lambd}
    for epoch in range(num_epoch):
        # Forward pass
        TrainOpts = model(TrainData)  ## training outputs
        TrainMSE = criterion(TrainOpts, TrainLabel)
        history['TrainMSE'].append(TrainMSE)
        
        TestOpts = model(TestData)  ## testing outputs
        TestMSE = criterion(TestOpts, TestLabel)
        history['TestMSE'].append(TestMSE)
        
        # Backward and optimize
        optimizer.zero_grad()
        TrainMSE.backward()
        
        ## update parameters
        optimizer.step()
        
        # Update parameters through soft thresholding
        NNZ_dict = {'fc1': 0, 'fc2': 0, 'fc3': 0, 'fc4': 0} ## dictionary storing nonzero weights for each layer 
        
        # Update parameters through soft thresholding
        for name, param in model.named_parameters():
            if "weight" in name:
                layer_name = name.rsplit('.')[0]
                ## soft thresholding
                SoftShrink = torch.nn.Softshrink(lambd=lambd*learning_rate)
                param.data = SoftShrink(param.data)
                NNZ_dict[layer_name] = torch.nonzero(param.data).size(0)
                
        NNZ = sum(NNZ_dict.values())
        history['NNZ'].append(NNZ)
         
        meetStopSign = CheckStop(NNZ_dict=NNZ_dict)
        if meetStopSign: 
            print(f"------Meet stop sign at Exp={Exp}, epoch={epoch+1}------")
            print(f"TrainMSE: {TrainMSE.item():.6f}, TestMSE: {TestMSE.item():.6f}")
            break
        
        if epoch % 1000 == 0 or epoch == num_epoch-1 :
            print(f'Epoch [{epoch+1}/{num_epoch}], TrainMSE: {TrainMSE.item():.6f}, TestMSE: {TestMSE.item():.6f}')
    
    
    ### record final result 
    result_MSE = {"TrainMSE": TrainMSE.item(), 
                  "TestMSE": TestMSE.item()}
        
    result_NNZ = {"NNZ_dict": NNZ_dict,
                      "NNZ": NNZ}
    
    print(f"//////////////////////Result of Exp {Exp}/////////////////////////////////")
    print(f"lambd = {lambd}")
    print(f"result_MSE = {result_MSE}")
    print(f"result_NNZ = {result_NNZ}")
    print("//////////////////////////////////////////////////////////////////////////")

    
    record = {"lambd": lambd, "model": model, 
              "result_MSE": result_MSE, "result_NNZ": result_NNZ, 
              "history": history}

    with open(f"Record_Exp{Exp}.pkl", 'wb') as dbfile:  ##  The file is closed using the with statement, which automatically closes the file when the block is exited.
            
        pickle.dump(record, dbfile)
    
    
    return model, NNZ_dict, meetStopSign




