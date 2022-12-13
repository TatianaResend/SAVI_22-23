#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch 
from model2 import Model
from dataset2 import Dataset
from statistics import mean

#! não acabei

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Create the dataset
    dataset_train = Dataset(3000,0.3,14,sigma=3)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=256, shuffle = True)

    dataset_test = Dataset(500,0.3,14,sigma=3)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle = True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch' + str(batch_idx) + 'has xs of size' + str(xs_ten.shape))

    # Draw training data
    # plt.plot(dataset.xs_np,dataset.ys_np_labels,'go',label = 'labels')
    # plt.legend(loc = 'best')
    # plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu

    model = Model()
    model.to(device) # move the model variablle to gpu if one exists
    
    learning_rate = 0.01
    maximum_num_epochs = 50
    termination_loss_threshold = 5
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # ----------------------------------------------
    # Training
    # ----------------------------------------------
    idx_epoch = 0
    while True:

        for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):

            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)

            # Aply the network to get predicted ys
            ys_ten_predicted = model.forward(xs_ten)

            # compote the error based on the predictions
            loss = criterion(ys_ten_predicted,ys_ten_labels)
            
            # Update the model, i.e. the neural network's weights
            optimizer.zero_grad() # resets the weight to make sure we are not accumulating
            loss.backward() 
            optimizer.step()

            # Report
            print('Epoch' + str(idx_epoch) + ', Loss ' + str(loss.item()))

            #losses.append(loss.data.item)


        idx_epoch += 1  # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break

    # ----------------------------------------
    # finalization
    # ----------------------------------------
    
    # Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(dataset_train.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()
    
    plt.plot(dataset_train.xs_np,dataset_train.ys_np_labels,'g.',label = 'labels')
    plt.plot(dataset_train.xs_np,ys_np_predicted,'rx',label = 'predicted')
    plt.legend(loc = 'best')

    # Plot the loss epoch graph
    plt.figure()
    plt.title('Training report')
    #plt.plot(range(0,len(epoch_losses)))
     # Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(dataset_train.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()
    
    plt.plot(dataset_train.xs_np,dataset_train.ys_np_labels,'g.',label = 'labels')
    plt.plot(dataset_train.xs_np,ys_np_predicted,'rx',label = 'predicted')
    plt.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":
    main()