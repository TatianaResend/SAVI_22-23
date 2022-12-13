#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch 
from model2 import Model
from dataset2 import Dataset


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Create the dataset
    dataset = Dataset(3000,0.3,14)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle = True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch' + str(batch_idx) + 'has xs of size' + str(xs_ten.shape))

    # Draw training data
    plt.plot(dataset.xs_np,dataset.ys_np_labels,'go',label = 'labels')
    plt.legend(loc = 'best')
    plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu

    model = Model()
    model.to(device) # move the model variablle to gpu if one exists
    
    learning_rate = 0.01
    maximum_num_epochs = 50
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
    ys_ten_predicted = model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()
    
    plt.plot(dataset.xs_np,dataset.ys_np_labels,'g.',label = 'labels')
    plt.plot(dataset.xs_np,ys_np_predicted,'rx',label = 'predicted')
    plt.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":
    main()