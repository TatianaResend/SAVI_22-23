#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init_()

        self.layer1 = torch.nn.Linear(1,1)

    def forward(self,xs):
        ys = self.layer1(xs)

        return  ys


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    # Read file with points
    file = open('pts.pkl','rb')
    pts = pickle.load(file)
    file.close()
    print('pts = ' + str(pts))
    
    # Convert the pts into np arrays
    xs_np = np.array(pts['xs'],dtype=np.float32).reshape(-1,1)
    ys_np_labels = np.array(pts['ys'],dtype=np.float32).reshape(-1,1)

    # Draw training data
    plt.plot(xs_np,ys_np_labels,'go',label = 'labels')
    plt.legend(loc = 'best')
    plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu

    model = Model()
    model.to(device) # move the model variablle to gpu if one exists
    
    learning_rate = 0.01
    maximum_num_epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    # ----------------------------------------------
    # Training
    # ----------------------------------------------
    idx_epoch = 0
    while True:

        xs_ten = torch.from_numpy(xs_np).to(device)
        ys_ten_labels = torch.from_numpy(ys_np_labels)

        # Aply the network to get predicted ys
        ys_ten_predicted = model.forward(xs_ten)

        # compote the error based on the predictions
        loss = criterion(ys_ten_predicted,ys_ten_labels)
        
        # Update the model, i.e. the neural network's weights
        optimizer.zero_grad() # resets the weight to make sure we are not accumulating
        loss.backward() 
        optimizer.step()

        # Report
        print('Epoch' + str(idx_epoch) + ', Loss' + str(loss.item()))

        idx_epoch += 1  # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break

    # ----------------------------------------
    # finalization
    # ----------------------------------------
    
    # Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(xs_ten)
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy
    
    plt.plot(xs_np,ys_np_labels,'rx',label = 'labels')
    plt.plot(xs_np,ys_np_predicted,'rx',label = 'predicted')
    plt.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":
    main()