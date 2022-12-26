#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch 
from model import Model
from dataset import Dataset
from statistics import mean
from colorama import Fore, Style
from statistics import mean
from tqdm import tqdm
import glob
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    
    # Define hyper parameters
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu

    model = Model()
    #model.to(device) # move the model variablle to gpu if one exists
    
    learning_rate = 0.01
    maximum_num_epochs = 50
    termination_loss_threshold = 0.01
    loss_fuction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # ------------------------------------------
    # Datasets
    # ------------------------------------------

    # Create the dataset
    dataset_path = './dogs-vs-cats-redux-kernels-edition/train'
    image_filenames = glob.glob(dataset_path + '/*.jpg')

    # sample only a few image to speed up development
    image_filenames = random.sample(image_filenames, k=900)

    # splip images into train and test
    train_image_filename, test_image_filename= train_test_split(image_filenames, test_size=0.2)

    #print(len(train_image_filename))

    dataset_train = Dataset(train_image_filename)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=256, shuffle = True)
    
    dataset_test = Dataset(test_image_filename)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle = True)

    #! Erro nesta parte
    # tensor_to_pil_image = transforms.ToPILImage()
    # for image_t, label_t in loader_train:
    #     print(image_t.shape)

    #     num_images = image_t.shape[0]
    #     image_idxs = random.sample(range(0,num_images),k=25)

    #     fig = plt.figure()
    #     for subplot_idx, image_idx in enumerate(image_idxs,start=1):

    #         image_pil = tensor_to_pil_image(image_t[image_idxs, :, :, :])
            
    #         ax = fig.add_suplot(5,5,subplot_idx)
    #         ax.xaxis.set_ticklabels([])
    #         ax.yaxis.set_ticklabels([])
    #         ax.xaxis.set_ticks([])
    #         ax.yaxis.set_ticks([])

    #         label = label_t[image_idx].data.item()
    #         class_name = 'dog' if label == 0 else 'cat'
    #         ax.set_xlabel(class_name)
    #         plt.imshow(image_pil)


    # # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu

    # model = Model()
    model.to(device) # move the model variablle to gpu if one exists
    
    # learning_rate = 0.01
    # maximum_num_epochs = 50
    # termination_loss_threshold = 10
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)



    # ----------------------------------------------
    # Training
    # ----------------------------------------------
    idx_epoch = 0
    epoch_train_losses = []
    while True:

        # Train batch by batch
        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # Apply the network to get the predicted ys
            lanel_t_predicted = model.forward(image_t)

            # Compute the error based on the predictions
            loss = loss_fuction(lanel_t_predicted, label_t)

            # Update the model, i.e. the neural network's weights 
            optimizer.zero_grad() # resets the weights to make sure we are not accumulating
            loss.backward() # propagates the loss error into each neuron
            optimizer.step() # update the weights

            # Report
            # print('Epoch ' + str(idx_epoch) + ' batch ' + str(batch_idx) + ', Loss ' + str(loss.item()))

            train_losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        # ---------------------------------------------------------------
        epoch_test_losses = []
        while True:

            # Train batch by batch
            test_losses = []
            for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

                image_t = image_t.to(device)
                label_t = label_t.to(device)

                # Apply the network to get the predicted ys
                lanel_t_predicted = model.forward(image_t)

                # Compute the error based on the predictions
                loss = loss_fuction(lanel_t_predicted, label_t)

                # Report
                # print('Epoch ' + str(idx_epoch) + ' batch ' + str(batch_idx) + ', Loss ' + str(loss.item()))

                test_losses.append(loss.data.item())

            # Compute the loss for the epoch
            epoch_test_loss = mean(train_losses)
            epoch_test_losses.append(epoch_test_loss)
            # ---------------------------------------------------------------------

        print(Fore.BLUE + 'Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
    
        idx_epoch += 1 # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break
        elif epoch_train_loss < termination_loss_threshold:
            print('Finished training. Reached target loss.')
            break

    # ----------------------------------------
    # finalization
    # ----------------------------------------
    
    # Run the model once to get ys_predicted
    lanel_t_predicted = model.forward(dataset_train.xs_ten.to(device))
    ys_np_predicted = lanel_t_predicted.cpu().detach().numpy()
    
    plt.title('Train dataset data')
    plt.plot(dataset_train.xs_np,dataset_train.ys_np_labels,'g.',label = 'labels')
    plt.plot(dataset_train.xs_np,ys_np_predicted,'rx',label = 'predicted')
    plt.legend(loc = 'best')

    # Plot the loss epoch graph
    plt.figure()
    plt.title('Training report')
    plt.plot(range(0, len(epoch_train_losses)), epoch_train_losses,'-b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.draw()

    # Plot the dataset test data and model predictions
    plt.figure()
    lanel_t_predicted = model.forward(dataset_test.xs_ten.to(device))
    ys_np_predicted = lanel_t_predicted.cpu().detach().numpy()

    plt.title('Test dataset data')
    plt.plot(dataset_test.xs_np, dataset_test.ys_np_labels,'g.', label = 'labels')
    plt.plot(dataset_test.xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')

    plt.show()

if __name__ == "__main__":
    main()