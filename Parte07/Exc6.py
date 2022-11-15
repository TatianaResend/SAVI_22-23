#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from random import randint, uniform
from colorama import Fore , Style
from scipy.optimize import least_squares
import math 
from Model import Polynomial

       

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Load file with points
    file = open("pts.pk1","rb")
    pts = pickle.load(file)
    file.close
    print('pts = ' + str(pts))

    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    print('Created a figure.')

    # Draw ground truth pts
    plt.plot(pts['xs'],pts['ys'],'sk', linewidth = 2, markersize = 6)

    # define the model
    model = Polynomial(pts)
    best_error = 1E6
    best_model = Sinusoid(pts)

    #line.draw()
    #plt.show()


    # ------------------------------------------
    # Execution
    # ------------------------------------------

    
    # set new values
    model.randomizeParams()

    result = least_squares(model.objectiveFunction,[model.a, model.b, model.h , model.k],verbose=2)


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    
    


if __name__ == "__main__":
    main()