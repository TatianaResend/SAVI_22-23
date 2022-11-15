#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from random import randint, uniform
from colorama import Fore , Style
import math 

class Sinusoid():
    """ define the model of a line segment
    """

    def __init__(self, gt):

        self.gt = gt
        self.randomizeParams() 
        self.first_draw = True

    def randomizeParams(self):
        self.a = uniform(-10,10)
        self.b = uniform(-10,10)
        self.h = uniform(-10,10)
        self.k = uniform(-10,10)
        
    def getY(self,x):
        return self.a * math.sin(self.b) + self.b

    def objectiveFunction(self):
        residuals = []

        # percorrer os pontos todos do grounth true
        for gt_x, gt_y in zip(self.gt['xs'],self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # error is the sum of the residuals
        error = sum(residuals)
        return error

    def draw(self, color = 'b--'):
        xi = -10
        xf = 10
        yi = self.getY(xi)
        yf = self.getY(xf)

        if self.first_draw:
            self.draw_handle = plt.plot([xi,xf],[yi,yf],color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handle, data=([xi,xf],[yi,yf]))  #update ln
        

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
    model = Sinusoid(pts)
    best_error = 1E6
    best_model = Sinusoid(pts)

    #line.draw()
    #plt.show()


    # ------------------------------------------
    # Execution
    # ------------------------------------------

    while True: # iterative setting new values for the params and recomputing the error
        
        # set new values
        model.randomizeParams()

        # compute error
        error = model.objectiveFunction()
        print(error)

        if error < best_error:  # we found a better model
            best_model.m = model.a    #copy the best found line params
            best_model.b = model.b
            best_model.b = model.h
            best_model.b = model.k
            best_error = error      # update best error
            print(Fore.RED + 'We found a better model!!!' + Style.RESET_ALL )

        # draw current model
        model.draw()
        best_model.draw('r-')

        plt.waitforbuttonpress(0.1)
        if not plt.fignum_exists(1):
            print('Terminating')
            break


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    
    


if __name__ == "__main__":
    main()