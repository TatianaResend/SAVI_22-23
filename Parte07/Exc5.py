#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from random import randint, uniform
from colorama import Fore , Style
from scipy.optimize import least_squares
import math 
from Model import Sinusoid

# class Sinusoid():
#     """ define the model of a line segment
#     """

#     def __init__(self, gt):

#         self.gt = gt
#         self.randomizeParams() 
#         self.first_draw = True

#     def randomizeParams(self):
#         self.a = uniform(-10,10)
#         self.b = uniform(-10,10)
#         self.h = uniform(-10,10)
#         self.k = uniform(-10,10)
        
#     def getY(self,x):
#         return self.a * math.sin(self.b) + self.b

#     def objectiveFunction(self,params):

#         self.a = params[0] 
#         self.b = params[1]
#         self.h = params[2]
#         self.k = params[3]

#         residuals = []

#         # percorrer os pontos todos do grounth true
#         for gt_x, gt_y in zip(self.gt['xs'],self.gt['ys']):
#             y = self.getY(gt_x)
#             residual = abs(y - gt_y)
#             residuals.append(residual)

#         # error is the sum of the residuals
#         error = sum(residuals)
        

#         # Draw
#         self.draw()
#         plt.waitforbuttonpress()

#         return error

#     def draw(self, color = 'b--'):
#         xi = -10
#         xf = 10
#         yi = self.getY(xi)
#         yf = self.getY(xf)

#         if self.first_draw:
#             self.draw_handle = plt.plot([xi,xf],[yi,yf],color, linewidth=2)
#             self.first_draw = False
#         else:
#             plt.setp(self.draw_handle, data=([xi,xf],[yi,yf]))  #update ln
        

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

    
    # set new values
    model.randomizeParams()

    result = least_squares(model.objectiveFunction,[model.a, model.b, model.h , model.k],verbose=2)


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    
    


if __name__ == "__main__":
    main()