#!/usr/bin/env python3

from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from random import randint, uniform
import math 
import numpy as np

class Line():
    """ define the model of a line segment
    """

    def __init__(self, gt):

        self.gt = gt
        self.randomizeParams() 
        self.first_draw = True

    def randomizeParams(self):
        self.m = uniform(-2,2)
        self.b = uniform(-5,5)

    def getY(self,x):
        return self.m * x + self.b

    def objectiveFunction(self,params):

        self.m = params[0] 
        self.b = params[1]

        residuals = []

        # percorrer os pontos todos do grounth true
        for gt_x, gt_y in zip(self.gt['xs'],self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # error is the sum of the residuals
        error = sum(residuals)

        # Draw
        self.draw()
        plt.waitforbuttonpress()

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

    def objectiveFunction(self,params):

        self.a = params[0] 
        self.b = params[1]
        self.h = params[2]
        self.k = params[3]

        residuals = []

        # percorrer os pontos todos do grounth true
        for gt_x, gt_y in zip(self.gt['xs'],self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # error is the sum of the residuals
        error = sum(residuals)
        

        # Draw
        self.draw()
        plt.waitforbuttonpress()

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


class Polynomial():
    """ define the model of a line segment
    """

    def __init__(self, gt):

        self.gt = gt
        self.randomizeParams() 
        self.first_draw = True
        self.xs_for_plot = list(np.linspace(-10,10,num=500))

    def randomizeParams(self):
        self.a = uniform(-10,10)
        self.b = uniform(-10,10)
        self.c = uniform(-10,10)
        self.d = uniform(-10,10)
        self.e = uniform(-10,10)
        self.f = uniform(-10,10)
        self.g = uniform(-10,10)
        self.h = uniform(-10,10)
    
    def getY(self,x):
        return self.a + self.b *x +self.c *x*x + self.d *x*x*x + self.e *x*x*x*x 
        + self.f *x*x*x*x*x + self.g *x*x*x*x*x*x + self.h *x*x*x*x*x*x*x*x 
        
    def getYs(self,xs):
        ys = []
        for x in xs:
            ys.append(self.getY)
        return self.a * math.sin(self.b) + self.b

    def objectiveFunction(self,params):

        self.a = params[0] 
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.e = params[4]
        self.f = params[5]
        self.g = params[6]
        self.h = params[7]

        residuals = []

        # percorrer os pontos todos do grounth true
        for gt_x, gt_y in zip(self.gt['xs'],self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # error is the sum of the residuals
        error = sum(residuals)
        

        # Draw
        self.draw()
        plt.waitforbuttonpress()

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
