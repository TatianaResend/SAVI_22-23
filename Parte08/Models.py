#!/usr/bin/env python3

import csv
import pickle
from copy import deepcopy
from random import randint, uniform
from turtle import color

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageMosaic():
    """Defines the model of an image mosaic
    """

    def __init__(self, q_image, t_image):
        self.q_image = q_image
        self.t_image = t_image

        self.q_image = self.q_image.astype(float) / 255.0
        self.t_image = self.t_image.astype(float) / 255.0

        self.q_height, self.q_width, _ = q_image.shape
        self.t_height, self.t_width, _ = t_image.shape

        self.mask = q_image[:,:,0] > 0
        self.randomizeParams()
      

    def randomizeParams(self):
        # start with neutral values
        self.q_scale = 1.0
        self.q_bias = 0.0
        self.t_scale = 1.0
        self.t_bias = 0.0


    def correctImages(self):

        # Query image
        self.q_image_c = self.q_scale * self.q_image + self.q_bias  # correction
        self.q_image_c[self.q_image_c>1] = 1    # saturate at 1
        self.q_image_c[self.q_image_c<0] = 0    # under saturate at 0

        # Target image
        self.t_image_c = self.t_scale * self.t_image + self.t_bias  # correction
        self.t_image_c[self.t_image_c>1] = 1    # saturate at 1
        self.t_image_c[self.t_image_c<0] = 0    # under saturate at 0


    def objectiveFunction(self, params):
        
        # Assume order q_scale, q_bias, t_scale, t_bias
        self.q_scale = params[0]
        self.q_bias = params[1]
        self.t_scale = params[2]
        self.t_bias = params[3]

        # correct images with the parameters
        self.correctImages()

        residuals = []  #each residual will be the difference of a pixels
        # for y in range(0,self.t_height):
        #     for x in range(0,self.t_width):
        #         if self.overlap_mask[y,x]:  #inside overlap region
        #             residual = (self.t_image_c[y,x] - self.q_image_c[y,x])**2
        #             residuals.append(residual)

        diffs = np.abs(self.t_image_c - self.q_image_c)
        diffs_in_overlap = diffs[self.mask]
        residuals = np.sum(diffs_in_overlap)

        # error is the sum of the residuals
        error = np.sum(residuals)
        print('error=' + str(error))

        # Draw for visualization
        self.draw()
        #plt.waitforbuttonpress(0.1)
        
        return residuals


    def drawFloatImage(self, win_name, image_f):
        image_uint8 = (image_f*255).astype(np.uint8)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 600, 400)
        cv2.imshow(win_name, image_uint8)


    def draw(self):

        stitched_image_f = deepcopy(self.t_image_c)
        stitched_image_f[self.mask] = (self.q_image_c[self.mask] + stitched_image_f[self.mask] ) / 2

        self.drawFloatImage('stitched_image', stitched_image_f)


        cv2.waitKey(20)