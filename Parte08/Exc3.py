#!/usr/bin/env python3

import cv2
import random as r
from copy import deepcopy
import numpy as np
import plotly as plt
from Models import ImageMosaic
from scipy.optimize import least_squares

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    """
        Load images
    """
    # Query image
    q_path = 'images/machu_pichu/1.png'
    q_image = cv2.imread(q_path)
    
    # Target image
    t_path = 'images/machu_pichu/2.png'
    t_image = cv2.imread(t_path)

    image_mosaic = ImageMosaic(q_image,t_image)
    # ------------------------------------------
    # Execution
    # ------------------------------------------

    x0 = [image_mosaic.q_scale, image_mosaic.q_bias ,image_mosaic.t_scale, image_mosaic.t_bias]
    result = least_squares(image_mosaic.objectiveFunction, x0 , verbose = 2)

    # ------------------------------------------
    # Termination
    # ------------------------------------------
  
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()