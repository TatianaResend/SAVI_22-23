#!/usr/bin/env python3

import numpy as np
import cv2
print(cv2.__version__)
# from matplotlib import pyplot as plt

# def main():
#     img = cv2.imread('blox.jpg')
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    
#     corners = np.int0(corners)
#     for i in corners:
#         x,y = i.ravel()
#         cv2.circle(img,(x,y),3,255,-1)
#     plt.imshow(img),plt.show()

# if __name__ == "__main__":
#     main()