#!/usr/bin/env python3

#Exercício 1 - Nightfall
from pickletools import uint8
import cv2
import numpy as np

def main():
    img = cv2.imread('./images/lake.jpg')
    h,w,_ = img.shape

    for i in np.arange(1,0.8,-0.01):
        img_new = (img*i).astype(np.uint8)
        img[:,int((w/2)):w,:] = img_new[:,int((w/2)):w,:]

        cv2.imshow('window',img)  
        cv2.waitKey(100)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()