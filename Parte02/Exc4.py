#!/usr/bin/env python3

#Exercício 4 - Destaca o Wally?
import cv2
import numpy as np

def main():
    imgRGB = cv2.imread('./images/scene.jpg')
    H,W,_ = imgRGB.shape
    imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./images/wally.png')

    h,w,_ = template.shape
    
    method = cv2.TM_CCOEFF_NORMED

    # Apply template Matching
    result = cv2.matchTemplate(imgRGB,template,method)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    imgGui = imgRGB * 0

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    color = (255,0,0) #BGR format
    cv2.rectangle(imgRGB,top_left, bottom_right, color, 2)
    
    mask = np.zeros((H,W)).astype(np.uint8)
    cv2.rectangle(mask,top_left, bottom_right, color, -1)
    
    mask_bool = mask.astype(bool)
    imgGui[mask_bool] = imgRGB[mask_bool]

    negated_mask = np.logical_not(mask_bool)
    imgGui[negated_mask] = imgGray[negated_mask]
    imgGRay_3 = cv2.merge([imgGray,imgGray,imgGray])

    cv2.imshow('window',imgGray) 
    cv2.imshow('window1',imgGui) 
    cv2.imshow('window2',mask) 
    cv2.imshow('window2',imgGRay_3) 
    cv2.waitKey(0)

        

if __name__ == "__main__":
    main()