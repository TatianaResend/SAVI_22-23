#!/usr/bin/env python3

#Exercício 4 - Destaca o Wally?
import cv2
import numpy as np

def main():
    
    imgRGB = cv2.imread('./images/scene.jpg')
    imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./images/wally.png')

    H,W,_ = imgRGB.shape
    h,w,_ = template.shape
    
    # Apply template Matching
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(imgRGB,template,method)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    imgGui = imgRGB * 0 #black image the same size as the original
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    color = (255,0,0) #BGR format
    
    mask = np.zeros((H,W)).astype(np.uint8)
    cv2.rectangle(mask,top_left, bottom_right, color, -1)

    mask_bool = mask.astype(bool)
    imgGui[mask_bool] = imgRGB[mask_bool]

    negated_mask = np.logical_not(mask_bool)
    imgGRay_3 = cv2.merge([imgGray,imgGray,imgGray])
    imgGui[negated_mask] = imgGRay_3[negated_mask]

    cv2.imshow('window',imgGui) 
    cv2.waitKey(0)

if __name__ == "__main__":
    main()