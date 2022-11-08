#!/usr/bin/env python3

import cv2
import random as r

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    image1 = cv2.imread('images/santorini/1.png')
    #image1 = cv2.resize(image1,(500,500))
   
    gray= cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    sift = cv2.SIFT_create(nfeatures=500)
    key_points, des = sift.detectAndCompute(gray,None)

    for idx, key_points in enumerate(key_points):
            x = int((key_points.pt[0]))
            y = int((key_points.pt[1]))
            color = (r.randint(0,255),r.randint(0,255),r.randint(0,255))
            cv2.circle(image1,(x,y),80,color,3)

    #kp = sift.detect(gray,None)
    #image1=cv2.drawKeypoints(gray,kp,image1)
    #cv2.imwrite('sift_keypoints.jpg',image1)



    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cv2.namedWindow('Image1',cv2.WINDOW_NORMAL)
    cv2.imshow('Image1',image1)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
