#!/usr/bin/env python3

import numpy as np
import cv2 as cv
CornerHarris = False
Shi_Tomasi = False
SIFT = False
SURF = False
FAST = False
BRIEF = False

def main():
    # select image:
    #filename = 'chessboard.png'
    #filename = 'blox.jpg'
    #filename = 'home.png'
    #filename = 'fly.png'
    filename = 'simple.png'


    # ---------------------------------------------------------------------
    # select method:
    #CornerHarris = True
    #Shi_Tomasi = True
    #SIFT = True
    #SURF = True
    #FAST = True
    BRIEF = True

    img = cv.imread(filename)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    if CornerHarris:
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray,2,3,0.04)
        # img - Input image. It should be grayscale and float32 type.
        # blockSize - It is the size of neighbourhood considered for corner detection
        # ksize - Aperture parameter of the Sobel derivative used.
        # k - Harris detector free parameter in the equation.

        #result is dilated for marking the corners, not important
        dst = cv.dilate(dst,None)
    
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[0,0,255]
        print('CornerHarris')

    if Shi_Tomasi:
        corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv.circle(img,(x,y),3,255,-1)
        print('Shi_Tomasi')

    if SIFT:
        sift = cv.SIFT_create()         #First we have to construct a SIFT object
        #kp = sift.detect(gray,None)
        kp, des = sift.detectAndCompute(gray,None)
        #img=cv.drawKeypoints(gray,kp,img)
        img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print('SIFT')

    if SURF:
        #faster than SIFT
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img,None)
        print('SURF')
        print(len(kp))

    if FAST:
        # Initiate FAST object with default values
        fast = cv.FastFeatureDetector_create()
        # find and draw the keypoints
        kp = fast.detect(img,None)
        img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        # Print all default params
        print( "Threshold: {}".format(fast.getThreshold()) )
        print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
        print( "neighborhood: {}".format(fast.getType()) )
        print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
        cv.imwrite('fast_true.png', img2)
        # Disable nonmaxSuppression
        fast.setNonmaxSuppression(0)
        kp = fast.detect(img, None)
        print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
        img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        cv.imwrite('fast_false.png', img3)

    if BRIEF:
        # Initiate FAST detector
        star = cv.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = star.detect(img,None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)
        print( brief.descriptorSize() )
        print( des.shape )

       

    
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()