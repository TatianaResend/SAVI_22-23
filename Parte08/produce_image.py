#!/usr/bin/env python3

import cv2
import random as r
from copy import deepcopy
import numpy as np
import plotly as plt


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    """
        Load images
    """
    # Query image
    q_path = 'images/machu_pichu/2.png'
    q_image = cv2.imread(q_path)
    q_gui = deepcopy(q_image)
    q_gray = cv2.cvtColor(q_image,cv2.COLOR_BGR2GRAY)
    q_win_name = 'Query Image'
    
    # Target image
    t_path = 'images/machu_pichu/1.png'
    t_image = cv2.imread(t_path)
    t_gui = deepcopy(t_image) 
    t_gray= cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
    t_win_name = 'Target Image'
    
    # ------------------------------------------
    # Execution
    # ------------------------------------------
    """ 
        Create SIFT
        (Scale-Invariant Feature Transform)
    """
    #create sift detectionobject
    sift = cv2.SIFT_create(nfeatures=200)

    """
        Detect KPs 
    """
    q_key_points, q_des = sift.detectAndCompute(q_gray,None)
    t_key_points, t_des = sift.detectAndCompute(t_gray,None)
   
    """
        Matches/Filter 
    """
    # Match features, FLANN parameters
    index_params = dict(algorithm = 1, trees = 15)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    best_two_matches = flann.knnMatch(q_des,t_des,k=2)

    # create a list containing only the best matches, 
    # and use David Lowe's test to compute the uniqueness of a match
    matches = []
    for best_two_match in best_two_matches:
        best_mach = best_two_match[0]
        second_best_match = best_two_match[1]
        
        # David Lowe's test
        if best_mach.distance < 0.3 * second_best_match.distance:
            matches.append(best_mach)

    """
        Find Homography
    """
    # First we need to create the np.array of size (n_pts,1,2) to feed into the findhomography
    num_pts = len(matches)
    q_pts_array= np.ndarray((num_pts,1,2), dtype=np.float32) # create a np array of the query poins to feed find
    t_pts_array = np.ndarray((num_pts,1,2), dtype=np.float32) # create a np array of the target poins to feed find

    for idx_match, match in enumerate(matches):
        
        q_idx = match.queryIdx
        q_x = q_key_points[q_idx].pt[0]
        q_y = q_key_points[q_idx].pt[1]
        q_pts_array[idx_match,0,0] = q_x
        q_pts_array[idx_match,0,1] = q_y

        t_idx = match.trainIdx
        t_x = t_key_points[t_idx].pt[0]
        t_y = t_key_points[t_idx].pt[1]
        t_pts_array[idx_match,0,0] = t_x
        t_pts_array[idx_match,0,1] = t_y
    
    M, mask = cv2.findHomography(q_pts_array, t_pts_array, cv2.RANSAC,5.0)
    #   q_pts_array -> coordinates of the points in the original plane
    #   t_pts_array -> coordinates of the points in the target plane
    #   method -> method used to compute a homography matrix (RANSAC - based robust method)
    #   ransacReprojThreshold -> Maximum allowed reprojection error to treat a point pair as an inlier 
    
    
    """
        Stich image

        T[y1t:y2t,x1t:x2t]=Q
        Averag t wit Q
        T[y1t:y2t,x1t:x2t]= (Q + T[y1t:y2t,x1t:x2t])/2
    """
    q_h,q_w,_ = q_image.shape
    t_h,t_w,_ = t_image.shape

    # Stitch images together
    stitched_image_h = t_h
    stitched_image_w = t_w

    q_image_warped = cv2.warpPerspective(q_image,M,(stitched_image_w,stitched_image_h))
    #   src -> imput image
    #   M -> 3x3 tranformation matrix
    #   dsize -> size of the output image

    #   dst -> output image that has the size dsize and the same type as src . 
    

    q_image_warped = q_image_warped[:,:,0:3]  #remove fourth channel
    
    #stitch image
    overlap_mask = q_image_warped > 0
    stitched_image = deepcopy(t_image)
    stitched_image[overlap_mask] = ((q_image_warped[overlap_mask].astype(float) + stitched_image[overlap_mask].astype(float) ) / 2).astype(np.uint8)

    # Draw keypoints
    for idx, key_point in enumerate(q_key_points):
            x = int((key_point.pt[0]))
            y = int((key_point.pt[1]))
            color = (r.randint(0,255),r.randint(0,255),r.randint(0,255))
            cv2.circle(q_gui,(x,y),30,color,3)

    for idx, key_point in enumerate(t_key_points):
            x = int((key_point.pt[0]))
            y = int((key_point.pt[1]))
            color = (r.randint(0,255),r.randint(0,255),r.randint(0,255))
            cv2.circle(t_gui,(x,y),30,color,3)

    # Show the matches image
    matches_image = cv2.drawMatches(q_image,q_key_points,t_image,t_key_points,matches,None)


    cv2.namedWindow(q_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(q_win_name, 600, 400)
    cv2.imshow(q_win_name, q_gui)

    cv2.namedWindow(t_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(t_win_name, 600, 400)
    cv2.imshow(t_win_name, t_gui)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', matches_image)

    cv2.namedWindow('q_image_warped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('q_image_warped', 600, 400)
    cv2.imshow('q_image_warped', q_image_warped)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 600, 400)
    cv2.imshow('mask', (overlap_mask*255).astype(np.uint8))

    cv2.namedWindow('stitched_image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('stitched_image', 600, 400)
    cv2.imshow('stitched_image', stitched_image)
    # ------------------------------------------
    # Termination
    # ------------------------------------------
  

    cv2.imwrite('./images/machu_pichu/query_warped.png',q_image_warped)
    cv2.imwrite('./images/machu_pichu/t_image.png',t_image)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
