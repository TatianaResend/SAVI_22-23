#!/usr/bin/env python3

import cv2
import numpy as np

def main():

    cap = cv2.VideoCapture("./docs/OxfordTownCentre/TownCentreXVID.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    while(cap.isOpened()): # this is an infinite loop

    # ------------------------------------------
    # Execution
    # ------------------------------------------

        if cv2.waitKey(10) == ord('q'):
            break
    # ------------------------------------------
    # Termination
    # ------------------------------------------

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()