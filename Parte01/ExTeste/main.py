#!/usr/bin/env python3
import numpy as np
import cv2

def main():
    print('Creating a new image!')

    image = np.ndarray((240,328),dtype=np.uint8)
    cv2.imshow('window',image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()