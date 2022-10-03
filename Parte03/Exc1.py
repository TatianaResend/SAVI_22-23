#!/usr/bin/env python3

from pickle import FALSE
import cv2
import numpy as np
from time import sleep

#Constants
width_min = 100; # rectangle width
height_min = 100; # rectangle height
offset = 6  # Allowed error between pixels
pos_line = 600  # Count line position
delay = 60  # FPS do vídeo
detec = []
line_x1 = 150; line_x2 = 1160

def find_center(x, y, widh, height):
    """
    :param x: x of the object
    :param y: y of the object
    :param width: object width
    :param height: object height
    :return: double that contains the coordinates of the center of an object
    """
    x1 = widh // 2
    y1 = height // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

def set_info(detec):
    global cars
    for (x, y) in detec:
        if (pos_line + offset) > y > (pos_line - offset):
            cars += 1
            cv2.line(frame1, (line_x1, pos_line), (line_x2, pos_line), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(cars))

def show_info(frame1, img_work):
    text = f'Carros: {cars}'
    cv2.putText(frame1, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", img_work)

cars = 0
cap = cv2.VideoCapture('./docs/traffic.mp4')
#subtraction = cv2.bgsegm.createBackgroundSubtractorMOG()  # Take the background and subtract from what's moving
subtraction = cv2.createBackgroundSubtractorKNN(detectShadows=False)    # Take the background and subtract from what's moving

while True:
    ret, frame1 = cap.read()  # Grab every frame of the video
    tempo = float(1 / delay)
    sleep(tempo)  # Delay between each processing

    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # RGB to GRAY
    blur = cv2.GaussianBlur(gray, (3, 3), 5)  # image smoothing

    img_sub = subtraction.apply(blur)  # Subtraction of the image applied in the blur
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        5, 5))  # 5x5 matrix, where the matrix format between 0 and 1 forms an ellipse inside
    element = np.ones((5,5))

    img_work = img_sub
    img_work = cv2.erode(img_work, np.ones((5,5)),iterations=2)
    img_work = cv2.dilate(img_work, np.ones((4,4)), iterations=2)
    img_work = cv2.morphologyEx(img_work, cv2.MORPH_CLOSE, kernel)  # fill in holes
       
    contour, img = cv2.findContours(img_work, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (line_x1, pos_line), (line_x2, pos_line), (255, 127, 0), 3)
    cv2.line(img_work, (line_x1, pos_line), (line_x2, pos_line), (255, 127, 0), 3)
    
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contorno = (w >= width_min) and (h >= height_min)
        if not validate_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = find_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)


    set_info(detec)
    show_info(frame1, img_work)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

# https://github.com/gustavogino/Vehicle-Counter/blob/master/main.py