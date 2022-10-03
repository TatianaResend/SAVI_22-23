#!/usr/bin/env python3

import cv2
#from matplotlib.pyplot import imshow
import numpy as np
from time import sleep

#Constants
widh_min = 100; widht_max = 400  # rectangle widh
height_min = 100; height_max = 400  # rectangle height
offset = 6  # Allowed error between pixels
pos_line = 600  # Count line position
delay = 60  # FPS do vídeo
detec = []
line_x1 = 180
line_x2 = 1160

def pega_centro(x, y, widh, height):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param widh: largura do objeto
    :param height: altura do objeto
    :return: dupla que contém as coordenadas do centro de um objeto
    """
    x1 = widh // 2
    y1 = height // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

def set_info(detec):
    global carros
    for (x, y) in detec:
        if (pos_line + offset) > y > (pos_line - offset):
            carros += 1
            cv2.line(frame1, (line_x1, pos_line), (line_x2, pos_line), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))

def show_info(frame1, img_work):
    text = f'Carros: {carros}'
    cv2.putText(frame1, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", img_work)

carros = 0
cap = cv2.VideoCapture('./docs/traffic.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()  # Pega o fundo e subtrai do que está se movendo

while True:
    ret, frame1 = cap.read()  # Pega cada frame do vídeo
    tempo = float(1 / delay)
    sleep(tempo)  # Dá um delay entre cada processamento

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Pega o frame e transforma para preto e branco
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Faz um blur para tentar remover as imperfeições da imagem
    
    img_sub = subtracao.apply(blur)  # Faz a subtração da imagem aplicada no blur
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        5, 5))  # Cria uma matriz 5x5, em que o formato da matriz entre 0 e 1 forma uma elipse dentro
    element = np.ones((5,5))

    img_work = img_sub
    img_work = cv2.erode(img_work, element)
    img_work = cv2.morphologyEx(img_work, cv2.MORPH_CLOSE, kernel)  # Tenta preencher todos os "buracos" da imagem
       
    contorno, img = cv2.findContours(img_work, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (line_x1, pos_line), (line_x2, pos_line), (255, 127, 0), 3)
    cv2.line(img_work, (line_x1, pos_line), (line_x2, pos_line), (255, 127, 0), 3)
    
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= widh_min) and (h >= height_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_work, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)
        cv2.circle(img_work, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    #show_info(img_work, img_work)
    show_info(frame1, img_work)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

# https://github.com/gustavogino/Vehicle-Counter/blob/master/main.py