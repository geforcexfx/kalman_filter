from collections import deque
import numpy as np
from numpy import dot
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import argparse
import imutils
import cv2
import urllib  # for reading image from URL
import math
import time
from numpy import *
from numpy.linalg import inv
from numpy.random import randn

dt=1

lower = {'red': (166, 84, 141)}  # assign new item lower['blue'] = (93, 10, 0)
upper = {'red': (186, 255, 255)}
colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 217), 'white': (50, 0, 50)}

direction = ""
counter = 0

e = 0
corx=0
cory=0
camera = cv2.VideoCapture('C:/Users/gefor/Desktop/Animotica_29_5_13_13_40.mp4')

class Point:
    def __init__(self, x=0, y=0, c=""):
        self.x = x
        self.y = y
        self.c = c
def predict(x2, P, u, F):
    x2 = np.matmul(F, x2) + u
    P = np.matmul(F, np.matmul(P, F.transpose())) + Rt
    return [x2, P]

def update(x2, P, Z, H, R):
    y = Z - np.matmul(H, x2)
    S = np.matmul(H,np.matmul(P, H.transpose())) + R
    K = np.matmul(P,np.matmul(H.transpose(), np.linalg.pinv(S)))
    x2 = x2 + np.matmul(K, y)
    P = np.matmul((I - np.matmul(K, H)),P)

    return [x2, P]
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

x1 = np.array(np.mat('0.0; 0.0; 0.0; 0.0'))           # initial state (location and velocity)
P = np.array(np.mat('10000, 0, 0, 0; 0, 10000, 0, 0; 0, 0, 10000, 0; 0, 0, 0, 10000'))    # initial uncertainty
u =  np.array(np.mat('0;0;0;0'))                                # external motion
F =  np.array(np.mat('1, 0, 0.1, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))          # next state function
H =  np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))                     # measurement function
R =  np.array(np.mat('10000, 0; 0, 10000'))                                    # measurement uncertainty
Rt =  np.array(np.mat('10000,0,0,0;0,10000,0,0;0,0,10000,0;0,0,0,10000'))           # next state uncertainty
I =  np.array(np.mat('1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1'))

while True:
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    for key, value in upper.items():
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 2:
                time1=time.time()
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame, key + " ball", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[key], 2)
                if key == 'red':
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                p = Point(int(x), int(y), key)
                corx = p.x
                cory = p.y

                x1, P = update(x1, P, np.mat('{}; {}'.format(corx, cory)), H, R)
                x1, P = predict(x1, P, u, F)
                predictedCoords=x1
                cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
                cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                         (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
                cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        else:
            x1, P = predict(x1, P, u, F)
            predictedCoords = x1
            cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
            cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                     (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    counter += 1
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
