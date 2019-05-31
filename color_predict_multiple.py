
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib #for reading image from URL
import math

lower = {'red':(1, 50, 50),  'blue':(100,80,0), 'yellow':(23, 59, 119)}
upper = {'red':(5, 255, 255), 'blue':(140,255,255), 'yellow':(54,255,255)}
colors = {'red':(0,0,255), 'blue':(255,0,0), 'yellow':(0, 255, 217)}
direction = ""
camera = cv2.VideoCapture('C:/Users/gefor/Videos/Animotica/Animotica_31_5_19_56_17.mp4')
class Point:
     def __init__(self, x=0, y=0,c=""):
        """ Create a new point at x, y """
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

list_obj=[]
# keep looping

x1_r = np.array(np.mat('0.0; 0.0; 0.0; 0.0'))           # initial state (location and velocity)
P_r = np.array(np.mat('10000, 0, 0, 0; 0, 10000, 0, 0; 0, 0, 10000, 0; 0, 0, 0, 10000'))    # initial uncertainty
u_r =  np.array(np.mat('0;0;0;0'))                                # external motion
F_r =  np.array(np.mat('1, 0, 0.1, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))          # next state function
H_r =  np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))                     # measurement function
R_r =  np.array(np.mat('100000, 0; 0, 100000'))                                    # measurement uncertainty
Rt =  np.array(np.mat('10000,0,0,0;0,10000,0,0;0,0,10000,0;0,0,0,10000'))           # next state uncertainty
I =  np.array(np.mat('1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1'))

x1_y = np.array(np.mat('0.0; 0.0; 0.0; 0.0'))           # initial state (location and velocity)
P_y = np.array(np.mat('10000, 0, 0, 0; 0, 10000, 0, 0; 0, 0, 10000, 0; 0, 0, 0, 10000'))    # initial uncertainty
u_y =  np.array(np.mat('0;0;0;0'))                                # external motion
F_y =  np.array(np.mat('1, 0, 0.1, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))          # next state function
H_y =  np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))                     # measurement function
R_y =  np.array(np.mat('10000, 0; 0, 10000'))                                    # measurement uncertainty


x1_g = np.array(np.mat('0.0; 0.0; 0.0; 0.0'))           # initial state (location and velocity)
P_g = np.array(np.mat('10000, 0, 0, 0; 0, 10000, 0, 0; 0, 0, 10000, 0; 0, 0, 0, 10000'))    # initial uncertainty
u_g =  np.array(np.mat('0;0;0;0'))                                # external motion
F_g =  np.array(np.mat('1, 0, 0.1, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))          # next state function
H_g =  np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))                     # measurement function
R_g =  np.array(np.mat('10000, 0; 0, 10000'))                                    # measurement uncertainty

while True:
    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width=600)
 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
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
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                if key=='red':
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    x1_r, P_r = update(x1_r, P_r, np.mat('{}; {}'.format(x, y)), H_r, R_r)
                    x1_r, P_r = predict(x1_r, P_r, u_r, F_r)
                    predictedCoords = x1_r
                    print("PREDICT: ", predictedCoords)
                    cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0,0, 255], 2, 8)
                    cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                             (predictedCoords[0] + 50, predictedCoords[1] - 30), [10000, 10, 255], 2, 8)
                    cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

                if key=='blue':
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    x1_g, P_g = update(x1_g, P_g, np.mat('{}; {}'.format(x, y)), H_g, R_g)
                    x1_g, P_g = predict(x1_g, P_g, u_g, F_g)
                    predictedCoords_g = x1_g
                    cv2.circle(frame, (predictedCoords_g[0], predictedCoords_g[1]), 20, [255, 0, 0], 2, 8)
                    cv2.line(frame, (predictedCoords_g[0] + 16, predictedCoords_g[1] - 15),
                             (predictedCoords_g[0] + 50, predictedCoords_g[1] - 30), [10000, 10, 255], 2, 8)
                    cv2.putText(frame, "Predicted", (int(predictedCoords_g[0] + 50), int(predictedCoords_g[1] - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
                if key=='yellow':
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                    x1_y, P_y = update(x1_y, P_y, np.mat('{}; {}'.format(x, y)), H_y, R_y)
                    x1_y, P_y = predict(x1_y, P_y, u_y, F_y)
                    predictedCoords_y = x1_y
                    cv2.circle(frame, (predictedCoords_y[0], predictedCoords_y[1]), 20, [0, 255, 255], 2, 8)
                    cv2.line(frame, (predictedCoords_y[0] + 16, predictedCoords_y[1] - 15),
                             (predictedCoords_y[0] + 50, predictedCoords_y[1] - 30), [10000, 10, 255], 2, 8)
                    cv2.putText(frame, "Predicted", (int(predictedCoords_y[0] + 50), int(predictedCoords_y[1] - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])


        else:

            x1_g, P_g = predict(x1_g, P_g, u_g, F_g)
            predictedCoords_g = x1_g
            print("PREDICT: ", predictedCoords_g)
            # predictedCoords = kfObj.Estimate(corx, cory)
            cv2.circle(frame, (predictedCoords_g[0], predictedCoords_g[1]), 20, [255, 0, 0], 2, 8)
            cv2.line(frame, (predictedCoords_g[0] + 16, predictedCoords_g[1] - 15),
                     (predictedCoords_g[0] + 50, predictedCoords_g[1] - 30), [10000, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (int(predictedCoords_g[0] + 50), int(predictedCoords_g[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

            x1_y, P_y = predict(x1_y, P_y, u_y, F_y)
            predictedCoords_y = x1_y
            cv2.circle(frame, (predictedCoords_y[0], predictedCoords_y[1]), 20, [0, 255, 255], 2, 8)
            cv2.line(frame, (predictedCoords_y[0] + 16, predictedCoords_y[1] - 15),
                     (predictedCoords_y[0] + 50, predictedCoords_y[1] - 30), [10000, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (int(predictedCoords_y[0] + 50), int(predictedCoords_y[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
            x1_r, P_r = predict(x1_r, P_r, u_r, F_r)
            predictedCoords = x1_r
            print("PREDICTr: ", predictedCoords)
            # predictedCoords = kfObj.Estimate(corx, cory)
            cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 0, 255], 2, 8)
            cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                     (predictedCoords[0] + 50, predictedCoords[1] - 30), [10000, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])


    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
