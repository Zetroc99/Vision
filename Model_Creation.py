import cv2 as cv
import mediapipe as mp
import numpy as np
import time

COUNT = 1
MAX_IMAGES = 50

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 640)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    cv.imshow('Test', image)

    if cv.waitKey(1) == 32 and COUNT < 3:
        cv.imwrite('model_img/image_{}.png'.format(COUNT), image)
        print('Frame {} saved successfully'.format(COUNT))
        COUNT += 1

    if COUNT == MAX_IMAGES:
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
