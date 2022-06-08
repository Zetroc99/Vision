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

import cv2 as cv
import numpy as np
import copy

# SOME CODE FOR JUMPING WITH HAND TRACKING
    # IF RIGHT HAND IN FIST IS RAISED:
        # PRESS SPACE BAR

points = np.array([[500, 500], [1000, 500], [1000, 1000],  [500, 1000]])  # points will come from detection rectangle
(x, y, w, h) = cv.boundingRect(points)

points = points - points.min(axis=0)

# Initialize the webcam
cap = cv.VideoCapture(0)

while True:
    # Read each frame from the webcam
    success, frame = cap.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray_img, (41, 41), 0)
    ret, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    mask = np.zeros(frame.shape, np.uint8)
    cv.drawContours(mask, [points], -1, (255, 255, 255), -1, cv.LINE_AA)
    result = cv.bitwise_and(frame, mask)

    cv.imshow("Output", thresh)

    if cv.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)