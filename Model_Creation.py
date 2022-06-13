import cv2 as cv
import numpy as np
import copy
import time
import mediapipe as mp

COUNT = 1
MAX_IMAGES = 50


def get_bbox_coordinates(landmarks, width, height):
    """
    """
    all_x, all_y = [], []  # store all x and y points in list
    for lm in landmarks:
        all_x.append(int(lm.x * width))  # multiply x by image width
        all_y.append(int(lm.y * height))  # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(
        all_y)  # return as (xmin, ymin, xmax, ymax)


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
w = 1280
h = 600
cap.set(3, w)
cap.set(4, h)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(min_detection_confidence=0.8,
                          min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        results = holistic.process(frame)
        frame.flags.writeable = True
        box = frame
        saved_img = frame

        try:
            hand_landmarks = results.right_hand_landmarks.landmark
        except:
            pass
        else:
            x_min, y_min, x_max, y_max = get_bbox_coordinates(hand_landmarks,
                                                              w, h)
            x_min = x_min - 50
            x_max = x_max + 50
            y_min = y_min - 50
            y_max = y_max + 100
            points = np.array([[x_min, y_min], [x_max, y_min],
                               [x_max, y_max], [x_min, y_max]])

        try:
            mask = np.zeros(frame.shape, np.uint8)
            cv.drawContours(mask, [points], -1, (255, 255, 255), -1, cv.LINE_AA)
        except:
            pass
        else:
            box = cv.bitwise_and(frame, mask)
            saved_img = frame[y_min:y_max, x_min:x_max]
            cv.imshow('Saved Image', cv.flip(saved_img, 1))
            cv.imshow('Box Image', cv.flip(box, 1))

        if cv.waitKey(1) == 32 and COUNT < 3:
            cv.imwrite('model_img/image_{}.png'.format(COUNT), saved_img)
            print('Frame {} saved successfully'.format(COUNT))
            COUNT += 1

        if COUNT == MAX_IMAGES:
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
