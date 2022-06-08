import cv2 as cv
import mediapipe as mp
# import pyautogui
# import pydirectinput
# import numpy as np
# import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def get_bbox_coordinates(landmarks, width, height):
    """
    """
    all_x, all_y = [], []  # store all x and y points in list
    for lm in landmarks:
        all_x.append(int(lm.x * width))  # multiply x by image width
        all_y.append(int(lm.y * height))  # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(
        all_y)  # return as (xmin, ymin, xmax, ymax)


cap = cv.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 1440)
w = 2560
h = 1440

with mp_holistic.Holistic(min_detection_confidence=0.8,
                          min_tracking_confidence=0.8) as holistic:
    #_, frame = cap.read()
    #h, w, c = frame.shape

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        results = holistic.process(image)
        image.flags.writeable = True
        try:
            hand_landmarks = results.right_hand_landmarks.landmark

            x_min, y_min, x_max, y_max = get_bbox_coordinates(hand_landmarks,
                                                              w, h)
            cv.rectangle(image, (x_min-100, y_min-100), (x_max+100, y_max+100),
                         (0, 255, 0), 2)

        except:
            pass
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        cv.imshow('Test', cv.flip(image,1))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
