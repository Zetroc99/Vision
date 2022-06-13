# THIS FILE IS MORE OF A TESTING GROUND FOR NEW IDEAS

import cv2 as cv
import mediapipe as mp
import numpy as np

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


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
w = 1280
h = 600
cap.set(3, w)
cap.set(4, h)

# creates box around hands
with mp_holistic.Holistic(min_detection_confidence=0.8,
                          min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        # gray_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # blur = cv.GaussianBlur(gray_img, (41, 41), 0)
        # ret, thresh = cv.threshold(blur, 100, 255,
        #                           cv.THRESH_BINARY + cv.THRESH_OTSU)

        results = holistic.process(frame)
        frame.flags.writeable = True
        box = frame
        try:
            hand_landmarks = results.right_hand_landmarks.landmark

            x_min, y_min, x_max, y_max = get_bbox_coordinates(hand_landmarks,
                                                              w, h)
            # shows highlighted box around the ROI
            # cv.rectangle(frame, (x_min - 50, y_min - 50),
            #             (x_max + 50, y_max + 100), (0, 255, 0), 2)
            points = np.array(
                [[x_min - 50, y_min - 50], [x_max + 50, y_min - 50],
                 [x_max + 50, y_max + 100], [x_min - 50, y_max + 100]])

        except:
            pass
        try:
            mask = np.zeros(frame.shape, np.uint8)
            cv.drawContours(mask, [points], -1, (255, 255, 255), -1, cv.LINE_AA)
        except:
            pass
        else:
            box = cv.bitwise_and(frame, mask)

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        cv.imshow('Test', cv.flip(box, 1))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
