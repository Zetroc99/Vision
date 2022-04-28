import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle between 3 adjacent landmarks
    adding up to a maximum angle of 180 degrees.
    :param a: 2x1 Numpy Array for the first angular point
    :param b: 2x1 Numpy Array for the second angular point
    :param c: 2x1 Numpy Array for the third angular point
    :return: Floating angle value between 0 and 180.

    Examples:
        >>> a1 = np.array([11,20])
        >>> b1 = np.array([23,4])
        >>> c1 = np.array([60,4])
        >>> calculate_angle(a1,b1,c1)
        126.86989764584402

    """
    # convert x,y coordinates of a,b,c to radians
    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1],
                                                               a[0] - b[0])
    angle = np.abs(radian * 180.0 / np.pi)
    # prevents angle from going beyond 180 deg
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_coordinates(landmarks, lms):
    """
    Gets the interior angles of the tracked landmark(s)
    and returns them as a Numpy Array
    :param landmarks:
    :param lms:
    :return:
    """

    coordinates = [[
        landmarks[lms[i]].x,
        landmarks[lms[i]].y]
        for i in range(len(lms))
    ]
    return np.array(coordinates)  # returns 3x2 Numpy array


class TrackCV:
    """ TrackCV class
    Will create an instance of body landmarks that need to be
    tracked in realtime.
    """

    # Instance variables (only applies to holistic solutions atm)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # TODO: change tracked_angles to output {'Left_Arm': [11, 13, 15]}
    def __init__(self, tracked_angles):
        """
        Record body joint angles that need to be tracked
        :param tracked_angles: dict consisting of triad of points that will be tracked
        """

        self.tracked_angles = tracked_angles  # dict of center joint and

    def track(self):
        """

        :return:
        """

        cap = cv2.VideoCapture(0)  # input should be arg
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        with self.mp_holistic.Holistic(min_detection_confidence=0.8,
                                       min_tracking_confidence=0.8) as holistic:
            while cap.isOpened():
                success, frame = cap.read()  # read video capture
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # Recolor Image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = holistic.process(image)

                # Recolor back to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image.flags.writeable = True
                # TODO: Make tracked_angle a dict that has k: joint and v: list of coordinates

                try:
                    p_landmarks = results.pose_landmarks.landmark
                    for lms in self.tracked_angles.values():
                        # get coordinates
                        lm_coords = get_coordinates(p_landmarks, lms)
                        # calculate angle
                        angle = calculate_angle(lm_coords[0],
                                                lm_coords[1],
                                                lm_coords[2])
                        # visualize
                        # visualize is for testing purposes, comment out when not needed
                        cv2.putText(image, str(round(angle, 2)),
                                    tuple(
                                        np.multiply(lm_coords[1], [640, 480]).astype(
                                            int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                                    cv2.LINE_AA
                                    )
                except:
                    pass
                # pose
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                        .get_default_pose_landmarks_style())

                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
