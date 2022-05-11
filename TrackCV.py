import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c) -> float:
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
    assert isinstance(a, np.ndarray), 'a is not a numpy array'
    assert isinstance(b, np.ndarray), 'b is not a numpy array'
    assert isinstance(c, np.ndarray), 'c is not a numpy array'
    # convert x,y coordinates of a,b,c to radians
    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1],
                                                               a[0] - b[0])
    angle = np.abs(radian * 180.0 / np.pi)
    # prevents angle from going beyond 180 deg
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_coordinates(landmarks, lms) -> np.ndarray:
    """
    Gets the interior angles of three tracked landmarks
    and returns them as a Numpy Array
    :param landmarks: google.protobuf.pyext._message.RepeatedCompositeContainer w/ size of total landmarks
    :param lms: list of length 3 with numbers that correspond to the landmark values
    :return: 3x2 Numpy Array made up of floating values from 0 to 180
    """

    coordinates = [[
        landmarks[lms[i]].x,
        landmarks[lms[i]].y]
        for i in range(len(lms))
    ]
    return np.array(coordinates)  # returns 3x2 Numpy array


def show_text(image, angle, lm_coords, frame_w, frame_h):
    cv2.putText(image, str(round(angle, 2)),
                tuple(
                    np.multiply(lm_coords[1], [frame_w, frame_h]).astype(
                        int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                cv2.LINE_AA
                )


def calculate_motion(group_landmarks, tracked_group, showText, image, frame_width, frame_height):
    for lms in tracked_group.values():
        # get coordinates
        lm_coords = get_coordinates(group_landmarks, lms)
        # get angle
        angle = calculate_angle(lm_coords[0],
                                lm_coords[1],
                                lm_coords[2])
        if showText:
            show_text(image, angle, lm_coords, frame_width, frame_height)


class TrackCV:
    """ TrackCV class
    Will create an instance of body landmarks that need to be
    tracked in realtime.
    """

    # Instance variables (only applies to holistic solutions atm)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    def __init__(self, tracked_pose=None, tracked_hands=None):  # TODO make sure None args work when None in self.track
        """
        Record body joint angles that need to be tracked
        :param tracked_pose: dict consisting of triad of points that will be tracked
        """

        self.tracked_pose = tracked_pose  # dict of center joint and
        self.tracked_hands = tracked_hands

    def track(self, frame_width=640, frame_height=480, min_dc=0.8, max_tc=0.8, showText=False):
        """

        :return:
        """

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        with self.mp_holistic.Holistic(min_detection_confidence=min_dc,
                                       min_tracking_confidence=max_tc) as holistic:
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

                try:  # TODO add hands / and make function for 'for' loop contents
                    p_landmarks = results.pose_landmarks.landmark
                    lh_landmarks = results.left_hand_landmarks.landmark
                    #rh_landmarks = results.

                    # calculate pose
                    calculate_motion(p_landmarks, self.tracked_pose, showText, image, frame_width, frame_height)
                    # calculate hands
                    calculate_motion(lh_landmarks, self.tracked_hands, showText, image, frame_width, frame_height)
                except:
                    pass
                # draw pose
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                        .get_default_pose_landmarks_style())
                # TODO add hands
                # draw hands
                self.mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                        .get_default_hand_landmarks_style())
                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def get_tracked_angles(self, pose=True):
        """
        Getter for tracked angle names/key values
        :return: list of tracked angle names
        """
        if pose:
            return list(self.tracked_pose.keys())
        else:
            return list(self.tracked_hands.keys())

    # TODO possible setter angles, in case more want to be added
