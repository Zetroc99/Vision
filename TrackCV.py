import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1],
                                                            a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_pose_lm():
    pose = mp.solutions.pose.PoseLandmark.__members__.items()
    pose_landmarks = [name for name, member in pose]
    return {k: v for v, k in enumerate(pose_landmarks)}


class TrackCV:
    """ TrackCV class

    vvvvvvvvvvv

    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    def __init__(self, tracked_angles):
        """Initialize MotionTracking Display

        """
        # self.landmarks = landmarks  # list of strings
        self.pose_lm = get_pose_lm()  # add hands later
        # for item in landmarks:
        #    if item in self.pose_lm:
        #        self.landmarks.append(item)
        self.tracked_angles = tracked_angles  # dict of center joint and

    def track(self):
        """

        :return:
        """

        cap = cv2.VideoCapture(0)  # input should be arg
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

                for lms in self.tracked_angles.values():
                    try:
                        p_landmarks = results.pose_landmarks.landmark
                        # get coordinates
                        lm_coords = self.get_coordinates(p_landmarks, lms)
                        # calculate angle
                        angle = calculate_angle(lm_coords[0],
                                                lm_coords[1],
                                                lm_coords[2])
                        print(angle)
                        # visualize
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
                if cv2.waitKey(1) == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

    def get_coordinates(self, landmarks, lms):  # lm is str
        coordinates = [[
            landmarks[getattr(self.mp_holistic.PoseLandmark, lms[i]).value].x,
            landmarks[getattr(self.mp_holistic.PoseLandmark, lms[i]).value].y]
            for i in range(3)
        ]
        return np.array(coordinates)  # returns 3x2 array
