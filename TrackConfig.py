import cv2
import mediapipe as mp
import numpy as np
import json
import sys
from typing import TypeVar


class TrackConfig:
    JSON = TypeVar('JSON')

    def __init__(self, game: str, config: JSON):
        self.game = game
        self.config = config

