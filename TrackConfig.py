import json
import sys
from typing import TypeVar

# TODO: will have to generate a .txt/.json file w/ a save() func


class TrackConfig:
    JSON = TypeVar('JSON')

    def __init__(self, game: str, config: JSON):
        self.game = game
        self.config = config

    def save(self):
        return None

    def modify(self):
        return None




