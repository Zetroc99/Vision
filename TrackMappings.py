import TrackCV as tcv
import numpy as np
from pynput import mouse, keyboard

_COMMON_OUTPUTS = ('JUMP', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
                   'CROUCH')
_COMMON_KEYS = ('LMB', 'RMB', 'SCROLL', 'SPACE','SHIFT', 'W',
                'A', 'S', 'D', 'Q', 'E', 'I', 'ESC')

mouse = mouse.Controller()
keyboard = keyboard.Controller()
