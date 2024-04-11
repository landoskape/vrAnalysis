import time
import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from . import support


class Simulator:
    def __init__(self, box_length, spacing=1):
        self.box_length = box_length
        self.spacing = spacing

    def set_environment(self, box_length=None, spacing=None):
        self.box_length = box_length or self.box_length
        self.spacing = spacing or self.spacing
        self.env = Environment(self.box_length, self.spacing)
