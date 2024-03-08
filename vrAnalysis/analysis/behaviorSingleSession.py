import time
from copy import copy
from tqdm import tqdm
import numpy as np
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

from .. import helpers
from .standardAnalysis import standardAnalysis


class behaviorSingleSession(standardAnalysis):
    """
    Performs basic behavioral analysis on single sessions.

    Takes as required input a vrexp object. Optional inputs define parameters of analysis.
    Standard usage:
    ---------------
    == I just started this file, will populate standard usage later! ==
    """

    def __init__(
        self,
        vrexp,
        autoload=True,
        distStep=1,
        speedThreshold=5,
        numcv=2,
        standardizeSpks=True,
    ):
        self.name = "behaviorSingleSession"
        self.vrexp = vrexp
        self.distStep = distStep
        self.speedThreshold = speedThreshold
        self.numcv = numcv

        # automatically load data
        self.dataloaded = False
        self.load_fast_data()
        if autoload:
            self.load_data()

    def envnum_to_idx(self, envnum):
        """
        convert list of environment numbers to indices of environment within this session
        e.g. if session has environments [1,3,4], and environment 3 is requested, turn it into index 1
        """
        envnum = helpers.check_iterable(envnum)
        return [np.where(self.environments == ev)[0][0] if ev in self.environments else np.nan for ev in envnum]

    def load_fast_data(self):
        # get environment data
        self.trial_envnum = self.vrexp.loadone("trials.environmentIndex")
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)

    def load_data(self, distStep=None, speedThreshold=None, numcv=None):
        """load standard data for basic behavioral analysis"""
        pass

    def clear_data(self):
        """method for clearing data to free up memory"""
        del self.omap
        del self.smap
        del self.distedges
        del self.distcenters
        del self.numTrials
        del self.boolFullTrials
        del self.idxFullTrials
        del self.idxFullTrialEachEnv
        self.dataloaded = False
