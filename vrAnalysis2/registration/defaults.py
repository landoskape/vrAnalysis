import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DefaultRigInfo:
    """this is prepared here in case the RigInfo field was not saved for behavioral data

    it's a serious fallback - the behavioral data should always have this information!
    we need to know the info to process behavior, but guessing is not a good idea so any
    time this is called, be careful and do your best to check your work.
    """

    computerName: str = "ZINKO"
    rotEncPos: str = "left"
    rotEncSign: int = -1
    wheelToVR: int = 4000
    wheelRadius: float = 9.75
    rotaryRange: int = 32


@dataclass
class B2RegistrationOpts:
    vrBehaviorVersion: int = 1
    facecam: bool = False
    imaging: bool = True
    oasis: bool = True
    moveRawData: bool = False
    redCellProcessing: bool = True
    clearOne: bool = True
    neuropilCoefficient: float = 0.7
    tau: float = 1.5
    fs: int = 6
