import re
import os
from pathlib import Path
from scipy import io
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .. import files
from ..helpers import PrettyDatetime

data_path = files.literature_data_path() / "pettitHarvey2022"


@dataclass
class PettitHarveySession:
    """Data class to store parsed session information with lazy loading"""

    mouse_name: str
    date: PrettyDatetime
    _behavior_file: Optional[str] = field(default=None, repr=False)
    _neural_file: Optional[str] = field(default=None, repr=False)

    # Private cache for lazy loading
    _behavior_data: Optional[Dict] = field(default=None, repr=False)
    _deconv_sm: Optional[np.ndarray] = field(default=None, repr=False)
    _dff: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def behavior(self) -> Dict:
        """Lazily load behavior data when first accessed"""
        if self._behavior_data is None:
            self._load_behavior_data()
        return self._behavior_data

    @property
    def spks(self) -> np.ndarray:
        """Lazily deconvolved calcium data when first accessed"""
        if self._deconv_sm is None:
            self._load_neural_data()
        return self._deconv_sm

    @property
    def dff(self) -> np.ndarray:
        """Lazily load delta F / F calcium data when first accessed"""
        if self._dff is None:
            self._load_neural_data()
        return self._dff

    def _load_neural_data(self):
        if self._neural_file is None or not os.path.exists(self._neural_file):
            raise FileNotFoundError(f"Neural file not found: {self._neural_file}")
        _neural_data = io.loadmat(self._neural_file, simplify_cells=True)
        self._deconv_sm = _neural_data["deconv_sm"]
        if "dff" in _neural_data:
            self._dff = _neural_data["dff"]
        else:
            self._dff = False

    def _load_behavior_data(self):
        if self._behavior_file is None or not os.path.exists(self._behavior_file):
            raise FileNotFoundError(f"Behavior file not found: {self._behavior_file}")
        self._behavior_data = io.loadmat(self._behavior_file, simplify_cells=True)
        self._behavior_data = {k: v for k, v in self._behavior_data.items() if not k.startswith("__")}

    def clear_cache(self) -> None:
        """Clear cached data to free memory"""
        self._behavior_data = None
        self._deconv_sm = None
        self._dff = None

    def has_behavior(self) -> bool:
        """Check if behavior file exists"""
        return self._behavior_file is not None and os.path.exists(self._behavior_file)

    def has_neural(self) -> bool:
        """Check if neural file exists"""
        return self._neural_file is not None and os.path.exists(self._neural_file)


def find_pettit_harvey_sessions(directory: str, include_behavior_only: bool = False) -> List[PettitHarveySession]:
    """
    Find and create PettitHarveySession objects for all sessions in directory

    Parameters:
    directory (str): Path to the directory containing the files

    Returns:
    List[PettitHarveySession]: List of session objects
    """
    directory = Path(directory)
    sessions = {}  # Dictionary to group behavior and neural files

    # Regular expression pattern to match file names
    pattern = r"(.*?)_(\d{4})(\d{2})(\d{2})_(behavior|neural)\.mat$"

    # First pass: group files by mouse and date
    for filepath in directory.glob("*.mat"):
        match = re.match(pattern, filepath.name)
        if match:
            mouse_name = match.group(1)
            year = int(match.group(2))
            month = int(match.group(3))
            day = int(match.group(4))
            data_type = match.group(5)

            # Create unique key for this session
            session_date = PrettyDatetime(year, month, day)
            session_key = (mouse_name, session_date)

            if session_key not in sessions:
                sessions[session_key] = {"behavior": None, "neural": None}

            # Store file path based on type
            sessions[session_key][data_type] = str(filepath)

    # Second pass: create PettitHarveySession objects
    session_objects = []
    for (mouse_name, date), files in sessions.items():
        session = PettitHarveySession(mouse_name=mouse_name, date=date, _behavior_file=files["behavior"], _neural_file=files["neural"])
        if session.has_neural or include_behavior_only:
            session_objects.append(session)

    return sorted(session_objects, key=lambda x: (x.mouse_name, x.date))


# @dataclass
# class MouseFile:
#     """Data class to store parsed file information"""

#     filename: str
#     mouse_name: str
#     date: datetime
#     data_type: str  # 'behavior' or 'neural'


# def parse_mouse_files(directory: str) -> List[MouseFile]:
#     """
#     Find and parse mouse data files in the specified directory

#     Parameters:
#     directory (str): Path to the directory containing the files

#     Returns:
#     List[MouseFile]: List of parsed file information
#     """
#     # Regular expression pattern
#     # Captures: (mouse_name)(year)(month)(day)(type)
#     pattern = r"(.*?)_(\d{4})(\d{2})(\d{2})_(behavior|neural)\.mat$"

#     parsed_files = []

#     # List all .mat files in directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".mat"):
#             match = re.match(pattern, filename)

#             if match:
#                 mouse_name = match.group(1)
#                 year = int(match.group(2))
#                 month = int(match.group(3))
#                 day = int(match.group(4))
#                 data_type = match.group(5)

#                 # Create datetime object
#                 file_date = datetime(year, month, day)

#                 parsed_files.append(MouseFile(filename=filename, mouse_name=mouse_name, date=file_date, data_type=data_type))

#     return parsed_files
