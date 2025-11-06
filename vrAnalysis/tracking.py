from pathlib import Path
from functools import wraps
import numpy as np
import pandas as pd
import joblib


# import package
from .sessions import B2Session
from . import helpers
from . import files
from roicat_support import tracking as roicat_tracking


def handle_idx_ses(func):
    """decorator to handle the idx_ses argument in a standard way for the tracker class"""

    @wraps(func)
    def wrapper(tracker_instance: "Tracker", *args, idx_ses=None, **kwargs):
        idx_ses = tracker_instance.get_idx_session(idx_ses=idx_ses)
        return func(tracker_instance, *args, idx_ses=idx_ses, **kwargs)

    return wrapper


def get_tracker(mouse_name: str, try_cache: bool = True) -> "Tracker":
    """get tracker object for a particular mouse
    (This is here for backwards compatibility with old code)
    """
    return Tracker(mouse_name)


class Tracker:
    _session_info: dict[str, list[str]]

    def __init__(self, mouse_name: str):
        """create tracker object for a particular mouse"""
        self.mouse_name: str = mouse_name
        self.tracking_path: Path = roicat_tracking.roicat_tracking_directory(mouse_name)

        # Get tracking data
        self.labels = self.get_tracking_files("labels")
        self.sample_silhouettes = self.get_tracking_files("sample_silhouettes")
        self.cluster_silhouettes = self.get_tracking_files("cluster_silhouettes")

        # identify sessions that were tracked and create session objects for them
        self.sessions: list[B2Session] = [B2Session.create(*session_name) for session_name in self.session_names]
        self.num_sessions: int = len(self.session_names)

    @classmethod
    def from_session(cls, session: B2Session, verbose: bool = False) -> "Tracker":
        """create tracker object for a particular session"""
        try:
            tracker = cls(session.mouse_name)
            if session in tracker.sessions:
                return tracker
            else:
                if verbose:
                    print(f"Tracker available for mouse {session.mouse_name} but not for session {session.session_name}")
                return None
        except Exception:
            if verbose:
                print(f"Tracker not available for mouse {session.mouse_name}")
            return None

    @property
    def session_names(self) -> list[tuple[str, str, str]]:
        if not hasattr(self, "_session_info"):
            self.get_tracking_files("labels")
        return list(zip(self._session_info["mouse_name"], self._session_info["date"], self._session_info["session"]))

    def __repr__(self):
        return f"Tracker(mouse_name={self.mouse_name}, num_sessions={self.num_sessions})"

    def mouse_path(self) -> Path:
        """path to mouse folder (assuming Alyx format)"""
        return files.local_data_path() / self.mouse_name

    def get_idx_session(self, idx_ses: list[int] | None = None) -> list[int]:
        return idx_ses if idx_ses is not None else np.arange(self.num_sessions)

    def get_tracking_files(self, filetype: str) -> list[np.ndarray] | np.ndarray:
        """get tracking files for a particular filetype"""
        tracker_files = list(self.tracking_path.glob(f"*{filetype}.npy"))
        if len(tracker_files) == 0:
            raise ValueError(f"No {filetype} files found for mouse {self.mouse_name}")
        if filetype not in ["cluster_silhouettes"]:
            # For other filetypes, there is one file per session so we should check that the session info is consistent
            # And identify sessions that were tracked
            mouse_names, dates, sessions = map(list, zip(*[roicat_tracking.identify_tracking_file_session(str(p)) for p in tracker_files]))
            if not hasattr(self, "_session_info"):
                self._session_info = {"mouse_name": mouse_names, "date": dates, "session": sessions}
            else:
                assert np.all(self._session_info["mouse_name"] == mouse_names), "mouse_name mismatch"
                assert np.all(self._session_info["date"] == dates), "date mismatch"
                assert np.all(self._session_info["session"] == sessions), "session mismatch"
        data = [np.load(p) for p in tracker_files]
        if filetype in ["cluster_silhouettes"]:
            data = data[0]
        return data

    @handle_idx_ses
    def get_tracked_idx(self, *, idx_ses: list[int] | None = None, use_session_filters: bool = True, keep_method: str = "any"):
        """get index to tracked ROIs for a list of sessions"""
        if len(idx_ses) == 1:
            raise ValueError("Can't track for a single session! Received idx_ses={idx_ses}.")

        labels = [self.labels[i] for i in idx_ses]
        sample_silhouettes = [self.sample_silhouettes[i] for i in idx_ses]

        # Get array of labels shared by all requested sessions
        shared_labels = np.intersect1d(labels[0], labels[1])
        for i in range(2, len(labels)):
            shared_labels = np.intersect1d(shared_labels, labels[i])

        # Get rid of -1 placeholder for no cluster found
        shared_labels = shared_labels[shared_labels >= 0]

        # For each session, get index to shared cluster label for each ROI
        # This is a (num_sessions, num_rois) array where each row contains
        # the index the ROIs in that session organized by the shared labels
        idx_tracked = np.stack([helpers.index_in_target(shared_labels, session_labels)[1] for session_labels in labels])

        # Get cluster_ids and quality metrics for tracked ROIs over these sessions
        cluster_ids = labels[0][idx_tracked[0]]
        sample_silhouettes_tracked = np.stack([ss[idx] for idx, ss in zip(idx_tracked, sample_silhouettes)])

        if use_session_filters:
            # apply session filters
            idx_rois = [self.sessions[ii].idx_rois for ii in idx_ses]
            tracked_idx_rois = np.stack([ir[it] for ir, it in zip(idx_rois, idx_tracked)], axis=0)
            if keep_method == "any":
                roi_valid_in_all = np.any(tracked_idx_rois, axis=0)
            elif keep_method == "all":
                roi_valid_in_all = np.all(tracked_idx_rois, axis=0)
            else:
                raise ValueError(f"keep_method must be either 'any' or 'all', not {keep_method}")
            idx_tracked = idx_tracked[:, roi_valid_in_all]
            cluster_ids = cluster_ids[roi_valid_in_all]
            sample_silhouettes_tracked = sample_silhouettes_tracked[:, roi_valid_in_all]

        # Get cluster silhouettes for tracked ROIs (including a session filter if requested)
        cluster_silhouettes_tracked = self.cluster_silhouettes[cluster_ids]

        extras = dict(
            cluster_ids=cluster_ids,
            sample_silhouettes=sample_silhouettes_tracked,
            cluster_silhouettes=cluster_silhouettes_tracked,
        )
        return idx_tracked, extras

    def get_cluster_idx(self, cluster_id: int):
        """get index to cluster in each tracked session

        Returns a list of length num_sessions, where each element is the index of the cluster in the session
        if the cluster is tracked in that session, and -1 otherwise.
        """
        if cluster_id < 0 or cluster_id >= self.cluster_silhouettes.shape[0]:
            raise ValueError(f"Cluster id {cluster_id} is out of range")
        cluster_idx = []
        for isession, label in enumerate(self.labels):
            if cluster_id in label:
                session_position = np.where(label == cluster_id)[0]
                if len(session_position) > 1:
                    raise ValueError(f"Cluster id {cluster_id} found in multiple positions for session {isession}")
                cluster_idx.append(session_position[0])
            else:
                cluster_idx.append(-1)
        return np.array(cluster_idx)
