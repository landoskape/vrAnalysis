import numpy as np
from dataclasses import asdict
from functools import wraps
from typing import Any
from math import floor, ceil

from vrAnalysis2.sessions.b2session import B2SessionParams
from vrAnalysis2.processors.spkmaps import SpkmapProcessor, SpkmapParams, Maps
from vrAnalysis2.tracking import Tracker
from vrAnalysis2.helpers import resolve_dataclass, argsort, named_transpose
from vrAnalysis2.helpers.debug import tic, toc, Timer


def handle_idx_ses(func):
    """
    Decorator to handle the idx_ses argument for MultiSessionSpkmaps methods.

    This ensures that methods can be called with either a specific idx_ses or
    will use all sessions that contain the specified environment.
    """

    @wraps(func)
    def wrapper(msm_instance: "MultiSessionSpkmaps", envnum, *args, idx_ses=None, **kwargs):
        # If no specific sessions are requested, use all sessions with the requested environment
        if idx_ses is None:
            idx_ses = msm_instance.idx_ses_with_env(envnum)
        else:
            if not msm_instance.check_env_in_sessions(envnum, idx_ses):
                raise ValueError(f"Requested environment {envnum} is not present in all requested sessions")

        # Call the original function with the processed arguments
        return func(msm_instance, envnum, *args, idx_ses=idx_ses, **kwargs)

    return wrapper


class MultiSessionSpkmaps:
    """
    Manages multiple sessions for spikemap analysis, similar to placeCellMultiSession
    but using the modern architecture with SpkmapProcessor.

    This class holds a collection of session processors and provides methods to:
    1. Get spkmaps across multiple sessions
    2. Get reliability values
    3. Handle tracked ROIs across sessions
    4. Get red cell indices

    It leverages the caching mechanisms of SpkmapProcessor for performance.
    """

    tracker: Tracker

    # Non-initialized fields
    processors: list[SpkmapProcessor]
    spkmap_params: SpkmapParams
    session_params: B2SessionParams

    def __init__(
        self,
        tracker: Tracker,
        spkmap_params: SpkmapParams | dict[str, Any] | None = None,
        session_params: B2SessionParams | dict[str, Any] | None = None,
    ):
        self.tracker = tracker
        self.spkmap_params = resolve_dataclass(spkmap_params, SpkmapParams)
        self.session_params = resolve_dataclass(session_params, B2SessionParams)

        for session in self.tracker.sessions:
            session.update_params(**asdict(self.session_params))
        self.processors = [SpkmapProcessor(session, self.spkmap_params) for session in self.tracker.sessions]

    def env_stats(self) -> dict[int, list[int]]:
        """
        helper for getting environment stats in sessions

        returns a dictionary where the keys represent the environments (by index)
        contained in at least one session from **self** (i.e. in self.pcss) and
        the keys represent the list of session indices in which the environment is
        present (e.g. env_stats()[1] is a list of session indices pointing to the
        self.pcss[i] such that environment 1 is in self.pcss[i].environments)
        """
        all_envs = [processor.session.environments for processor in self.processors]
        envs = sorted(list(set([env for envs in all_envs for env in envs])))
        return dict(zip(envs, [self.idx_ses_with_env(env) for env in envs]))

    def env_selector(self, envmethod="most"):
        """
        method for selecting environment based on some rules

        envmethod is a string or int indicating which environment to use
        envmethod=='most':
            return the environment with the most sessions
        envmethod=='least':
            return the environment with the least sessions
        envmethod=='first':
            return the environment the mouse experienced first
        envmethod=='second':
            return the environment the mouse experienced second
        envmethod=='last':
            return the environment the mouse experienced last
        envmethod== int :
            return the environment the mouse experienced Nth where N is the provided value of envmethod (or last to first if negative)
            uses standard python indexing (0 means first, -1 means last)
        """
        # get environment stats
        stats = self.env_stats()

        # pick environment
        if envmethod in ("most", "least"):
            # get number of sessions per environment
            num_per_env = [len(idx_ses) for idx_ses in stats.values()]
            # get index to environment with most or least sessions
            if envmethod == "most":
                envidx = num_per_env.index(max(num_per_env))
            else:
                envidx = num_per_env.index(min(num_per_env))
            envnum = list(stats.keys())[envidx]

        elif envmethod in ("first", "second", "last") or isinstance(envmethod, int):
            if envmethod == "last":
                envmethod = -1  # convert to integer representation
            elif isinstance(envmethod, str):
                # convert to integer representation
                envmethod = {val: idx for idx, val in enumerate(["first", "second"])}[envmethod]

            # get first session in each environment
            first_ses = [idx_ses[0] for idx_ses in stats.values()]
            envnum = list(stats.keys())[argsort(first_ses)[envmethod]]

        else:
            raise ValueError("did not recognize env method. see docstring")

        return envnum

    def idx_ses_selector(self, envnum, sesmethod="all"):
        """
        method for selecting index of sessions with envnum based on some rules

        sesmethod is a string or int or float between -1 < 1, indicating which sessions to pick
        sesmethod=='all':
            return all sessions in the environment
        sesmethod=='first':
            return first session in environment
        sesmethod=='last':
            return last session in environment
        sesmethod==float in between -1 and 1
            return first or last fraction of sessions in environment (first if positive, last if negative)
        sesmethod==integer:
            return first or last N sessons where N is the value of sesmethod (first if positive, last if negative)
        """
        # get environment stats
        stats = self.env_stats()

        # pick sessions
        if sesmethod == "all":
            idx_ses = stats[envnum]

        elif sesmethod in ("first", "last"):
            sesnum = dict(zip(("first", "last"), [0, -1]))[sesmethod]
            idx_ses = stats[envnum][sesnum]

        elif (-1 < sesmethod < 0) or (0 < sesmethod < 1) or isinstance(sesmethod, int):
            if -1 < sesmethod < 0:
                num_ses = len(stats[envnum])
                sesmethod = floor(sesmethod * num_ses)
            elif 0 < sesmethod < 1:
                num_ses = len(stats[envnum])
                sesmethod = ceil(sesmethod * num_ses)
            if sesmethod < 0:
                idx_ses = stats[envnum][sesmethod:]
            else:
                idx_ses = stats[envnum][:sesmethod]

        else:
            raise ValueError("did not recognize sesmethod. see docstring")

        return idx_ses

    def env_idx_ses_selector(self, envmethod="most", sesmethod="all", verbose=False):
        """
        method for selecting environment and index of sessions based on some rules

        see env_selector and idx_ses_selector for explanations
        """
        envnum = self.env_selector(envmethod=envmethod)
        idx_ses = self.idx_ses_selector(envnum, sesmethod=sesmethod)
        if verbose:
            print(
                self.tracker.mouse_name,
                f"Env ({envmethod}): {envnum}, Sessions ({sesmethod}): {idx_ses}",
            )
        return envnum, idx_ses

    def idx_ses_with_env(self, envnum):
        """Return the indices of sessions that contain the specified environment"""
        return [ii for ii, processor in enumerate(self.processors) if envnum in processor.session.environments]

    def check_env_in_sessions(self, envnum: int, idx_ses: list[int]) -> bool:
        """
        Check if environment is present in all specified sessions.

        Parameters
        ----------
        envnum : int
            Environment number
        idx_ses : list[int]
            Indices of sessions to check

        Returns
        -------
        bool
            True if environment is in all sessions, False otherwise
        """
        return all(envnum in self.processors[i].session.environments for i in idx_ses)

    @handle_idx_ses
    def get_spkmaps(
        self,
        envnum: int,
        *,
        average: bool = True,
        smooth: float | None = None,
        reliability_method: str | None = None,
        spks_type: str | None = None,
        tracked: bool = True,
        use_session_filters: bool = True,
        pop_nan: bool = True,
        idx_ses: list[int] = None,
        include_latents: bool = False,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Get spkmaps for the specified environment and sessions.

        Parameters
        ----------
        envnum : int
            Environment number
        average : bool, default=True
            Whether to average maps across trials
        smooth : float or None, default=None
            Smoothing parameter (overrides self.params.smooth_width)
        reliability_method : str, default="leave_one_out"
            Which reliability method to get values for
        spks_type : str or None, default=None
            Which spks type to get maps for
        tracked : bool, default=True
            Whether to include only tracked cells
        use_session_filters : bool, default=True
            Whether to use session filters to select "good" cells
        pop_nan : bool, default=True
            Whether to remove NaN values
        idx_ses : list[int] or int or None, default=None
            Indices of sessions to process
        include_latents : bool, default=False
            Whether to include latent variables (not implemented yet)

        Returns
        -------
        list[np.ndarray]
            list of spkmaps, one for each selected session
        extras : dict
            Dictionary containing additional information about the spkmaps
        """
        # For updating SpkmapParams if requested
        params = {}
        if smooth is not None:
            params["smooth_width"] = float(smooth)
        if reliability_method is not None:
            params["reliability_method"] = reliability_method
        if spks_type is not None:
            for processor in self.processors:
                processor.session.update_params(spks_type=spks_type)

        # If we're using tracked cells, get tracking indices for requested sessions
        if tracked:
            # If using tracked cells, then these indices will determine which cells are included based
            # on the session filters, so we should turn _off_ use_session_filters for the remaining
            # processing steps!!!
            idx_tracked, tracking_extras = self.tracker.get_tracked_idx(idx_ses=idx_ses, use_session_filters=use_session_filters)
            _use_session_filters = False
        else:
            idx_tracked = None
            tracking_extras = None
            _use_session_filters = use_session_filters

        # If we need to filter the remaining variables by session filters, get them
        if _use_session_filters:
            idx_rois = [self.processors[i].session.idx_rois for i in idx_ses]

        # Get spkmaps
        spkmaps = []
        for ii, ises in enumerate(idx_ses):
            # First get maps for this session (it'll be all of them, probably cached so very fast)
            c_env_maps = self.processors[ises].get_env_maps(use_session_filters=_use_session_filters, params=params)
            # Then filter to the requested environment and add to spkmap list
            c_env_maps.filter_environments([envnum])
            spkmaps.append(c_env_maps.spkmap[0])

        # Retrieve red cell idx
        idx_red = [self.processors[i].session.get_red_idx() for i in idx_ses]
        if _use_session_filters:
            idx_red = [ired[iroi] for ired, iroi in zip(idx_red, idx_rois)]

        # Get reliability values for ROIs
        reliability = []
        for ii, ises in enumerate(idx_ses):
            c_rel = self.processors[ises].get_reliability(use_session_filters=_use_session_filters, params=params)
            c_rel = c_rel.filter_by_environment([envnum])
            reliability.append(c_rel.values[0])

        if tracked:
            spkmaps = [sm[it] for sm, it in zip(spkmaps, idx_tracked)]
            idx_red = [ired[it] for ired, it in zip(idx_red, idx_tracked)]
            reliability = [rel[it] for rel, it in zip(reliability, idx_tracked)]

        if average:
            spkmaps = [np.nanmean(s, axis=1) for s in spkmaps]

        positions = []
        for ii, ises in enumerate(idx_ses):
            positions.append(self.processors[ises].dist_centers)
            if ii > 0:
                if not np.allclose(positions[0], positions[ii]):
                    raise ValueError("All sessions must have the same distance centers... something went wrong!")
        positions = positions[0]

        if pop_nan:
            axis = 0 if average else (0, 1)
            idx_nan_positions = np.any(np.stack([np.any(np.isnan(s), axis=axis) for s in spkmaps]), axis=0)
            spkmaps = [s[..., ~idx_nan_positions] for s in spkmaps]
            positions = positions[~idx_nan_positions]

        pfloc, pfidx = named_transpose([self.get_place_field(spkmap, method="max", positions=positions) for spkmap in spkmaps])

        extras = dict(
            idx_tracked=idx_tracked,
            idx_red=idx_red,
            reliability=reliability,
            pfloc=pfloc,
            pfidx=pfidx,
            positions=positions,
        )
        if tracking_extras is not None:
            extras.update(tracking_extras)

        return spkmaps, extras

    @handle_idx_ses
    def get_reliability(
        self,
        envnum: int,
        *,
        smooth: float | None = None,
        reliability_method: str | None = None,
        spks_type: str | None = None,
        tracked: bool = True,
        use_session_filters: bool = True,
        idx_ses: list[int] = None,
        include_latents: bool = False,
        use_saved_spkmap: bool = False,
        force_reload_pcss: bool = False,
        **kwargs,
    ) -> list[Maps]:
        """
        Get reliability values for the specified environment and sessions.

        Parameters
        ----------
        envnum : int
            Environment number
        smooth : float or None, default=None
            Smoothing parameter (overrides self.params.smooth_width)
        reliability_method : str, default="leave_one_out"
            Which reliability method to get values for
        spks_type : str or None, default=None
            Which spks type to get maps for
        tracked : bool, default=True
            Whether to include only tracked cells
        use_session_filters : bool, default=True
            Whether to use session filters to select "good" cells
        idx_ses : list[int] or int or None, default=None
            Indices of sessions to process
        include_latents : bool, default=False
            Whether to include latent variables
        use_saved_spkmap : bool, default=False
            Whether to use saved spkmaps
        force_reload_pcss : bool, default=False
            Whether to force reload of data
        **kwargs : dict
            Additional arguments for SpkmapProcessor

        Returns
        -------
        list[Maps]
            list of Maps objects, one for each selected session
        """
        # For updating SpkmapParams if requested
        params = {}
        if smooth is not None:
            params["smooth_width"] = smooth
        if reliability_method is not None:
            params["reliability_method"] = reliability_method
        if spks_type is not None:
            for processor in self.processors:
                processor.session.update_params(spks_type=spks_type)

        # If we're using tracked cells, get tracking indices for requested sessions
        if tracked:
            # If using tracked cells, then these indices will determine which cells are included based
            # on the session filters, so we should turn _off_ use_session_filters for the remaining
            # processing steps!!!
            idx_tracked, extras = self.tracker.get_tracked_idx(idx_ses=idx_ses, use_session_filters=use_session_filters)
            _use_session_filters = False
        else:
            _use_session_filters = use_session_filters

        # If we need to filter the remaining variables by session filters, get them
        if _use_session_filters:
            idx_rois = [self.processors[i].session.idx_rois for i in idx_ses]

        # Retrieve red cell idx
        idx_red = [self.processors[i].session.get_red_idx() for i in idx_ses]
        if _use_session_filters:
            idx_red = [ired[iroi] for ired, iroi in zip(idx_red, idx_rois)]

        # Get reliability values for ROIs
        reliability = []
        for ii, ises in enumerate(idx_ses):
            c_rel = self.processors[ises].get_reliability(use_session_filters=_use_session_filters, params=params)
            c_rel = c_rel.filter_by_environment([envnum])
            reliability.append(c_rel.values[0])

        if tracked:
            idx_red = [ired[it] for ired, it in zip(idx_red, idx_tracked)]
            reliability = [rel[it] for rel, it in zip(reliability, idx_tracked)]

        extras = dict(
            idx_red=idx_red,
        )
        return reliability, extras

    def get_place_field(self, spkmap: np.ndarray, method: str = "max", positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Get place field location and sorting index

        Makes an assumption about spkmap based on ndims --
        assumes either (numROIs, numPositions) or (numROIs, numTrials, numPositions)

        Note that if method is "com", then the spkmap must be nonnegative! (Will clip any negative values
        and continue, so will potentially generate buggy behavior if spkmap isn't based on mostly positive signals!)

        Parameters
        ----------
        spkmap : np.ndarray
            Spikmap to get place field for
        method : str, default="max"
            Method to use to get place field location
        positions : np.ndarray or None, default=None
            Positions to use for place field location when method is "com"

        Returns
        -------
        pfloc : np.ndarray
            Place field location for each ROI
        pfidx : np.ndarray
            Sorting index for each ROI
        """
        if spkmap.ndim == 3:
            spkmap = np.nanmean(spkmap, axis=1)

        if positions is None:
            positions = np.arange(spkmap.shape[1])

        # if method is 'com' (=center of mass), use weighted mean to get place field location
        if method == "com":
            nonnegative_profile = np.maximum(spkmap, 0)
            pfloc = np.nansum(nonnegative_profile * positions.reshape(1, -1), axis=1) / np.nansum(nonnegative_profile, axis=1)

        # if method is 'max' (=maximum rate), use maximum to get place field location
        if method == "max":
            pfloc = positions[np.nanargmax(spkmap, axis=1)]

        # Then sort...
        pfidx = np.argsort(pfloc)

        return pfloc, pfidx
