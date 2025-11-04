# inclusions
from dataclasses import dataclass
from typing import Union
from datetime import datetime
import numpy as np
import scipy as sp
import scipy.io as scio
from .. import helpers
from ..sessions.base import LoadingRecipe
from ..sessions import B2Session
from .behavior import register_behavior
from .oasis import oasis_deconvolution
from .redcell import RedCellProcessing
from .defaults import B2RegistrationOpts, DefaultRigInfo


@dataclass(init=False)
class B2Registration(B2Session):
    def __init__(self, mouse_name: str, date_string: str, session_id: str, **user_opts: dict):
        super().__init__(mouse_name, date_string, session_id)
        self.opts = B2RegistrationOpts(**user_opts)

        if not self.data_path.exists():
            raise ValueError(f"Session directory does not exist for {self.session_print()}")

        if not self.one_path.exists():
            self.one_path.mkdir(parents=True)

    def _additional_loading(self):
        """Override to skip loading registered data.

        Registration objects produce registered data rather than load it,
        so we skip the parent's _additional_loading() which tries to load
        from saved JSON files.
        """
        pass

    def do_preprocessing(self):
        if self.opts.clearOne:
            self.clear_one_data(certainty=True)
        self.process_timeline()
        self.process_behavior()
        self.process_imaging()
        self.process_red_cells()
        self.process_facecam()
        self.process_behavior_to_imaging()

    # --------------------------------------------------------------- preprocessing methods ------------------------------------------------------------
    def process_timeline(self):
        # load these files for raw behavioral & timeline data
        self.load_timeline_structure()
        self.load_behavior_structure()

        # get time stamps, photodiode, trial start and end times, room position, lick times, trial idx, visual data visible
        mpepStartTimes = []
        for mt, me in zip(self.tl_file["mpepUDPTimes"], self.tl_file["mpepUDPEvents"]):
            if isinstance(me, str):
                if "TrialStart" in me:
                    mpepStartTimes.append(mt)
                elif "StimStart" in me:
                    mpepStartTimes.append(mt)

        mpepStartTimes = np.array(mpepStartTimes)
        timestamps = self.get_timeline_var("timestamps")  # load timestamps

        # Get rotary position -- (load timeline measurement of rotary encoder, which is a circular position counter, use vrExperiment function to convert to a running measurement of position)
        rotaryEncoder = self.get_timeline_var("rotaryEncoder")
        rotaryPosition = self.convert_rotary_encoder_to_position(rotaryEncoder, self.vr_file["rigInfo"])

        # Get Licks (uses an edge counter)
        lickDetector = self.get_timeline_var("lickDetector")  # load lick detector copy
        lickSamples = np.where(helpers.diffsame(lickDetector) == 1)[0].astype(np.uint64)  # timeline samples of lick times

        # Get Reward Commands (measures voltage of output -- assume it's either low or high)
        rewardCommand = self.get_timeline_var("rewardCommand")  # load reward command signal
        rewardCommand = np.round(rewardCommand / np.max(rewardCommand))
        rewardSamples = np.where(helpers.diffsame(rewardCommand) > 0.5)[0].astype(np.uint64)  # timeline samples when reward was delivered

        # Now process photodiode signal
        photodiode = self.get_timeline_var("photoDiode")  # load lick detector copy

        # Remove any slow trends
        pdDetrend = sp.signal.detrend(photodiode)
        pdDetrend = (pdDetrend - pdDetrend.min()) / pdDetrend.ptp()

        # median filter and take smooth derivative
        hfpd = 10
        refreshRate = 30  # hz
        refreshSamples = int(1.0 / refreshRate / np.mean(np.diff(timestamps)))
        pdMedFilt = sp.ndimage.median_filter(pdDetrend, size=refreshSamples)
        pdDerivative, pdIndex = helpers.fivePointDer(pdMedFilt, hfpd, returnIndex=True)
        pdDerivative = sp.stats.zscore(pdDerivative)
        pdDerTime = timestamps[pdIndex]

        # find upward and downward peaks, not perfect but in practice close enough
        locUp = sp.signal.find_peaks(pdDerivative, height=1, distance=refreshSamples / 2)
        locDn = sp.signal.find_peaks(-pdDerivative, height=1, distance=refreshSamples / 2)
        flipTimes = np.concatenate((pdDerTime[locUp[0]], pdDerTime[locDn[0]]))
        flipValue = np.concatenate((np.ones(len(locUp[0])), np.zeros(len(locDn[0]))))
        flipSortIdx = np.argsort(flipTimes)
        flipTimes = flipTimes[flipSortIdx]
        flipValue = flipValue[flipSortIdx]

        # Naive Method (just look for flips before and after trialstart/trialend mpep message:
        # A sophisticated message uses the time of the photodiode ramps, but those are really just for safety and rare manual curation...
        firstFlipIndex = np.array([np.where(flipTimes >= mpepStart)[0][0] for mpepStart in mpepStartTimes])
        startTrialIndex = helpers.nearestpoint(flipTimes[firstFlipIndex], timestamps)[0]  # returns frame index of first photodiode flip in each trial

        # Check that first flip is always down -- all of the vrControl code prepares trials in this way
        if datetime.strptime(self.date, "%Y-%m-%d") >= datetime.strptime("2022-08-30", "%Y-%m-%d"):
            # But it didn't prepare it this way before august 30th :(
            assert np.all(flipValue[firstFlipIndex] == 0), f"In session {self.sessionPrint()}, first flips in trial are not all down!!"

        # Check shapes of timeline arrays
        assert timestamps.ndim == 1, "timelineTimestamps is not a 1-d array!"
        assert timestamps.shape == rotaryPosition.shape, "timeline timestamps and rotary position arrays do not have the same shape!"

        # Save timeline oneData
        self.saveone(timestamps, "wheelPosition.times")
        self.saveone(rotaryPosition, "wheelPosition.position")
        self.saveone(timestamps[lickSamples], "licks.times")
        self.saveone(timestamps[rewardSamples], "rewards.times")
        self.saveone(timestamps[startTrialIndex], "trials.startTimes")
        self.preprocessing.append("timeline")

    def process_behavior(self):
        self = register_behavior(self, self.opts.vrBehaviorVersion)

        # Confirm that vrBehavior has been processed
        self.preprocessing.append("vrBehavior")

    def process_imaging(self):
        if not self.opts.imaging:
            print(f"In session {self.session_print()}, imaging setting set to False in opts['imaging']. Skipping image processing.")
            return None

        if not self.s2p_path.exists():
            raise ValueError(f"In session {self.session_print()}, suite2p processing was requested but suite2p directory does not exist.")

        # identifies which planes were processed through suite2p (assume that those are all available planes)
        # identifies which s2p outputs are available from each plane
        self.set_value("planeNames", [plane.parts[-1] for plane in self.s2p_path.glob("plane*/")])
        self.set_value("planeIDs", [int(planeName[5:]) for planeName in self.get_value("planeNames")])
        npysInPlanes = [[npy.stem for npy in list((self.s2p_path / planeName).glob("*.npy"))] for planeName in self.get_value("planeNames")]
        commonNPYs = list(set.intersection(*[set(npy) for npy in npysInPlanes]))
        unionNPYs = list(set.union(*[set(npy) for npy in npysInPlanes]))
        if set(commonNPYs) < set(unionNPYs):
            print(
                f"The following npy files are present in some but not all plane folders within session {self.session_print()}: {list(set(unionNPYs) - set(commonNPYs))}"
            )
            print(f"Each plane folder contains the following npy files: {commonNPYs}")
        self.set_value("available", commonNPYs)  # a list of npy files available in each plane folder

        # required variables (anything else is either optional or can be computed independently)
        required = ["stat", "ops", "F", "Fneu", "iscell"]
        if not self.opts.oasis:
            # add deconvolved spikes to required variable if we aren't recomputing it here
            required.append("spks")
        for varName in required:
            assert varName in self.get_value("available"), f"{self.session_print()} is missing {varName} in at least one suite2p folder!"
        # get number of ROIs in each plane
        self.set_value("roiPerPlane", [iscell.shape[0] for iscell in self.load_s2p("iscell", concatenate=False)])
        # get number of frames in each plane (might be different!)
        self.set_value("framePerPlane", [F.shape[1] for F in self.load_s2p("F", concatenate=False)])
        assert_msg = f"The frame count in {self.session_print()} varies by more than 1 frame! ({self.get_value('framePerPlane')})"
        assert np.max(self.get_value("framePerPlane")) - np.min(self.get_value("framePerPlane")) <= 1, assert_msg
        self.set_value("numROIs", np.sum(self.get_value("roiPerPlane")))  # number of ROIs in session
        # number of frames to use when retrieving imaging data (might be overwritten to something smaller if timeline handled improperly)
        self.set_value("numFrames", np.min(self.get_value("framePerPlane")))

        # Get timeline sample corresponding to each imaging volume
        timeline_timestamps = self.loadone("wheelPosition.times")
        changeFrames = (
            np.append(
                0,
                np.diff(np.ceil(self.get_timeline_var("neuralFrames") / len(self.get_value("planeIDs")))),
            )
            == 1
        )
        frame_samples = np.where(changeFrames)[0]  # TTLs for each volume (increments by 1 for each plane)
        frame_to_time = timeline_timestamps[frame_samples]  # get timelineTimestamps of each imaging volume

        # Handle mismatch between number of imaging frames saved by scanImage (and propagated through suite2p), and between timeline's measurement of the scanImage frame counter
        if len(frame_to_time) != self.get_value("numFrames"):
            if len(frame_to_time) - 1 == self.get_value("numFrames"):
                # If frame_to_time had one more frame, just trim it and assume everything is fine. This happens when a new volume was started but not finished, so does not required communication to user.
                frame_samples = frame_samples[:-1]
                frame_to_time = frame_to_time[:-1]
            elif len(frame_to_time) - 2 == self.get_value("numFrames"):
                print(
                    "frame_to_time had 2 more than suite2p output. This happens sometimes. I don't like it. I think it's because scanimage sends a TTL before starting the frame"
                )
                frame_samples = frame_samples[:-2]
                frame_to_time = frame_to_time[:-2]
            else:
                # If frameSamples has too few frames, it's possible that the scanImage signal to timeline was broken but scanImage still continued normally.
                numMissing = self.get_value("numFrames") - len(frame_samples)  # measure number of missing frames
                if numMissing < 0:
                    # If frameSamples had many more frames, generate an error -- something went wrong that needs manual inspection
                    print(
                        f"In session {self.session_print()}, frameSamples has {len(frame_samples)} elements, but {self.get_value('numFrames')} frames were reported in suite2p. Cannot resolve."
                    )
                    raise ValueError("Cannot fix mismatches when suite2p data is missing!")
                # It's possible that the scanImage signal to timeline was broken but scanImage still continued normally.
                if numMissing > 1:
                    print(
                        f"In session {self.session_print()}, more than one frameSamples sample was missing. Consider using tiff timelineTimestamps to reproduce accurately."
                    )
                print(
                    (
                        f"In session {self.session_print()}, frameSamples has {len(frame_samples)} elements, but {self.get_value('numFrames')} frames were saved by suite2p. "
                        "Will extend frameSamples using the typical sampling rate and nearestpoint algorithm."
                    )
                )
                # If frame_to_time difference vector is consistent within 1%, then use mean (which is a little more accurate), otherwise use median
                frame_to_time = timeline_timestamps[frame_samples]
                medianFramePeriod = np.median(np.diff(frame_to_time))  # measure median sample period
                consistentFrames = np.all(
                    np.abs(np.log(np.diff(frame_to_time) / medianFramePeriod)) < np.log(1.01)
                )  # True if all frames take within 1% of median frame period
                if consistentFrames:
                    samplePeriod_f2t = np.mean(np.diff(frame_to_time))
                else:
                    samplePeriod_f2t = np.median(np.diff(frame_to_time))
                appendFrames = frame_to_time[-1] + samplePeriod_f2t * (
                    np.arange(numMissing) + 1
                )  # add elements to frame_to_time, assume sampling rate was perfect
                frame_to_time = np.concatenate((frame_to_time, appendFrames))
                frame_samples = helpers.nearestpoint(frame_to_time, timeline_timestamps)[0]

        # average percentage difference between all samples differences and median -- just a useful metric to be saved --
        self.set_value(
            "samplingDeviationMedianPercentError", np.exp(np.mean(np.abs(np.log(np.diff(frame_to_time) / np.median(np.diff(frame_to_time))))))
        )
        self.set_value(
            "samplingDeviationMaximumPercentError", np.exp(np.max(np.abs(np.log(np.diff(frame_to_time) / np.median(np.diff(frame_to_time))))))
        )

        # recompute deconvolution if requested
        spks = self.load_s2p("spks")
        if self.opts.oasis:
            # set parameters for oasis and get corrected fluorescence traces
            g = np.exp(-1 / self.opts.tau / self.opts.fs)
            fcorr = self.loadfcorr(try_from_one=False)
            results = oasis_deconvolution(fcorr, g)
            ospks = np.stack(results)

            # Check that the shape is correct
            msg = f"In session {self.session_print()}, oasis was run and did not produce the same shaped array as suite2p spks..."
            assert ospks.shape == spks.shape, msg

        # save onedata (no assertions needed, loadS2P() handles shape checks and this function already handled any mismatch between frameSamples and suite2p output
        self.saveone(frame_to_time, "mpci.times")
        self.saveone(LoadingRecipe("S2P", "F", transforms=["transpose"]), "mpci.roiActivityF")
        self.saveone(LoadingRecipe("S2P", "Fneu", transforms=["transpose"]), "mpci.roiNeuropilActivityF")
        self.saveone(LoadingRecipe("S2P", "spks", transforms=["transpose"]), "mpci.roiActivityDeconvolved")
        if "redcell" in self.get_value("available"):
            self.saveone(LoadingRecipe("S2P", "redcell", transforms=["idx_column1"]), "mpciROIs.redS2P")
        self.saveone(LoadingRecipe("S2P", "iscell"), "mpciROIs.isCell")
        self.saveone(self.get_roi_position(), "mpciROIs.stackPosition")
        if self.opts.oasis:
            self.saveone(ospks.T, "mpci.roiActivityDeconvolvedOasis")
        self.preprocessing.append("imaging")

    def process_facecam(self):
        print("Facecam preprocessing has not been coded yet!")

    def process_behavior_to_imaging(self):
        if not self.opts.imaging:
            print(f"In session {self.session_print()}, imaging setting set to False in opts['imaging']. Skipping behavior2imaging processing.")
            return None

        # compute translation mapping from behave frames to imaging frames
        idx_behave_to_frame = helpers.nearestpoint(self.loadone("positionTracking.times"), self.loadone("mpci.times"))[0]
        self.saveone(idx_behave_to_frame.astype(int), "positionTracking.mpci")

    def process_red_cells(self):
        if not (self.opts.imaging) or not (self.opts.redCellProcessing):
            return  # if not requested, skip function
        # if imaging was processed and redCellProcessing was requested, then try to preprocess red cell features
        if "redcell" not in self.get_value("available"):
            print(f"In session {self.session_print()}, 'redcell' is not an available suite2p output, although 'redCellProcessing' was requested.")
            return

        # create RedCellProcessing object
        red_cell_processing = RedCellProcessing(self)

        # compute red-features
        dot_parameters = {"lowcut": 12, "highcut": 250, "order": 3, "fs": 512}
        corr_parameters = {"width": 20, "lowcut": 12, "highcut": 250, "order": 3, "fs": 512}
        phase_parameters = {"width": 40, "eps": 1e6, "winFunc": "hamming"}

        print(f"Computing red cell features for {self.session_print()}... (usually takes 10-20 seconds)")
        dot_product = red_cell_processing.compute_dot(plane_idx=None, **dot_parameters)
        corr_coeff = red_cell_processing.compute_corr(plane_idx=None, **corr_parameters)
        phase_corr = red_cell_processing.cropped_phase_correlation(plane_idx=None, **phase_parameters)[3]

        # initialize annotations
        self.saveone(np.full(self.get_value("numROIs"), False), "mpciROIs.redCellIdx")
        self.saveone(np.full((2, self.get_value("numROIs")), False), "mpciROIs.redCellManualAssignments")

        # save oneData
        self.saveone(dot_product, "mpciROIs.redDotProduct")
        self.saveone(corr_coeff, "mpciROIs.redPearson")
        self.saveone(phase_corr, "mpciROIs.redPhaseCorrelation")
        self.saveone(np.array(dot_parameters), "parametersRedDotProduct.keyValuePairs")
        self.saveone(np.array(corr_parameters), "parametersRedPearson.keyValuePairs")
        self.saveone(np.array(phase_parameters), "parametersRedPhaseCorrelation.keyValuePairs")

    # -------------------------------------- methods for handling timeline data produced by rigbox ------------------------------------------------------------
    def load_timeline_structure(self):
        tl_file_name = self.data_path / f"{self.date}_{self.session_id}_{self.mouse_name}_Timeline.mat"  # timeline.mat file name
        self.tl_file = scio.loadmat(tl_file_name, simplify_cells=True)["Timeline"]  # load matlab structure

    def timeline_inputs(self, ignore_timestamps=False):
        if not hasattr(self, "tl_file"):
            self.load_timeline_structure()
        hw_inputs = [hwInput["name"] for hwInput in self.tl_file["hw"]["inputs"]]
        if ignore_timestamps:
            return hw_inputs
        return ["timestamps", *hw_inputs]

    def get_timeline_var(self, var_name):
        if not hasattr(self, "tl_file"):
            self.load_timeline_structure()
        if var_name == "timestamps":
            return self.tl_file["rawDAQTimestamps"]
        else:
            inputNames = self.timeline_inputs(ignore_timestamps=True)
            assert var_name in inputNames, f"{var_name} is not a tl_file in session {self.session_print()}"
            return np.squeeze(self.tl_file["rawDAQData"][:, np.where([inputName == var_name for inputName in inputNames])[0]])

    def convert_rotary_encoder_to_position(self, rotaryEncoder, rigInfo):
        # rotary encoder is a counter with a big range that sometimes flips around it's axis
        # first get changes in encoder position, fix any big jumps in value, take the cumulative movement and scale to centimeters
        rotary_movement = helpers.diffsame(rotaryEncoder)
        idx_high_values = rotary_movement > 2 ** (rigInfo.rotaryRange - 1)
        idx_low_values = rotary_movement < -(2 ** (rigInfo.rotaryRange - 1))
        rotary_movement[idx_high_values] -= 2**rigInfo.rotaryRange
        rotary_movement[idx_low_values] += 2**rigInfo.rotaryRange
        return rigInfo.rotEncSign * np.cumsum(rotary_movement) * (2 * np.pi * rigInfo.wheelRadius) / rigInfo.wheelToVR

    # -------------------------------------- methods for handling vrBehavior data produced by vrControl ------------------------------------------------------------
    def load_behavior_structure(self):
        vr_file_name = self.data_path / f"{self.date}_{self.session_id}_{self.mouse_name}_VRBehavior_trial.mat"  # vrBehavior output file name
        self.vr_file = scio.loadmat(vr_file_name, struct_as_record=False, squeeze_me=True)
        if "rigInfo" not in self.vr_file.keys():
            print(f"Assuming default settings for B2 using `DefaultRigInfo()` in session: {self.session_print()}!!!")
            self.vr_file["rigInfo"] = DefaultRigInfo()
        if not (hasattr(self.vr_file["rigInfo"], "rotaryRange")):
            self.vr_file["rigInfo"].rotaryRange = 32

    def convert_dense(self, data: Union[np.ndarray, sp.sparse.spmatrix]) -> np.ndarray:
        data = data[: self.get_value("numTrials")]
        if sp.sparse.issparse(data):
            data = data.toarray().squeeze()
        else:
            data = np.asarray(data).squeeze()
        return data

    def create_index(self, time_stamps):
        # requires timestamps as (numTrials x numSamples) dense numpy array
        if np.any(np.isnan(time_stamps)):
            return [np.where(~np.isnan(t))[0] for t in time_stamps]  # in case we have dense timestamps with nans where no data
        else:
            return [np.nonzero(t)[0] for t in time_stamps]  # in case we have sparse timestamps with 0s where no data

    def get_vr_data(self, data, nzindex):
        return [d[nz] for (d, nz) in zip(data, nzindex)]
