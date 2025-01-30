from tqdm import tqdm
import numpy as np
from vrAnalysis import helpers
from vrAnalysis import database
from scipy.sparse import csc_array

sessiondb = database.vrDatabase("vrSessions")

# Go through all the sessions with imaging
ises = sessiondb.iterSessions(imaging=True)


if __name__ == "__main__":

    for ses in tqdm(ises, desc="Processing sessions...", leave=True):

        # Check if npy version is present
        if (ses.onePath() / "mpci.roiSignificantFluorescence.npy").exists():
            # Delete the npy version -- we're saving a sparse array now
            (ses.onePath() / "mpci.roiSignificantFluorescence.npy").unlink()

        if "mpci.roiSignificantFluorescence" in ses.printSavedOne():
            print(f"Skipping {ses} because it already has significant fluorescence data")
            continue

        # Load fluorescence data (num_frames, num_rois)
        fcorr = ses.loadfcorr().T
        ftimes = ses.loadone("mpci.times")

        # Get standardized dff
        percentile = 30
        window_duration = 60
        dff_std = helpers.get_standardized_dff(fcorr, ses.opts["fs"], percentile, window_duration)

        # Measure significant transients
        # This is an array of shape (num_frames, num_rois, num_thresholds)
        # Each element is a boolean indicating whether the frame is significant at the given threshold
        threshold_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        significant_transients = helpers.get_significant_transients(dff_std, threshold_levels=threshold_levels, verbose=True)

        keep_frames = np.any(significant_transients, axis=2)
        significant_fluorescence = np.zeros_like(fcorr)
        significant_fluorescence[keep_frames] = fcorr[keep_frames]

        # Save the significant fluorescence data
        significant_fluorescence = csc_array(significant_fluorescence)
        ses.saveone(significant_fluorescence, "mpci.roiSignificantFluorescence", sparse=True)
        print(f"Saved significant fluorescence data for session {ses}")

        ses.clearBuffer()
