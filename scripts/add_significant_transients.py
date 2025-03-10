from tqdm import tqdm
import numpy as np
from vrAnalysis2 import helpers
from vrAnalysis2.database import get_database
from scipy.sparse import csc_array

sessiondb = get_database("vrSessions")

# Go through all the sessions with imaging
ises = sessiondb.iter_sessions(imaging=True)


if __name__ == "__main__":

    for session in tqdm(ises, desc="Processing sessions...", leave=True):

        # Check if npy version is present
        if (session.one_path / "mpci.roiSignificantFluorescence.npy").exists():
            # Delete the npy version -- we're saving a sparse array now
            (session.one_path / "mpci.roiSignificantFluorescence.npy").unlink()

        if "mpci.roiSignificantFluorescence" in session.print_saved_one():
            print(f"Skipping {session} because it already has significant fluorescence data")
            continue

        else:
            print(f"Processing {session}")

        # Load fluorescence data (num_frames, num_rois)
        fcorr = session.loadfcorr().T
        ftimes = session.loadone("mpci.times")

        # Get standardized dff
        percentile = 30
        window_duration = 60
        dff_std = helpers.get_standardized_dff(fcorr, session.opts["fs"], percentile, window_duration)

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
        session.saveone(significant_fluorescence, "mpci.roiSignificantFluorescence", sparse=True)
        print(f"Saved significant fluorescence data for session {session}")

        session.clear_cache()
