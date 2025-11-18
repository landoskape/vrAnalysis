from tqdm import tqdm
import numpy as np
from vrAnalysis import helpers
from vrAnalysis.database import get_database
from scipy.sparse import csc_array

# Go through all the sessions with imaging
sessiondb = get_database("vrSessions")
ises = sessiondb.iter_sessions(imaging=True)

# These are the files we will save as sparse arrays in onedata
one_files = ["mpci.roiSignificantFluorescence", "mpci.roiSignificantFluorescenceBase", "mpci.roiSignificantFluorescenceRebase"]

# Do we want to redo the computations?
redo_preprocessing = False


if __name__ == "__main__":

    for isession, session in enumerate(ises, start=1):

        # Check if npy version is present for any of the one files
        for one_file in one_files:
            if (session.one_path / f"{one_file}.npy").exists():
                # Delete the npy version -- we're saving a sparse array now
                (session.one_path / f"{one_file}.npy").unlink()

            if redo_preprocessing:
                if (session.one_path / f"{one_file}.npz").exists():
                    # Delete the npz version as well -- we're redoing the computations
                    (session.one_path / f"{one_file}.npz").unlink()

        files_all_exist = all(one_file in session.print_saved_one() for one_file in one_files)
        if files_all_exist and not redo_preprocessing:
            print(f"Skipping {session} because it already has significant fluorescence data")
            continue

        print(f"Processing {session} ({isession} / {len(ises)})")

        # Load fluorescence data (num_frames, num_rois)
        fcorr = session.loadfcorr().T
        ftimes = session.loadone("mpci.times")

        # Get standardized dff
        percentile = 30
        window_duration = 60
        dff_std = helpers.get_standardized_dff(fcorr, session.opts.fs, percentile, window_duration)

        # Measure significant transients
        # This is an array of shape (num_frames, num_rois, num_thresholds)
        # Each element is a boolean indicating whether the frame is significant at the given threshold
        threshold_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        significant_transients = helpers.get_significant_transients(dff_std, threshold_levels=threshold_levels, verbose=True)

        keep_frames = np.any(significant_transients, axis=2)
        significant_fluorescence = np.zeros_like(fcorr)
        significant_fluorescence[keep_frames] = fcorr[keep_frames]

        # Baseline the significant fluorescence
        significant_min_greater_than_0 = np.minimum.reduce(
            significant_fluorescence,
            axis=0,
            where=significant_fluorescence > 0,
            initial=np.inf,
        )
        significant_min_greater_than_0 = np.nan_to_num(significant_min_greater_than_0, posinf=0.0)
        significant = csc_array(significant_fluorescence)
        significant_base = csc_array(np.maximum(significant_fluorescence, 0))
        significant_rebase = csc_array(np.maximum(significant_fluorescence - significant_min_greater_than_0, 0))

        # Save the significant fluorescence data
        session.saveone(significant, "mpci.roiSignificantFluorescence", sparse=True)
        session.saveone(significant_base, "mpci.roiSignificantFluorescenceBase", sparse=True)
        session.saveone(significant_rebase, "mpci.roiSignificantFluorescenceRebase", sparse=True)
        print(f".....saved significant fluorescence data for session {session}.")

        session.clear_cache()
