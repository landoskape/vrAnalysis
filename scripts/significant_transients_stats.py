from tqdm import tqdm
import numpy as np
from _old_vrAnalysis import helpers
from _old_vrAnalysis import database
from _old_vrAnalysis import analysis
from scipy.sparse import csc_array

sessiondb = database.vrDatabase("vrSessions")

# Go through all the sessions with imaging
ises = sessiondb.iterSessions(imaging=True)


if __name__ == "__main__":

    for ses in tqdm(ises, desc="Processing sessions...", leave=True):
        pcss = analysis.placeCellSingleSession(ses, onefile="mpci.roiSignificantFluorescence", autoload=False)

        # spkmaps = pcss.get_spkmap()
        # self.occmap, self.speedmap, _, self.rawspkmap, self.sample_counts, self.distedges = helpers.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)
