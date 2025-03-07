"""
sameCellCandidates - Analysis tools for identifying and visualizing same-cell ROIs across imaging planes

This module provides tools for analyzing potential same-cell candidates in volumetric imaging data.
It helps identify ROIs (Regions Of Interest) that might represent the same cell across different
imaging planes based on their spatial relationships and activity correlations.

Quick Start
----------
Basic usage for analyzing correlation between ROIs across planes:

    from vrAnalysis.analysis import sameCellCandidates
    
    # Initialize with a VR experiment object
    scc = sameCellCandidates(vrexp)
    
    # Create histograms of correlated ROI pairs across planes
    scc.planePairHistograms(
        corrCutoff=[0.5, 0.6, 0.7, 0.8],  # correlation thresholds
        distanceCutoff=50,                 # maximum distance in μm
        withShow=True
    )

Visualization Tools
-----------------
The module provides two interactive visualization tools:

1. Basic Cluster Explorer:
    For exploring ROI clusters based on correlation and distance:

    # Initialize explorer with default parameters
    explorer = clusterExplorer(
        scc,
        corrCutoff=0.4,      # minimum correlation
        distanceCutoff=20,   # maximum distance in μm
        keep_planes=[1,2,3,4]
    )

2. ROICaT Cluster Explorer:
    For exploring clusters identified by ROICaT:

    # Initialize with ROICaT labels
    explorer = clusterExplorerROICaT(
        scc,
        roicat_labels,       # cluster labels from ROICaT
        corrCutoff=0.4,
        distanceCutoff=20
    )

Analysis Examples
---------------
1. Analyze correlation vs distance relationships:

    # Create scatter plot of correlations vs distances
    scc.scatterForThresholds(
        keep_planes=[1,2,3,4],
        distanceCutoff=250
    )

    # Plot cumulative distributions of correlations
    scc.cdfForThresholds(
        distanceCutoff=50,
        distanceDistant=(50, 250)
    )

2. Analyze cluster sizes and distributions:

    # Plot cluster size distributions
    scc.clusterSize(
        corrCutoffs=[0.3, 0.4, 0.5, 0.6, 0.7],
        distanceCutoff=30
    )

    # Analyze distance distributions
    scc.distanceDistribution(
        corrCutoffs=np.linspace(0.2, 0.6, 5),
        maxDistance=200
    )

Batch Processing
--------------
For analyzing multiple sessions:

    from vrAnalysis.database import vrDatabase
    
    vrdb = vrDatabase()
    for ses in vrdb.iterSessions(imaging=True, vrRegistration=True):
        scc = sameCellCandidates(ses)
        scc.planePairHistograms(
            corrCutoff=[0.5, 0.6, 0.7, 0.8],
            distanceCutoff=50,
            withSave=True,    # save figures
            withShow=False    # don't display
        )

Notes
-----
- All distance measurements are in micrometers (μm)
- Correlation values range from -1 to 1
- Default plane indices are [1, 2, 3, 4]
- Interactive visualizations support keyboard navigation (left/right arrows)
  and mouse interaction for ROI selection
"""

import time
from copy import copy
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, TextBox
import networkx as nx

from .. import helpers
from .standardAnalysis import standardAnalysis


def getConnectedGroups(G):
    """Utility especially useful for clusterSize()

    getConnectedGroups returns a list of lists where each list contains the ROIs that are
    connected to each other given an undirected adjacency matrix G
    """
    # start with assertions to minimize work in the case of an error
    assert G.ndim == 2 and G.shape[0] == G.shape[1], "G isn't a square"
    assert np.all(G == G.T), "G isn't symmetrical"

    # define useful methods
    def convert_to_set_representation(iG):
        N = iG.shape[0]
        graph = []
        for n in range(N):
            cidx = iG[n] >= 0
            cconn = iG[n][cidx]
            graph.append(set(cconn))
        return graph

    def get_all_connected_groups(graph):
        already_seen = set()
        result = []
        for node in range(len(graph)):
            if node not in already_seen:
                node_group = get_connected_group(graph, node)
                result.append(node_group)
                already_seen = already_seen.union(node_group)
        return result

    def get_connected_group(graph, node):
        prevSet = set([node])
        nextSet = prevSet.union(*[graph[n] for n in prevSet])
        while nextSet > prevSet:
            prevSet = copy(nextSet)
            nextSet = prevSet.union(*[graph[n] for n in prevSet])
        return nextSet

    # now get connected groups
    N = G.shape[0]
    cG = 1 * (G > 0)  # 1 if connected, 0 otherwise
    iG = np.arange(N).reshape(1, -1) * cG  # index of roi if connected, -1 if not connected (for row-column pairs)
    iG[cG == 0] = -1  # set unconnected pairs to -1
    graph = convert_to_set_representation(iG)  # retrieve set of just nonnegative indices (e.g. those that are connected) for each node
    components = get_all_connected_groups(graph)  #
    return [list(c) for c in components]


class sameCellCandidates(standardAnalysis):
    """
    Analyzes potential same-cell candidates across imaging planes in volumetric data.

    Takes as required input a vrexp object. Optional inputs define parameters of analysis,
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).

    Key Features:
    - Measures cross-correlation between ROI pairs
    - Analyzes spatial relationships and distances between ROIs
    - Provides visualization tools for ROI relationships
    - Supports filtering by correlation thresholds, distances, and planes

    Standard usage:
    ---------------
    To make histograms of the number of pairs exceeding a given correlation as a function of which plane
    each pair is in (and using any other filtering):
    vrdb = database.vrDatabase() # get database object
    for ses in vrdb.iterSessions(imaging=True, vrRegistration=True):
        # go through each session that has been registered and has imaging data
        scc = analysis.sameCellCandidates(ses)
        scc.planePairHistograms(corrCutoff=[0.5, 0.6, 0.7, 0.8], distanceCutoff=50, withSave=True, withShow=False)

    Or, for plotting the results of one session within a notebook:
    scc.planePairHistograms(corrCutoff=[0.5, 0.6, 0.7, 0.8], distanceCutoff=50, withSave=False, withShow=True)

    There's also two related methods for looking at the relationship between distance and correlation.
    One of them makes a scatter plot of all pairs passing the filtering requirements. This is:
    scc.scatterForThresholds(keep_planes=[1,2,3,4], distanceCutoff=250);

    The other makes a cumulative distribution plot for ROIs of two different distance ranges...

    Parameters
    ----------
    vrexp : object
        VR experiment object containing imaging data and ROI information
    onefile : str, optional
        Path to activity data file (default: "mpci.roiActivityDeconvolvedOasis")
    autoload : bool, optional
        Whether to automatically load data on initialization (default: True)
    keep_planes : list, optional
        List of plane indices to analyze (default: [1, 2, 3, 4])
    """

    def __init__(
        self,
        vrexp,
        onefile="mpci.roiActivityDeconvolvedOasis",
        autoload=True,
        keep_planes=[1, 2, 3, 4],
    ):
        self.name = "sameCellCandidates"
        self.onefile = onefile
        self.vrexp = vrexp
        self.keep_planes = keep_planes if keep_planes is not None else [i for i in range(len(vrexp.value["roiPerPlane"]))]

        # automatically do measurements
        self.dataloaded = False
        if autoload:
            self.load_data()

    def load_data(self, onefile=None):
        """Load standard data for measuring same cell candidates.

        Parameters
        ----------
        onefile : str, optional
            Path to activity data file. If None, uses the class's onefile attribute.
        """
        # update onefile if using a different measure of activity
        self.onefile = self.onefile if onefile is None else onefile

        self.roiPerPlane = [self.vrexp.value["roiPerPlane"][kp] for kp in sorted(self.keep_planes)]
        self.numROIs = sum(self.roiPerPlane)

        # get relevant data
        stackPosition = self.vrexp.loadone("mpciROIs.stackPosition")
        roiPlaneIdx = stackPosition[:, 2].astype(np.int32)  # plane index
        # figure out which ROIs are in the target planes
        self.idxROI_inTargetPlane = np.any(np.stack([roiPlaneIdx == pidx for pidx in self.keep_planes]), axis=0)

        npix = np.array([s["npix"] for s in self.vrexp.loadS2P("stat")[self.idxROI_inTargetPlane]]).astype(np.int32)  # roi size (in pixels of mask)
        data = self.vrexp.loadone(self.onefile)[:, self.idxROI_inTargetPlane]  # activity array
        xyPos = stackPosition[self.idxROI_inTargetPlane, 0:2] * 1.3  # xy position to um
        roiPlaneIdx = roiPlaneIdx[self.idxROI_inTargetPlane].astype(np.int32)

        # comparisons
        self.xcROIs = sp.spatial.distance.squareform(np.corrcoef(data.T), checks=False)
        self.pwDist = sp.spatial.distance.pdist(xyPos)
        self.idxRoi1, self.idxRoi2 = self.pairValFromVec(np.arange(self.numROIs), squareform=True)
        self.planePair1, self.planePair2 = self.pairValFromVec(roiPlaneIdx, squareform=True)
        self.npixPair1, self.npixPair2 = self.pairValFromVec(npix, squareform=True)
        self.xposPair1, self.xposPair2 = self.pairValFromVec(xyPos[:, 0], squareform=True)
        self.yposPair1, self.yposPair2 = self.pairValFromVec(xyPos[:, 1], squareform=True)
        self.numPairs = len(self.xcROIs)
        assert (
            self.numPairs == self.numROIs * (self.numROIs - 1) / 2
        ), f"math flex failed: numPairs={self.numPairs}, math: {self.numROIs*(self.numROIs-1)/2}"
        self.dataloaded = True

    def totalFromPairs(self, pairs):
        """Calculate total number of ROIs from number of pairs.

        Parameters
        ----------
        pairs : int
            Number of ROI pairs

        Returns
        -------
        int
            Total number of ROIs that would generate the given number of pairs
        """
        assert type(pairs) == int, "pairs is not an integer..."
        n = (1 + np.sqrt(1 + 8 * pairs)) / 2
        assert n.is_integer(), "pairs is not a valid binomial coefficient choose 2..."
        return int(n)

    def pairValFromVec(self, vector, squareform=True):
        """Convert vector to pair values.

        Parameters
        ----------
        vector : numpy.ndarray
            1D array of values
        squareform : bool, optional
            Whether to perform squareform without checks (default: True)

        Returns
        -------
        tuple
            Two arrays containing paired values from the input vector
        """
        assert isinstance(vector, np.ndarray) and vector.ndim == 1, "vector must be 1d numpy array"
        N = len(vector)
        p1 = vector.reshape(-1, 1).repeat(N, axis=1)
        p2 = vector.reshape(1, -1).repeat(N, axis=0)
        if squareform:
            p1 = sp.spatial.distance.squareform(p1, checks=False)
            p2 = sp.spatial.distance.squareform(p2, checks=False)
        return p1, p2

    def getPairFilter(
        self,
        npixCutoff=None,
        keep_planes=None,
        corrCutoff=None,
        distanceCutoff=None,
        extraFilter=None,
    ):
        """Generate boolean filter for ROI pairs based on multiple criteria.

        Parameters
        ----------
        npixCutoff : int, optional
            Minimum number of pixels for ROI masks
        keep_planes : list, optional
            List of plane indices to include
        corrCutoff : float, optional
            Minimum correlation threshold
        distanceCutoff : float, optional
            Maximum distance between ROI pairs in μm
        extraFilter : numpy.ndarray, optional
            Additional boolean filter to apply

        Returns
        -------
        numpy.ndarray
            Boolean array indicating which pairs pass all filters
        """
        assert self.dataloaded, "data is not loaded yet, use 'run()' to get key datapoints"
        if keep_planes is not None:
            assert set(keep_planes) <= set(
                self.keep_planes
            ), f"requested planes are not stored in data, at initialization, you only loaded: {self.keep_planes}"

        pairIdx = np.full(self.numPairs, True)
        if npixCutoff is not None:
            # remove pairs from index if they don't pass the cutoff
            pairIdx &= self.npixPair1 > npixCutoff
            pairIdx &= self.npixPair2 > npixCutoff
        if keep_planes is not None:
            pairIdx &= np.any(np.stack([self.planePair1 == pidx for pidx in keep_planes]), axis=0)
            pairIdx &= np.any(np.stack([self.planePair2 == pidx for pidx in keep_planes]), axis=0)
        if corrCutoff is not None:
            pairIdx &= self.xcROIs > corrCutoff
        if distanceCutoff is not None:
            pairIdx &= self.pwDist < distanceCutoff
        if extraFilter is not None:
            pairIdx &= extraFilter
        return pairIdx

    def filterPairs(self, pairIdx):
        """Filter ROI pair data based on boolean index.

        Parameters
        ----------
        pairIdx : numpy.ndarray
            Boolean array indicating which pairs to keep

        Returns
        -------
        tuple
            Filtered versions of all pair measurements
        """
        xcROIs = self.xcROIs[pairIdx]
        pwDist = self.pwDist[pairIdx]
        planePair1, planePair2 = self.planePair1[pairIdx], self.planePair2[pairIdx]
        npixPair1, npixPair2 = self.npixPair1[pairIdx], self.npixPair2[pairIdx]
        xposPair1, xposPair2 = self.xposPair1[pairIdx], self.xposPair2[pairIdx]
        yposPair1, yposPair2 = self.yposPair1[pairIdx], self.yposPair2[pairIdx]
        return (
            xcROIs,
            pwDist,
            planePair1,
            planePair2,
            npixPair1,
            npixPair2,
            xposPair1,
            xposPair1,
            yposPair1,
            yposPair2,
        )

    def scatterForThresholds(self, keep_planes=None, distanceCutoff=None, outputFig=False):
        """Create scatter plot visualizing correlation vs distance relationships.

        Parameters
        ----------
        keep_planes : list, optional
            List of plane indices to include
        distanceCutoff : float, optional
            Maximum distance between ROI pairs in μm
        outputFig : bool, optional
            Whether to return the figure object (default: False)

        Returns
        -------
        matplotlib.figure.Figure, optional
            Figure object if outputFig is True
        """

        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(keep_planes=keep_planes, distanceCutoff=distanceCutoff)

        randomFilter = pairIdx & (np.random.random(self.xcROIs.shape) < 0.05)
        xcROIs = self.xcROIs[randomFilter]
        planePair1 = self.planePair1[randomFilter]
        planePair2 = self.planePair2[randomFilter]
        pwDist = self.pwDist[randomFilter]

        # Plane relationship categories
        idxSamePlane = planePair1 == planePair2
        idxNeighbor = np.abs(planePair1 - planePair2) == 1
        idxDistant = np.abs(planePair1 - planePair2) > 1

        # Put them in an iterable list
        idxCategory = [idxSamePlane, idxNeighbor, idxDistant]
        nameCategory = ["same plane", "neighbor", "distant"]

        # Plotting parameters
        maxDist = max(pwDist)
        xmin, xmax = 0, maxDist
        ymin, ymax = -1, 1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Make figures
        plt.close("all")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for idx, (a, idxCat, name) in enumerate(zip(ax, idxCategory, nameCategory)):
            a.scatter(pwDist[idxCat], xcROIs[idxCat], c="k", alpha=0.2)
            a.set_title(f"{name}")

        plt.show()

        if outputFig:
            return fig

    def cdfForThresholds(
        self,
        cdfVals=np.linspace(0, 1, 11),
        ylim=None,
        keep_planes=None,
        distanceCutoff=50,
        corrCutoff=None,
        distanceDistant=(50, 250),
        withSave=False,
        withShow=True,
    ):
        """Make color-coded cumulative distribution plots to visualize correlation thresholds.

        Creates plots comparing correlation distributions between ROIs at different distances
        and across different planes.

        Parameters
        ----------
        cdfVals : numpy.ndarray, optional
            Values at which to evaluate CDF (default: np.linspace(0, 1, 11))
        ylim : tuple, optional
            Y-axis limits for the plot
        keep_planes : list, optional
            List of plane indices to include
        distanceCutoff : float, optional
            Maximum distance for "close" ROI pairs in μm (default: 50)
        corrCutoff : float, optional
            Minimum correlation threshold
        distanceDistant : tuple, optional
            (min, max) distance range for "distant" ROI pairs in μm (default: (50, 250))
        withSave : bool, optional
            Whether to save the figure (default: False)
        withShow : bool, optional
            Whether to display the figure (default: True)
        """
        assert type(distanceDistant) == tuple and len(distanceDistant) == 2, "distanceDistant should be a tuple specifying the range"

        # filter pairs based on optional cutoffs and plane indices (and more...)
        closeIdx = self.getPairFilter(keep_planes=keep_planes, distanceCutoff=distanceCutoff, corrCutoff=corrCutoff)
        farIdx = self.getPairFilter(
            keep_planes=keep_planes,
            distanceCutoff=distanceDistant[1],
            corrCutoff=corrCutoff,
            extraFilter=(self.pwDist > distanceDistant[0]),
        )

        def makeCDF(self, idx):
            xcROIs = self.xcROIs[idx]
            planePair1 = self.planePair1[idx]
            planePair2 = self.planePair2[idx]
            totalPairs = len(xcROIs)

            out = np.zeros(len(cdfVals))
            for ii, cval in enumerate(cdfVals):
                out[ii] = np.sum(xcROIs < cval) / totalPairs

            return out

        nameDistance = [
            f"{distanceDistant[0]} < distance < {distanceDistant[1]}",
            f"distance < {distanceCutoff}",
        ]
        colorDistance = ["k", "b"]
        cdfs = [makeCDF(self, farIdx), makeCDF(self, closeIdx)]

        # Make figures
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for name, color, data in zip(nameDistance, colorDistance, cdfs):
            ax[0].plot(cdfVals, data, c=color, marker=".", label=name)
        if ylim is not None:
            ax[0].set_ylim(ylim[0], ylim[1])
        ax[0].set_xlabel("Correlation Coefficient of Pair")
        ax[0].set_ylabel("CDF")
        ax[0].set_title("Full CDF")
        ax[0].legend(loc="lower right")

        for name, color, data in zip(nameDistance, colorDistance, cdfs):
            ax[1].plot(cdfVals, data, c=color, marker=".", label=name)
        ax[1].set_xlim(0.2, 1)
        ax[1].set_ylim(0.98, 1)
        ax[1].set_xlabel("Correlation Coefficient of Pair")
        ax[1].set_ylabel("CDF")
        ax[1].set_title("Zoom in")
        ax[1].legend(loc="lower right")

        # Save figure if requested
        if withSave:
            self.saveFigure(fig.number, "cdfCorrelations")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def roiCountHandling(
        self, roiCountCutoffs=np.linspace(0, 1, 11), maxBinConnections=25, keep_planes=None, distanceCutoff=40, withSave=False, withShow=True
    ):
        """Analyze statistics about ROI removal and connection graphs.

        Measures and visualizes how many ROIs are removed under different correlation
        thresholds and analyzes properties of the connection graph.

        Parameters
        ----------
        roiCountCutoffs : numpy.ndarray, optional
            Correlation thresholds to test (default: np.linspace(0, 1, 11))
        maxBinConnections : int, optional
            Maximum number of connections to show in histogram (default: 25)
        keep_planes : list, optional
            List of plane indices to include
        distanceCutoff : float, optional
            Maximum distance between ROI pairs in μm (default: 40)
        withSave : bool, optional
            Whether to save the figure (default: False)
        withShow : bool, optional
            Whether to display the figure (default: True)

        Notes
        -----
        This method uses the Maximal Independent Set (MIS) algorithm which can be
        computationally intensive for large numbers of cutoffs.
        """
        if len(roiCountCutoffs) > 11:
            print(
                f"Note: number of roiCountCutoffs is {len(roiCountCutoffs)}"
                "this could lead to an extremely long processing time due to the MIS algorithm!"
            )

        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(keep_planes=keep_planes, distanceCutoff=distanceCutoff)
        xcROIs = self.xcROIs[pairIdx]
        idxRoi1 = self.idxRoi1[pairIdx]
        idxRoi2 = self.idxRoi2[pairIdx]
        numROIs = len(set(np.concatenate((idxRoi1, idxRoi2))))

        def removeCount_simple(self, cutoff):
            # Count removed by removing any ROI in a pair
            idxMatches = xcROIs > cutoff  # True if pair is matched
            t = time.time()
            m1 = idxRoi1[idxMatches]
            m2 = idxRoi2[idxMatches]
            return len(set(np.concatenate((m1, m2)))), time.time() - t

        def removeCount_MIS(self, cutoff):
            # Count removed by removing only ROIs necessary to make a (stochastic) maximal independent set
            idxMatches = xcROIs > cutoff  # True if pair is matched
            t = time.time()
            removeFull = np.full(len(pairIdx), False)  # Start with every pair being valid
            removeFull[pairIdx] = idxMatches  # assign matched ROIs to their proper location in the full list
            adjacencyMatrix = sp.spatial.distance.squareform(1 * removeFull)  # convert pair boolean to adjacency matrix (1 if connected)
            N = adjacencyMatrix.shape[0]
            assert N == self.numROIs, "oops!"
            graph = nx.from_numpy_array(adjacencyMatrix)
            numInMIS = len(nx.maximal_independent_set(graph))
            return N - numInMIS, time.time() - t

        def countConnections(self, cutoff):
            idxMatches = xcROIs > cutoff  # True if pair is matched
            fullPairs = np.full(len(self.xcROIs), False)
            fullPairs[pairIdx] = idxMatches  # set any bad pairs to True
            G = sp.spatial.distance.squareform(fullPairs)
            firstOrder = np.sum(G, axis=1)  # number of first order connections per ROI
            return firstOrder

        filterNames = [f"rmv>{rcc}" for rcc in roiCountCutoffs]
        roiCounts = [removeCount_simple(self, cutoff) for cutoff in roiCountCutoffs]
        roiCountTime = [rc[1] for rc in roiCounts]
        roiCounts = [rc[0] for rc in roiCounts]

        misCounts = [removeCount_MIS(self, cutoff) for cutoff in roiCountCutoffs]
        misCountTime = [mc[1] for mc in misCounts]
        misCounts = [mc[0] for mc in misCounts]

        filtConnCutoffs = np.round(np.linspace(0.3, 0.9, 7), 1)
        filtConnNames = [f"corr>{fcc}" for fcc in filtConnCutoffs]
        roiConnections = [countConnections(self, cutoff) for cutoff in filtConnCutoffs]
        maxNumConnections = max([max(rc) for rc in roiConnections])
        binCounts = [np.unique(rc, return_counts=True) for rc in roiConnections]

        plt.close("all")
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(
            roiCountCutoffs,
            roiCounts,
            c="k",
            marker=".",
            markersize=12,
            label="aggressive removal",
        )
        ax[0].plot(roiCountCutoffs, misCounts, c="b", marker=".", markersize=12, label="MIS removal")
        ax[0].axhline(y=numROIs, c="k", linestyle="--", label="Total ROIs")
        ax[0].set_xlabel("Correlation Cutoff")
        ax[0].set_ylabel("ROIs removed")
        ax[0].set_title(f"# ROIs Removed (dist < {distanceCutoff}um)")
        # ax[0].set_yscale('log')
        ax[0].legend()

        ax[1].plot(
            roiCountCutoffs,
            roiCountTime,
            c="k",
            marker=".",
            markersize=12,
            label="aggressive removal",
        )
        ax[1].plot(roiCountCutoffs, misCountTime, c="b", marker=".", markersize=12, label="MIS removal")
        ax[1].set_xlabel("Correlation Cutoff")
        ax[1].set_ylabel("Time for removal (s)")
        ax[1].set_title("Speed of algorithm")
        ax[1].legend()

        for name, (bins, counts) in zip(filtConnNames, binCounts):
            ax[2].plot(bins, counts, marker=".", markersize=12, label=name)
        ax[2].set_xlabel("Number Pairs Per ROI")
        ax[2].set_ylabel("Counts")
        ax[2].set_xlim(0, maxBinConnections)
        ax[2].set_title('Num "connected" ROIs')
        ax[2].set_yscale("log")
        ax[2].legend()

        # Save figure if requested
        if withSave:
            self.saveFigure(fig.number, "roiCountStats")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def clusterSize(
        self,
        corrCutoffs=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        distanceCutoff=30,
        minDistance=None,
        keep_planes=[1, 2, 3, 4],
        verbose=True,
        withSave=False,
        withShow=True,
    ):
        """Analyze and plot cluster size distributions under different correlation thresholds.

        Parameters
        ----------
        corrCutoffs : list, optional
            List of correlation thresholds to analyze (default: [0.3-0.9])
        distanceCutoff : float, optional
            Maximum distance between ROI pairs in μm (default: 30)
        minDistance : float, optional
            Minimum distance between ROI pairs in μm
        keep_planes : list, optional
            List of plane indices to analyze (default: [1, 2, 3, 4])
        verbose : bool, optional
            Whether to print progress updates (default: True)
        withSave : bool, optional
            Whether to save the figure (default: False)
        withShow : bool, optional
            Whether to display the figure (default: True)
        """
        numCutoffs = len(corrCutoffs)
        extraFilter = self.pwdist > minDistance if minDistance is not None else None
        planeIdx = self.getPairFilter(keep_planes=keep_planes)  # just plane filter for pulling out the relevant pairs
        pairIdx = self.getPairFilter(distanceCutoff=distanceCutoff, keep_planes=keep_planes, extraFilter=extraFilter)
        connBins, connCounts = [], []
        corrBins, corrCounts = [], []
        for i, cc in enumerate(corrCutoffs):
            if verbose:
                print(f"Measuring cluster & correlation distribution for cutoff {round(cc,2)} : {i+1}/{numCutoffs}")
            cPairs = 1 * (pairIdx[planeIdx] & (self.xcROIs[planeIdx] > cc))  # connected after filtering for requested planes
            cAdjMat = sp.spatial.distance.squareform(cPairs)  # adjacency matrix of connected pairs
            cGroup = getConnectedGroups(cAdjMat)  # list of connected groups

            # get distribution of sizes of connected group
            bins, counts = np.unique([len(ccg) for ccg in cGroup], return_counts=True)
            connBins.append(list(bins))
            connCounts.append(list(counts))

            # get distribution of sizes of correlations per ROI
            bins, counts = np.unique(np.sum(cAdjMat, axis=1), return_counts=True)
            corrBins.append(list(bins))
            corrCounts.append(list(counts))

        # method for unifying bins for common sense plotting
        def getCommonBinCounts(bins, counts):
            plotBins = sorted(list(set({}).union(*[set(b) for b in bins])))
            plotCounts = [np.zeros_like(plotBins) for _ in range(len(counts))]
            for i, (hb, hc) in enumerate(zip(bins, counts)):
                for b, c in zip(hb, hc):
                    plotCounts[i][plotBins.index(b)] += c
            return plotBins, plotCounts

        # get common reference bins and counts
        plotConnBins, plotConnCounts = getCommonBinCounts(connBins, connCounts)
        plotCorrBins, plotCorrCounts = getCommonBinCounts(corrBins, corrCounts)

        # plot results
        plt.close("all")
        cmap = mpl.colormaps["jet"].resampled(numCutoffs)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        for i, (pConn, pCorr) in enumerate(zip(plotConnCounts, plotCorrCounts)):
            ax[0].plot(plotConnBins, pConn, color=cmap(i), marker=".", label=f"corr > {corrCutoffs[i]}")
            ax[1].plot(plotCorrBins, pCorr, color=cmap(i), marker=".", label=f"corr > {corrCutoffs[i]}")
        ax[0].set_xlabel("Size Connected Group")
        ax[1].set_xlabel("Number Correlated ROIs")
        ax[0].set_ylabel("Counts")
        ax[1].set_ylabel("Counts")
        ax[0].set_title("Size connected groups")
        ax[1].set_title("Number correlated ROIs")
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        # Save figure if requested
        if withSave:
            self.saveFigure(fig.number, "connectionCorrelationStats")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def planePairHistograms(
        self, corrCutoff=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], distanceCutoff=None, withSave=False, withShow=True, returnFigure=False
    ):
        """Create histograms of ROI pairs across planes meeting correlation thresholds.

        Parameters
        ----------
        corrCutoff : list or float, optional
            Single correlation threshold or list of thresholds (default: [0.25-0.85])
        distanceCutoff : float, optional
            Maximum distance between ROI pairs in μm
        withSave : bool, optional
            Whether to save the figure (default: False)
        withShow : bool, optional
            Whether to display the figure (default: True)
        returnFigure : bool, optional
            Whether to return the figure (default: False)
        """
        if len(corrCutoff) > 1:
            minCutoff = min(corrCutoff)
            corrCutoff = sorted(corrCutoff)  # make sure it goes from smallest to highest cutoff
        else:
            if not (type(corrCutoff) == list):
                corrCutoff = [corrCutoff]
            minCutoff = corrCutoff[0]

        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(corrCutoff=minCutoff, distanceCutoff=distanceCutoff)  # no filtering at the moment

        # get full list of possible plane/plane pair names
        ppStr = [str(int(p1)) + str(int(p2)) for (p1, p2) in zip(self.planePair1[pairIdx], self.planePair2[pairIdx])]
        ppUniq = np.unique(ppStr)

        ppCounts = []
        for cc in corrCutoff:
            # get idx of current correlation cutoff
            cidx = self.xcROIs > cc
            # make string pair name for planePair indices within this cutoff
            cppstr = np.array([str(int(p1)) + str(int(p2)) for (p1, p2) in zip(self.planePair1[pairIdx & cidx], self.planePair2[pairIdx & cidx])])
            # append counts to list
            ppCounts.append(np.array([sum(cppstr == puniq) for puniq in ppUniq]))

        # Create colormap for each cutoff
        cmap = helpers.ncmap("plasma", 0, len(corrCutoff) - 1)

        # Make plot
        plt.close("all")
        fig = plt.figure()
        for idx, ppc in enumerate(ppCounts):
            plt.bar(
                x=range(len(ppUniq)),
                height=ppc,
                color=cmap(idx),
                tick_label=ppUniq,
                label=f"Corr > {corrCutoff[idx]}",
            )
        plt.xlabel("Plane Indices of Pair")
        plt.ylabel("Counts")
        title = "Pairs exceeding correlation"
        if distanceCutoff is not None:
            title += f" (distance < {distanceCutoff})"
        plt.title(title)
        plt.legend(loc="best")
        plt.rcParams.update({"font.size": 12})

        # Save figure if requested
        if withSave:
            self.saveFigure(fig.number, "planePairHistogram")

        # Show figure if requested
        if withShow:
            plt.show()
        else:
            if not returnFigure:
                plt.close()

        if returnFigure:
            return fig

    def distanceDistribution(
        self, corrCutoffs=np.linspace(0.2, 0.6, 5), maxDistance=200, normalize="counts", keep_planes=[1, 2, 3, 4], withSave=False, withShow=True
    ):
        """Analyze and plot the distribution of distances between correlated ROIs.

        Parameters
        ----------
        corrCutoffs : numpy.ndarray, optional
            Correlation thresholds to analyze (default: np.linspace(0.2, 0.6, 5))
        maxDistance : float, optional
            Maximum distance to include in analysis in μm (default: 200)
        normalize : str, optional
            Type of normalization to apply: "counts", "relative", or "probability" (default: "counts")
        keep_planes : list, optional
            List of plane indices to analyze (default: [1, 2, 3, 4])
        withSave : bool, optional
            Whether to save the figure (default: False)
        withShow : bool, optional
            Whether to display the figure (default: True)
        """
        planeIdx = self.getPairFilter(keep_planes=keep_planes, distanceCutoff=maxDistance)
        corrIdx = [planeIdx & (self.xcROIs > cc) for cc in corrCutoffs]
        corrDistanceDistribution = [self.pwDist[cidx] for cidx in corrIdx]
        maxDistance = max([np.max(cdd) for cdd in corrDistanceDistribution])
        bins = np.arange(0, maxDistance + 1, 2) - 0.5
        centers = helpers.edge2center(bins)
        counts = [np.histogram(cdd, bins=bins)[0] for cdd in corrDistanceDistribution]

        samePlaneIdx = self.planePair1 == self.planePair2
        planeIdx = self.getPairFilter(keep_planes=keep_planes, distanceCutoff=maxDistance, extraFilter=samePlaneIdx)
        corrIdx = [planeIdx & (self.xcROIs > cc) for cc in corrCutoffs]
        corrDistanceDistribution = [self.pwDist[cidx] for cidx in corrIdx]
        maxDistance = max([np.max(cdd) for cdd in corrDistanceDistribution])
        samePlane_counts = [np.histogram(cdd, bins=bins)[0] for cdd in corrDistanceDistribution]

        if normalize == "relative":
            counts = [c / np.max(c) for c in counts]
            samePlane_counts = [c / np.max(c) for c in samePlane_counts]
            ylabel = "relative counts"
        elif normalize == "probability":
            counts = [c / np.sum(c) for c in counts]
            samePlane_counts = [c / np.sum(c) for c in samePlane_counts]
            ylabel = "probability"
        elif normalize == "counts":
            ylabel = "counts"
        else:
            raise ValueError("value of normalize not recognized")

        cmap = mpl.colormaps["jet"].resampled(len(corrCutoffs))
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        for i, (cc, count, samePlane_count) in enumerate(zip(corrCutoffs, counts, samePlane_counts)):
            ax[0].plot(centers, count, color=cmap(i), label=f"corr>{round(cc,1)}")
            ax[1].plot(centers, samePlane_count, color=cmap(i), label=f"corr>{round(cc,1)}")
        ax[0].set_xlabel("Distance (um)")
        ax[0].set_ylabel(ylabel)
        ax[0].set_title("Distance Given Correlation")
        ax[0].legend(loc="upper right")

        ax[1].set_xlabel("Distance (um)")
        ax[1].set_ylabel(ylabel)
        ax[1].set_title("Same Plane Pairs")
        ax[1].legend(loc="upper right")

        if withSave:
            self.saveFigure(fig.number, "distanceDistributionGivenCorrelation")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def makeHistograms(
        self,
        thresholds=[40, 10, 5, 3, 1],
        ncorrbins=51,
        withShow=True,
        withSave=False,
        npixCutoff=None,
        keep_planes=None,
    ):
        """Makes histograms of the correlation coefficients between ROIs within plane or across all planes, filtering for xy - distance"""
        binEdges = np.linspace(-1, 1, ncorrbins)
        binCenters = helpers.edge2center(binEdges)
        barWidth = np.diff(binEdges[:2])

        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(npixCutoff=npixCutoff, keep_planes=keep_planes)
        (
            xcROIs,
            pwDist,
            planePair1,
            planePair2,
            npixPair1,
            npixPair2,
            xposPair1,
            xposPair1,
            yposPair1,
            yposPair2,
        ) = self.filterPairs(pairIdx)

        # same plane information
        samePlaneIdx = planePair1 == planePair2  # boolean for same plane pairs
        planePair = planePair1 * samePlaneIdx  # plane for same plane pairs
        planePair[~samePlaneIdx] = -1  # set plane to -1 for any pairs not in the same plane

        # Get index of ROIs within a certain distance from each other
        idxClose = [pwDist < th for th in thresholds]

        # Do it for all data
        fullCounts = np.histogram(xcROIs, bins=binEdges)[0]
        closeCounts = [np.histogram(xcROIs[ic], bins=binEdges)[0] for ic in idxClose]

        # Then do it for each plane individually
        fcPlane = np.stack([np.histogram(xcROIs[planePair == planeIdx], bins=binEdges)[0] for planeIdx in self.vrexp.value["planeIDs"]])
        ccPlane = [
            np.stack([np.histogram(xcROIs[(planePair == planeIdx) & ic], bins=binEdges)[0] for planeIdx in self.vrexp.value["planeIDs"]])
            for ic in idxClose
        ]

        # now plot data
        plt.close("all")
        cmap = helpers.ncmap("winter", 0, len(thresholds) - 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Plot histograms for full count
        ax[0].bar(binCenters, fullCounts, width=barWidth, color="k", alpha=1, label="full distribution")
        for idx, counts in enumerate(closeCounts):
            ax[0].bar(
                binCenters,
                counts,
                width=barWidth,
                color=cmap(idx),
                alpha=0.4,
                label=f"Threshold: {thresholds[idx]}",
            )
        ax[0].set_yscale("log")
        ax[0].set_xlabel("correlation")
        ax[0].set_ylabel("counts")
        ax[0].legend(loc="upper left")
        ax[0].set_title(f"All pairs across planes - {self.onefile}")

        ax[1].bar(
            binCenters,
            np.nanmean(fcPlane, axis=0),
            width=barWidth,
            alpha=1,
            color="k",
            label="full (each plane)",
        )
        for idx, counts in enumerate(ccPlane):
            ax[1].bar(
                binCenters,
                np.nanmean(counts, axis=0),
                width=barWidth,
                alpha=0.4,
                color=cmap(idx),
                label=f"Threshold: {thresholds[idx]}",
            )
        ax[1].set_yscale("log")
        ax[1].set_xlabel("correlation")
        ax[1].set_ylabel("counts")
        ax[1].legend(loc="upper left")
        ax[1].set_title("Average within-plane distribution")

        # Save figure if requested
        if withSave:
            self.saveFigure(fig.number, self.onefile)

        # Show figure if requested
        plt.show() if withShow else plt.close()


class clusterExplorer(sameCellCandidates):
    """Interactive visualization tool for exploring ROI clusters based on correlation and distance.

    This class provides an interactive matplotlib interface for exploring relationships
    between ROIs, allowing users to visualize:
    - ROI activity traces
    - Neuropil signals
    - Spatial relationships between ROIs
    - Cluster memberships

    Parameters
    ----------
    scc : sameCellCandidates
        Parent sameCellCandidates object containing the analysis data
    maxCluster : int, optional
        Maximum number of ROIs to display in a cluster (default: 25)
    corrCutoff : float, optional
        Minimum correlation threshold for including ROI pairs (default: 0.4)
    maxCutoff : float, optional
        Maximum correlation threshold (default: None)
    distanceCutoff : float, optional
        Maximum distance between ROI pairs in μm (default: 20)
    minDistance : float, optional
        Minimum distance between ROI pairs in μm (default: None)
    keep_planes : list, optional
        List of plane indices to analyze (default: [1, 2, 3, 4])
    activity : str, optional
        Name of activity measure to use (default: "mpci.roiActivityF")
    """

    def __init__(
        self,
        scc,
        maxCluster=25,
        corrCutoff=0.4,
        maxCutoff=None,
        distanceCutoff=20,
        minDistance=None,
        keep_planes=[1, 2, 3, 4],
        activity="mpci.roiActivityF",
    ):
        for att, val in vars(scc).items():
            setattr(self, att, val)

        self.maxCluster = maxCluster
        self.default_alpha = 0.8
        self.default_linewidth = 1

        # Load activity and suite2p data
        self.timestamps = self.vrexp.loadone("mpci.times")
        self.activity = self.vrexp.loadone(activity)[:, self.idxROI_inTargetPlane]
        self.neuropil = self.vrexp.loadone("mpci.roiNeuropilActivityF")[:, self.idxROI_inTargetPlane]
        self.stat = self.vrexp.loadS2P("stat")[self.idxROI_inTargetPlane]
        self.roiCentroid = self.vrexp.loadone("mpciROIs.stackPosition")[self.idxROI_inTargetPlane, :2]
        self.roiPlaneIdx = self.vrexp.loadone("mpciROIs.stackPosition")[self.idxROI_inTargetPlane, 2].astype(np.int32)

        # Create look up table for plane colormap
        if keep_planes is not None:
            assert set(keep_planes) <= set(self.keep_planes), "requested planes include some not stored in sameCellCandidate object"
        roiPlanes = np.unique(self.roiPlaneIdx) if keep_planes is None else copy(keep_planes)
        numPlanes = len(roiPlanes)
        self.planeColormap = mpl.colormaps.get_cmap("jet").resampled(numPlanes)
        self.planeToCmap = lambda plane: plane if keep_planes is None else keep_planes.index(plane)

        # Create extrafilter if requested
        if maxCutoff is not None or minDistance is not None:
            extraFilter = np.full(len(self.xcROIs), True)
            if maxCutoff is not None:
                extraFilter &= self.xcROIs < maxCutoff
            if minDistance is not None:
                extraFilter &= self.pwDist > minDistance
        else:
            extraFilter = None

        # generate pair filter and return list of ROIs in clusters based on correlation, distance, and plane
        self.boolIdx = self.getPairFilter(
            corrCutoff=corrCutoff,
            distanceCutoff=distanceCutoff,
            keep_planes=keep_planes,
            extraFilter=extraFilter,
        )
        self.allROIs = list(set(self.idxRoi1[self.boolIdx]).union(set(self.idxRoi2[self.boolIdx])))
        numPairs = len(self.allROIs)

        # create figure
        plt.close("all")
        self.ax = []
        self.fig = plt.figure(figsize=(14, 4))  # , layout='constrained')
        self.ax.append(self.fig.add_subplot(1, 4, 1))
        self.ax.append(self.fig.add_subplot(1, 4, 2, sharex=self.ax[0]))
        self.ax.append(self.fig.add_subplot(1, 4, 3, sharex=self.ax[0]))
        self.ax.append(self.fig.add_subplot(1, 4, 4))

        self.fig.subplots_adjust(bottom=0.25)

        self.axIdx = self.fig.add_axes([0.25, 0.1, 0.65, 0.05])
        self.axText = self.fig.add_axes([0.05, 0.1, 0.1, 0.05])

        # define the values to use for snapping
        allowedPairs = np.arange(numPairs)

        # create the sliders
        initIdx = 0
        self.sSeed = Slider(
            self.axIdx,
            "ROI Seed",
            0,
            len(allowedPairs) - 1,
            valinit=initIdx,
            valstep=allowedPairs,
            color="black",
        )

        self.text_box = TextBox(self.axText, "ROI", textalignment="center")
        self.text_box.set_val("0")  # Trigger `submit` with the initial string.

        # initialize plot objects
        self.dLine = []
        self.nLine = []
        self.roiHull = []
        for n in range(self.maxCluster):
            self.dLine.append(
                self.ax[0].plot(
                    self.timestamps,
                    self.activity[:, 0],
                    lw=self.default_linewidth,
                    c="k",
                    alpha=self.default_alpha,
                )[0]
            )
            self.nLine.append(
                self.ax[1].plot(
                    self.timestamps,
                    self.neuropil[:, 0],
                    lw=self.default_linewidth,
                    c="k",
                    alpha=self.default_alpha,
                )[0]
            )
            hull = self.getMinMaxHull(n)
            self.roiHull.append(self.ax[3].plot(hull[0], hull[1], c=self.planeColormap(hull[2]), lw=self.default_linewidth)[0])
        self.im = self.ax[2].imshow(
            self.activity[:, :10].T,
            vmin=0,
            vmax=1,
            extent=(self.timestamps[0], self.timestamps[-1], 0, 1),
            aspect="auto",
            interpolation="nearest",
            cmap="hot",
            origin="lower",
        )

        cb = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=self.planeColormap),
            ax=self.ax[3],
            label="Plane Idx",
            boundaries=np.arange(numPlanes + 1) - 0.5,
            values=range(numPlanes),
        )
        cb.ax.set_yticks(range(numPlanes))
        cb.ax.set_yticklabels(roiPlanes)

        self.title1 = self.ax[0].set_title("activity")
        self.title2 = self.ax[1].set_title("neuropil")
        self.title3 = self.ax[2].set_title("full cluster activity")
        self.title4 = self.ax[3].set_title("roi outlines")

        # then add real data to them
        self.updatePlot(initIdx)

        cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        kid = self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        self.sSeed.on_changed(self.changeSlider)
        self.text_box.on_submit(self.typeROI)

        plt.show()

    def changeSlider(self, event):
        """Handle slider value change events.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The slider change event
        """
        self.updatePlot(int(self.sSeed.val))

    def typeROI(self, event):
        """Handle ROI number text input events.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The text input event
        """
        valid = self.text_box.text.isnumeric()
        if valid:
            self.sSeed.set_val(int(self.text_box.text))
        else:
            self.text_box.set_val("not an int")

    def onkey(self, event):
        """Handle keyboard events for navigation.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The keyboard event. Left/right arrows change ROI selection.
        """
        if event.key == "right":
            self.sSeed.set_val(self.sSeed.val + 1)
        if event.key == "left":
            self.sSeed.set_val(self.sSeed.val - 1)

    def onclick(self, event):
        """Handle mouse click events for ROI selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The mouse click event
        """
        selected = False
        if event.inaxes == self.ax[2]:
            plotIndex = int(np.floor(event.ydata))
            roiSelection = self.idxToPlot[plotIndex]
            selected = True

        elif event.inaxes == self.ax[3]:
            # color based on click position
            return None

        else:
            self.title4.set_text("ROI Outlines")

        # first reset colors and alphas and zorder
        for i in range(self.numToPlot):
            self.dLine[i].set(color=self.plot_cmap(i), alpha=self.default_alpha, zorder=i)
            self.nLine[i].set(color=self.plot_cmap(i), alpha=self.default_alpha, zorder=i)
            self.roiHull[i].set(
                color=self.planeColormap(self.planeToCmap(self.hulls[i][2])),
                zorder=i,
                lw=self.default_linewidth,
            )

        if not selected:
            return

        # then make the new one black
        self.dLine[plotIndex].set(color="k", alpha=1, zorder=self.maxCluster + 10)
        self.nLine[plotIndex].set(color="k", alpha=1, zorder=self.maxCluster + 10)
        self.roiHull[plotIndex].set(color="k", zorder=self.maxCluster + 10, lw=self.default_linewidth * 2)
        roiIndex = self.idxToPlot[plotIndex]
        roiPlane = int(self.roiPlaneIdx[roiIndex])
        inPlaneIndex = self.inPlaneIndex(roiIndex)
        self.title4.set_text(f"ROI Index: {inPlaneIndex} Plane: {roiPlane}")

    def getMinMaxHull(self, idxroi):
        """Calculate the min-max hull of an ROI for visualization.

        Parameters
        ----------
        idxroi : int
            Index of the ROI

        Returns
        -------
        tuple
            (x_coordinates, y_coordinates, plane_index) of the hull
        """
        ypix = self.stat[idxroi]["ypix"]
        xpix = self.stat[idxroi]["xpix"]
        allx, invx = np.unique(xpix, return_inverse=True)
        miny, maxy = list(map(list, zip(*[(min(ypix[xpix == u]), max(ypix[xpix == u])) for u in allx])))
        xpoints = np.append(np.concatenate((allx, allx[::-1])), allx[0])
        ypoints = np.append(np.concatenate((miny, maxy[::-1])), miny[0])
        planeIdx = self.roiPlaneIdx[idxroi]
        return xpoints, ypoints, planeIdx

    def getConvexHull(self, idxroi):
        """Calculate the convex hull of an ROI for visualization.

        Parameters
        ----------
        idxroi : int
            Index of the ROI

        Returns
        -------
        tuple
            (x_coordinates, y_coordinates, plane_index) of the hull
        """
        roipix = np.stack((self.stat[idxroi]["ypix"], self.stat[idxroi]["xpix"])).T
        hull = sp.spatial.ConvexHull(roipix)
        xpoints, ypoints = (
            hull.points[[*hull.vertices, hull.vertices[0]], 1],
            hull.points[[*hull.vertices, hull.vertices[0]], 0],
        )
        planeIdx = self.roiPlaneIdx[idxroi]
        return xpoints, ypoints, planeIdx

    def getCluster(self, iSeed):
        """Get cluster of ROIs connected to the seed ROI.

        Parameters
        ----------
        iSeed : int
            Index of the seed ROI

        Returns
        -------
        tuple
            (activity_traces, neuropil_traces) for ROIs in the cluster
        """
        includesSeed = (self.idxRoi1 == iSeed) | (self.idxRoi2 == iSeed)
        iCluster = np.nonzero(self.boolIdx & includesSeed)[0]
        self.idxROIs = list(set(self.idxRoi1[iCluster]).union(set(self.idxRoi2[iCluster])))
        self.numInCluster = len(self.idxROIs)
        self.numToPlot = min(self.numInCluster, self.maxCluster)
        self.idxToPlot = sorted(np.random.choice(self.idxROIs, self.numToPlot, replace=False))
        self.hulls = [self.getMinMaxHull(i) for i in self.idxToPlot]
        return self.activity[:, self.idxToPlot].T, self.neuropil[:, self.idxToPlot].T

    def updatePlot(self, newIdx):
        """Update all plot elements for a new ROI selection.

        Parameters
        ----------
        newIdx : int
            Index of the newly selected ROI
        """
        dTraces, nTraces = self.getCluster(self.allROIs[newIdx])
        ydlim = np.min(dTraces), np.max(dTraces)
        ynlim = np.min(nTraces), np.max(nTraces)
        rhxlim = min([np.min(h[0]) for h in self.hulls]), max([np.max(h[0]) for h in self.hulls])
        rhylim = min([np.min(h[1]) for h in self.hulls]), max([np.max(h[1]) for h in self.hulls])
        rhxcenter = np.mean(rhxlim)
        rhycenter = np.mean(rhylim)
        rhrange = max([np.diff(rhxlim), np.diff(rhylim)])

        self.ax[0].set_ylim(ydlim[0], ydlim[1])
        self.ax[1].set_ylim(ynlim[0], ynlim[1])
        self.ax[3].set_xlim(rhxcenter - rhrange / 2, rhxcenter + rhrange / 2)  # rhxlim[0], rhxlim[1])
        self.ax[3].set_ylim(rhycenter - rhrange / 2, rhycenter + rhrange / 2)  # ylim[0], rhylim[1])
        self.plot_cmap = helpers.ncmap("plasma", self.numToPlot)
        for i, (dl, nl, rh, d, n, hull) in enumerate(zip(self.dLine, self.nLine, self.roiHull, dTraces, nTraces, self.hulls)):
            dl.set(ydata=d, color=self.plot_cmap(i), visible=True)
            nl.set(ydata=n, color=self.plot_cmap(i), visible=True)
            rh.set(
                xdata=hull[0],
                ydata=hull[1],
                color=self.planeColormap(self.planeToCmap(hull[2])),
                visible=True,
            )
        for i in range(len(dTraces), self.maxCluster):
            self.dLine[i].set(visible=False)
            self.nLine[i].set(visible=False)
            self.roiHull[i].set(visible=False)

        self.ax[3].invert_yaxis()
        newImshow = sp.signal.savgol_filter(dTraces / np.max(dTraces, axis=1, keepdims=True), 15, 1, axis=1)
        # newImshow = newImshow - np.mean(newImshow, axis=1, keepdims=True)
        self.im.set(data=newImshow, extent=(self.timestamps[0], self.timestamps[-1], 0, self.numToPlot))
        self.title1.set_text(f"Activity - idx:{newIdx}")
        self.title2.set_text(f"Neuropil - numInCluster:{self.numInCluster}")
        self.fig.canvas.draw_idle()

    def inPlaneIndex(self, roi):
        """Convert global ROI index to within-plane index.

        Parameters
        ----------
        roi : int
            Global ROI index

        Returns
        -------
        int
            ROI index within its imaging plane
        """
        idxToRoiPlane = self.keep_planes.index(self.roiPlaneIdx[roi])  # if first keepPlane is 1 and roiPlane is 1, returns 0
        return roi - sum(self.roiPerPlane[:idxToRoiPlane])


class clusterExplorerROICaT(sameCellCandidates):
    """Interactive visualization tool for exploring ROI clusters identified by ROICaT.

    Similar to clusterExplorer, but specifically designed to work with ROICaT cluster
    labels. ROICaT is a tool for identifying putative same-cell ROIs across imaging planes.

    Parameters
    ----------
    scc : sameCellCandidates
        Parent sameCellCandidates object containing the analysis data
    roicat_labels : array-like
        ROICaT cluster labels for each ROI
    maxCluster : int, optional
        Maximum number of ROIs to display in a cluster (default: 25)
    corrCutoff : float, optional
        Minimum correlation threshold for including ROI pairs (default: 0.4)
    maxCutoff : float, optional
        Maximum correlation threshold (default: None)
    distanceCutoff : float, optional
        Maximum distance between ROI pairs in μm (default: 20)
    minDistance : float, optional
        Minimum distance between ROI pairs in μm (default: None)
    keep_planes : list, optional
        List of plane indices to analyze (default: [1, 2, 3, 4])
    activity : str, optional
        Name of activity measure to use (default: "mpci.roiActivityF")
    """

    def __init__(
        self,
        scc,
        roicat_labels,
        maxCluster=25,
        corrCutoff=0.4,
        maxCutoff=None,
        distanceCutoff=20,
        minDistance=None,
        keep_planes=[1, 2, 3, 4],
        activity="mpci.roiActivityF",
    ):
        for att, val in vars(scc).items():
            setattr(self, att, val)
        self.maxCluster = maxCluster
        self.default_alpha = 0.8
        self.default_linewidth = 1

        # Load ROICaT labels
        self.roicat_labels = roicat_labels[self.idxROI_inTargetPlane]
        self.roicat_clusters = np.unique(self.roicat_labels[self.roicat_labels >= 0])

        # Load activity and suite2p data
        self.timestamps = self.vrexp.loadone("mpci.times")
        self.activity = self.vrexp.loadone(activity)[:, self.idxROI_inTargetPlane]
        self.neuropil = self.vrexp.loadone("mpci.roiNeuropilActivityF")[:, self.idxROI_inTargetPlane]
        self.stat = self.vrexp.loadS2P("stat")[self.idxROI_inTargetPlane]
        self.roiCentroid = self.vrexp.loadone("mpciROIs.stackPosition")[self.idxROI_inTargetPlane, :2]
        self.roiPlaneIdx = self.vrexp.loadone("mpciROIs.stackPosition")[self.idxROI_inTargetPlane, 2].astype(np.int32)

        # Create look up table for plane colormap
        if keep_planes is not None:
            assert set(keep_planes) <= set(self.keep_planes), "requested planes include some not stored in sameCellCandidate object"
        roiPlanes = np.unique(self.roiPlaneIdx) if keep_planes is None else copy(keep_planes)
        numPlanes = len(roiPlanes)
        self.planeColormap = mpl.colormaps.get_cmap("jet").resampled(numPlanes)
        self.planeToCmap = lambda plane: plane if keep_planes is None else keep_planes.index(plane)

        # create figure
        plt.close("all")
        self.ax = []
        self.fig = plt.figure(figsize=(14, 4))  # , layout='constrained')
        self.ax.append(self.fig.add_subplot(1, 4, 1))
        self.ax.append(self.fig.add_subplot(1, 4, 2, sharex=self.ax[0]))
        self.ax.append(self.fig.add_subplot(1, 4, 3, sharex=self.ax[0]))
        self.ax.append(self.fig.add_subplot(1, 4, 4))

        self.fig.subplots_adjust(bottom=0.25)

        self.axIdx = self.fig.add_axes([0.25, 0.1, 0.65, 0.05])
        self.axText = self.fig.add_axes([0.05, 0.1, 0.1, 0.05])

        # define the values to use for snapping
        allowedPairs = copy(self.roicat_clusters)

        # create the sliders
        initIdx = 0
        self.sSeed = Slider(
            self.axIdx,
            "ROI Seed",
            0,
            len(allowedPairs) - 1,
            valinit=initIdx,
            valstep=allowedPairs,
            color="black",
        )

        self.text_box = TextBox(self.axText, "ROI", textalignment="center")
        self.text_box.set_val("0")  # Trigger `submit` with the initial string.

        # initialize plot objects
        self.dLine = []
        self.nLine = []
        self.roiHull = []
        for n in range(self.maxCluster):
            self.dLine.append(
                self.ax[0].plot(
                    self.timestamps,
                    self.activity[:, 0],
                    lw=self.default_linewidth,
                    c="k",
                    alpha=self.default_alpha,
                )[0]
            )
            self.nLine.append(
                self.ax[1].plot(
                    self.timestamps,
                    self.neuropil[:, 0],
                    lw=self.default_linewidth,
                    c="k",
                    alpha=self.default_alpha,
                )[0]
            )
            hull = self.getMinMaxHull(n)
            self.roiHull.append(self.ax[3].plot(hull[0], hull[1], c=self.planeColormap(hull[2]), lw=self.default_linewidth)[0])
        self.im = self.ax[2].imshow(
            self.activity[:, :10].T,
            vmin=0,
            vmax=1,
            extent=(self.timestamps[0], self.timestamps[-1], 0, 1),
            aspect="auto",
            interpolation="nearest",
            cmap="hot",
            origin="lower",
        )

        cb = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=self.planeColormap),
            ax=self.ax[3],
            label="Plane Idx",
            boundaries=np.arange(numPlanes + 1) - 0.5,
            values=range(numPlanes),
        )
        cb.ax.set_yticks(range(numPlanes))
        cb.ax.set_yticklabels(roiPlanes)

        self.title1 = self.ax[0].set_title("activity")
        self.title2 = self.ax[1].set_title("neuropil")
        self.title3 = self.ax[2].set_title("full cluster activity")
        self.title4 = self.ax[3].set_title("roi outlines")

        # then add real data to them
        self.updatePlot(initIdx)

        cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        kid = self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        self.sSeed.on_changed(self.changeSlider)
        self.text_box.on_submit(self.typeROI)

        plt.show()

    def changeSlider(self, event):
        """Handle slider value change events.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The slider change event
        """
        self.updatePlot(int(self.sSeed.val))

    def typeROI(self, event):
        """Handle ROI number text input events.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The text input event
        """
        valid = self.text_box.text.isnumeric()
        if valid:
            self.sSeed.set_val(int(self.text_box.text))
        else:
            self.text_box.set_val("not an int")

    def onkey(self, event):
        """Handle keyboard events for navigation.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The keyboard event. Left/right arrows change ROI selection.
        """
        if event.key == "right":
            self.sSeed.set_val(self.sSeed.val + 1)
        if event.key == "left":
            self.sSeed.set_val(self.sSeed.val - 1)

    def onclick(self, event):
        """Handle mouse click events for ROI selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The mouse click event
        """
        selected = False
        if event.inaxes == self.ax[2]:
            plotIndex = int(np.floor(event.ydata))
            roiSelection = self.idxToPlot[plotIndex]
            selected = True

        elif event.inaxes == self.ax[3]:
            # color based on click position
            return None

        else:
            self.title4.set_text("ROI Outlines")

        # first reset colors and alphas and zorder
        for i in range(self.numToPlot):
            self.dLine[i].set(color=self.plot_cmap(i), alpha=self.default_alpha, zorder=i)
            self.nLine[i].set(color=self.plot_cmap(i), alpha=self.default_alpha, zorder=i)
            self.roiHull[i].set(
                color=self.planeColormap(self.planeToCmap(self.hulls[i][2])),
                zorder=i,
                lw=self.default_linewidth,
            )

        if not selected:
            return

        # then make the new one black
        self.dLine[plotIndex].set(color="k", alpha=1, zorder=self.maxCluster + 10)
        self.nLine[plotIndex].set(color="k", alpha=1, zorder=self.maxCluster + 10)
        self.roiHull[plotIndex].set(color="k", zorder=self.maxCluster + 10, lw=self.default_linewidth * 2)
        roiIndex = self.idxToPlot[plotIndex]
        roiPlane = int(self.roiPlaneIdx[roiIndex])
        inPlaneIndex = self.inPlaneIndex(roiIndex)
        self.title4.set_text(f"ROI Index: {inPlaneIndex} Plane: {roiPlane}")

    def getMinMaxHull(self, idxroi):
        """Calculate the min-max hull of an ROI for visualization.

        Parameters
        ----------
        idxroi : int
            Index of the ROI

        Returns
        -------
        tuple
            (x_coordinates, y_coordinates, plane_index) of the hull
        """
        ypix = self.stat[idxroi]["ypix"]
        xpix = self.stat[idxroi]["xpix"]
        allx, invx = np.unique(xpix, return_inverse=True)
        miny, maxy = list(map(list, zip(*[(min(ypix[xpix == u]), max(ypix[xpix == u])) for u in allx])))
        xpoints = np.append(np.concatenate((allx, allx[::-1])), allx[0])
        ypoints = np.append(np.concatenate((miny, maxy[::-1])), miny[0])
        planeIdx = self.roiPlaneIdx[idxroi]
        return xpoints, ypoints, planeIdx

    def getConvexHull(self, idxroi):
        """Calculate the convex hull of an ROI for visualization.

        Parameters
        ----------
        idxroi : int
            Index of the ROI

        Returns
        -------
        tuple
            (x_coordinates, y_coordinates, plane_index) of the hull
        """
        roipix = np.stack((self.stat[idxroi]["ypix"], self.stat[idxroi]["xpix"])).T
        hull = sp.spatial.ConvexHull(roipix)
        xpoints, ypoints = (
            hull.points[[*hull.vertices, hull.vertices[0]], 1],
            hull.points[[*hull.vertices, hull.vertices[0]], 0],
        )
        planeIdx = self.roiPlaneIdx[idxroi]
        return xpoints, ypoints, planeIdx

    def getCluster(self, iSeed):
        """Get cluster of ROIs connected to the seed ROI.

        Parameters
        ----------
        iSeed : int
            Index of the seed ROI

        Returns
        -------
        tuple
            (activity_traces, neuropil_traces) for ROIs in the cluster
        """
        iCluster = self.roicat_labels == iSeed
        self.idxROIs = np.where(iCluster)[0]
        self.numInCluster = np.sum(iCluster)
        assert self.numInCluster > 1, "found cluster with 1 ROI, this shouldn't happen"
        self.numToPlot = min(self.numInCluster, self.maxCluster)
        self.idxToPlot = copy(self.idxROIs)
        self.hulls = [self.getMinMaxHull(i) for i in self.idxToPlot]
        return self.activity[:, self.idxToPlot].T, self.neuropil[:, self.idxToPlot].T

    def updatePlot(self, newIdx):
        """Update all plot elements for a new ROI selection.

        Parameters
        ----------
        newIdx : int
            Index of the newly selected ROI
        """
        dTraces, nTraces = self.getCluster(newIdx)
        ydlim = np.min(dTraces), np.max(dTraces)
        ynlim = np.min(nTraces), np.max(nTraces)
        rhxlim = min([np.min(h[0]) for h in self.hulls]), max([np.max(h[0]) for h in self.hulls])
        rhylim = min([np.min(h[1]) for h in self.hulls]), max([np.max(h[1]) for h in self.hulls])
        rhxcenter = np.mean(rhxlim)
        rhycenter = np.mean(rhylim)
        rhrange = max([np.diff(rhxlim), np.diff(rhylim)])

        self.ax[0].set_ylim(ydlim[0], ydlim[1])
        self.ax[1].set_ylim(ynlim[0], ynlim[1])
        self.ax[3].set_xlim(rhxcenter - rhrange / 2, rhxcenter + rhrange / 2)  # rhxlim[0], rhxlim[1])
        self.ax[3].set_ylim(rhycenter - rhrange / 2, rhycenter + rhrange / 2)  # ylim[0], rhylim[1])
        self.plot_cmap = helpers.ncmap("plasma", self.numToPlot)
        for i, (dl, nl, rh, d, n, hull) in enumerate(zip(self.dLine, self.nLine, self.roiHull, dTraces, nTraces, self.hulls)):
            dl.set(ydata=d, color=self.plot_cmap(i), visible=True)
            nl.set(ydata=n, color=self.plot_cmap(i), visible=True)
            rh.set(
                xdata=hull[0],
                ydata=hull[1],
                color=self.planeColormap(self.planeToCmap(hull[2])),
                visible=True,
            )
        for i in range(len(dTraces), self.maxCluster):
            self.dLine[i].set(visible=False)
            self.nLine[i].set(visible=False)
            self.roiHull[i].set(visible=False)

        self.ax[3].invert_yaxis()
        newImshow = sp.signal.savgol_filter(dTraces / np.max(dTraces, axis=1, keepdims=True), 15, 1, axis=1)
        # newImshow = newImshow - np.mean(newImshow, axis=1, keepdims=True)
        self.im.set(data=newImshow, extent=(self.timestamps[0], self.timestamps[-1], 0, self.numToPlot))
        self.title1.set_text(f"Activity - idx:{newIdx}")
        self.title2.set_text(f"Neuropil - numInCluster:{self.numInCluster}")
        self.fig.canvas.draw_idle()

    def inPlaneIndex(self, roi):
        """Convert global ROI index to within-plane index.

        Parameters
        ----------
        roi : int
            Global ROI index

        Returns
        -------
        int
            ROI index within its imaging plane
        """
        idxToRoiPlane = self.keep_planes.index(self.roiPlaneIdx[roi])  # if first keepPlane is 1 and roiPlane is 1, returns 0
        return roi - sum(self.roiPerPlane[:idxToRoiPlane])
