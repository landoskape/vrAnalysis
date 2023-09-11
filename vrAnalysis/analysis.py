import numpy as np
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

from . import helpers
from . import fileManagement as fm

# common variables
analysisDirectory = fm.analysisPath()

class sameCellCandidates:
    '''Measures cross-correlation of pairs of ROI activity with spatial distance
    
    Takes as required input a vrexp object. Optional inputs define parameters of analysis, 
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).
    '''
    def __init__(self, vrexp, thresholds=[40, 10, 5, 3, 1], ncorrbins=51, onefile='mpci.roiActivityDeconvolvedOasis', autorun=True):
        self.thresholds = thresholds
        self.ncorrbins = ncorrbins
        self.onefile = onefile
        self.vrexp = vrexp
        
        self.binEdges = np.linspace(-1, 1, self.ncorrbins)
        self.binCenters = helpers.edge2center(self.binEdges)
        self.barWidth = np.diff(self.binEdges[:2])
        
        # automatically do measurements
        if autorun: self.run()
    
    def run(self, onefile=None, npixCutoff=25, keepPlanes=None):
        self.onefile = self.onefile if onefile is None else onefile
        
        npix = np.array([s['npix'] for s in self.vrexp.loadS2P('stat')]) # get number of pixels for each ROI
        idxNPIX = npix > npixCutoff
        
        # Analyze cross-plane possibilities
        data = self.vrexp.loadone(self.onefile)
        stackPosition = self.vrexp.loadone('mpciROIs.stackPosition')
        roiPlaneIdx = stackPosition[:,2]
        xyPos = stackPosition[:,0:2] * 1.3 # convert to um (assume it's always the same at B2...)

        if keepPlanes is not None:
            idxKeepPlanes = np.isin(roiPlaneIdx, keepPlanes)
        else:
            idxKeepPlanes = np.full(idxNPIX.shape, True)
        
        # Filter ROIs 
        idxROIsToUse = idxKeepPlanes & idxNPIX
        
        data = data[:, idxROIsToUse]
        roiPlaneIdx = roiPlaneIdx[idxROIsToUse]
        xyPos = xyPos[idxROIsToUse]
        
        # Get correlation coefficients
        xcROIs = np.corrcoef(data.T)

        # Get index of plane if in same plane (-1 otherwise)
        samePlaneIdx=(roiPlaneIdx.reshape(-1,1)==roiPlaneIdx.reshape(1,-1))
        planePair = roiPlaneIdx * samePlaneIdx
        planePair[~samePlaneIdx]=-1

        # Convert to vector representation
        xcPairs = sp.spatial.distance.squareform(xcROIs, checks=False)
        spPairs = sp.spatial.distance.squareform(planePair, checks=False)

        # Measure spatial distance between ROI centroids
        pwDistance = sp.spatial.distance.pdist(xyPos)

        # Get index of ROIs within a certain distance from each other
        idxClose = [pwDistance < th for th in self.thresholds]

        # Do it for all data
        self.fullCounts = np.histogram(xcPairs, bins=self.binEdges)[0]
        self.closeCounts = [np.histogram(xcPairs[ic], bins=self.binEdges)[0] for ic in idxClose]

        # Then do it for each plane individually
        self.fcPlane = np.stack([np.histogram(xcPairs[spPairs==planeIdx], bins=self.binEdges)[0] for planeIdx in self.vrexp.value['planeIDs']])
        self.ccPlane = [np.stack([np.histogram(xcPairs[(spPairs==planeIdx) & ic], bins=self.binEdges)[0] for planeIdx in self.vrexp.value['planeIDs']]) for ic in idxClose]
    
    def somaDendritePairs(self, onefile=None, corrCutoff=0.1, npixCutoff=25):
        self.onefile = self.onefile if onefile is None else onefile
        
        # Analyze cross-plane possibilities
        data = self.vrexp.loadone(self.onefile)
        stackPosition = self.vrexp.loadone('mpciROIs.stackPosition')
        npix = np.array([s['npix'] for s in self.vrexp.loadS2P('stat')]) # get number of pixels for each ROI
        roiPlaneIdx = stackPosition[:,2]
        xyPos = stackPosition[:,0:2] * 1.3 # convert to um (assume it's always the same at B2...)

        # Get correlation coefficients
        xcROIs = np.corrcoef(data.T)
        
        planeDifference = roiPlaneIdx.reshape(-1,1) - roiPlaneIdx.reshape(1,-1)
        roiSizeDifference = npix.reshape(-1,1) - npix.reshape(1,-1)
        
        # Convert to vector representation
        xcPairs = sp.spatial.distance.squareform(xcROIs, checks=False)
        pdiffPairs = sp.spatial.distance.squareform(planeDifference, checks=False)
        rsdiffPairs = sp.spatial.distance.squareform(roiSizeDifference, checks=False)
        
        # Measure spatial distance between ROI centroids
        pwDistance = sp.spatial.distance.pdist(xyPos)
        
        # filter to pairs that exceed a certain correlation coefficient
        idxCorrelated = xcPairs > corrCutoff 
        xcCorr = xcPairs[idxCorrelated]
        planeDiffCorr = pdiffPairs[idxCorrelated]
        roiSizeDiffCorr = rsdiffPairs[idxCorrelated]
        
        # And plot some results
        plt.close('all')
        
        fig,ax = plt.subplots(1,4,figsize=(16,4))
        ax[0].scatter(planeDiffCorr, xcCorr, c='k', alpha=0.3)
        ax[0].set_xlabel('planediff')
        ax[0].set_ylabel('cross-corr')
        
        ax[1].scatter(planeDiffCorr, roiSizeDiffCorr, c='k', alpha=0.3)
        ax[1].set_xlabel('planediff')
        ax[1].set_ylabel('roiSize diff')
        
        ax[2].scatter(roiSizeDiffCorr, xcCorr, c='k', alpha=0.3)
        ax[2].set_xlabel('roiSize diff')
        ax[2].set_ylabel('cross-corr')
        
        plt.show()
        
    def plotSession(self, withSave=False):
        cmap = helpers.ncmap('winter', 0, len(self.thresholds)-1)
        fig,ax = plt.subplots(1,2,figsize=(12,4))
        
        # Plot histograms for full count
        ax[0].bar(self.binCenters, self.fullCounts, width=self.barWidth, color='k', alpha=1, label='full distribution')
        for idx, counts in enumerate(self.closeCounts):
            ax[0].bar(self.binCenters, counts, width=self.barWidth, color=cmap(idx), alpha=0.4, label=f"Threshold: {self.thresholds[idx]}")
        ax[0].set_yscale('log')
        ax[0].set_xlabel('correlation')
        ax[0].set_ylabel('counts')
        ax[0].legend(loc='upper left')
        ax[0].set_title(f"All pairs across planes - {self.onefile}")
        
        ax[1].bar(self.binCenters, np.nanmean(self.fcPlane,axis=0), width=self.barWidth, alpha=1, color='k', label='full (each plane)')
        for idx, counts in enumerate(self.ccPlane):
            ax[1].bar(self.binCenters, np.nanmean(counts,axis=0), width=self.barWidth, alpha=0.4, color=cmap(idx), label=f"Threshold: {self.thresholds[idx]}")
        ax[1].set_yscale('log')
        ax[1].set_xlabel('correlation')
        ax[1].set_ylabel('counts')
        ax[1].legend(loc='upper left')
        ax[1].set_title("Average within-plane distribution")
        
        if withSave:
            print(str(self.vrexp))
            plt.savefig(self.saveDirectory() / str(self.vrexp))
            
        return fig, ax
        
    def saveDirectory(self):
        # Define and create target directory
        dirName = analysisDirectory / 'sameCellCandidates' / self.onefile
        if not(dirName.is_dir()): dirName.mkdir(parents=True)
        return dirName
    
        
        
    
    