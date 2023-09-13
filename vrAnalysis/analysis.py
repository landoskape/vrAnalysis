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
        self.dataloaded = False
        if autorun: self.run()
    
    def pairValFromVec(self, vector, squareform=True):
        '''Convert vector to pair values, optionally perform squareform without checks'''
        assert isinstance(vector, np.ndarray) and vector.ndim==1, "vector must be 1d numpy array"
        N = len(vector)
        p1 = vector.reshape(-1,1).repeat(N,axis=1)
        p2 = vector.reshape(1,-1).repeat(N,axis=0)
        if squareform:
            p1 = sp.spatial.distance.squareform(p1, checks=False)
            p2 = sp.spatial.distance.squareform(p2, checks=False)
        return p1, p2
    
    def getPairFilter(self, npixCutoff=None, keepPlanes=None, corrCutoff=None, distanceCutoff=None):
        assert self.dataloaded, "data is not loaded yet, use 'run()' to get key datapoints"
        pairIdx = np.full(self.numPairs, True)
        if npixCutoff is not None:
            # remove pairs from index if they don't pass the cutoff
            pairIdx &= self.npixPair1 > npixCutoff
            pairIdx &= self.npixPair2 > npixCutoff
        if keepPlanes is not None:
            pairIdx &= np.isin(self.planePair1, keepPlanes)
            pairIdx &= np.isin(self.planePair2, keepPlanes)
        if corrCutoff is not None:
            pairIdx &= self.xcROIs > corrCutoff
        if distanceCutoff is not None:
            pairIdx &= self.pwDist < distanceCutoff
        return pairIdx
    
    def filterPairs(self, pairIdx):
        xcROIs = self.xcROIs[pairIdx]
        pwDist = self.pwDist[pairIdx]
        planePair1, planePair2 = self.planePair1[pairIdx], self.planePair2[pairIdx]
        npixPair1, npixPair2 = self.npixPair1[pairIdx], self.npixPair2[pairIdx]
        xposPair1, xposPair2 = self.xposPair1[pairIdx], self.xposPair2[pairIdx]
        yposPair1, yposPair2 = self.yposPair1[pairIdx], self.yposPair2[pairIdx]
        return xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2
    
    def run(self, onefile=None):
        '''load standard data for measuring same cell candidate'''
        # update onefile if using a different measure of activity
        self.onefile = self.onefile if onefile is None else onefile
        self.numROIs = sum(self.vrexp.value['roiPerPlane'])
        
        # get relevant data
        npix = np.array([s['npix'] for s in self.vrexp.loadS2P('stat')]).astype(np.int32) # roi size (in pixels of mask)
        data = self.vrexp.loadone(self.onefile) # activity array
        stackPosition = self.vrexp.loadone('mpciROIs.stackPosition')
        roiPlaneIdx = stackPosition[:,2].astype(np.int32) # plane index
        xyPos = stackPosition[:,0:2] * 1.3 # xy position to um
        
        # comparisons
        self.xcROIs = sp.spatial.distance.squareform(np.corrcoef(data.T), checks=False)
        self.pwDist = sp.spatial.distance.pdist(xyPos)
        self.planePair1, self.planePair2 = self.pairValFromVec(roiPlaneIdx, squareform=True)
        self.npixPair1, self.npixPair2 = self.pairValFromVec(npix, squareform=True)
        self.xposPair1, self.xposPair2 = self.pairValFromVec(xyPos[:,0], squareform=True)
        self.yposPair1, self.yposPair2 = self.pairValFromVec(xyPos[:,1], squareform=True)
        self.numPairs = len(self.xcROIs)
        assert self.numPairs==self.numROIs*(self.numROIs-1)/2, f"math flex failed: numPairs={self.numPairs}, math: {self.numROIs*(self.numROIs-1)/2}"
        self.dataloaded = True
        
    def scatterForThresholds(self, keepPlanes=None, distanceCutoff=None, outputFig=False):
        '''Make color-coded scatter plot to visualize potential thresholds for distance and planes'''
        
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(keepPlanes=keepPlanes, distanceCutoff=distanceCutoff) # no filtering at the moment
        xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2 = self.filterPairs(pairIdx)
        
        print('hello')
        
        randomFilter = np.random.random(xcROIs.shape) < 0.05
        xcROIs = xcROIs[randomFilter]
        planePair1 = planePair1[randomFilter]
        planePair2 = planePair2[randomFilter]
        pwDist = pwDist[randomFilter]
        
        # These three categories define a color-code
        ccode = 'kbr' 
        idxSamePlane = planePair1 == planePair2
        idxNeighbor = np.abs(planePair1 - planePair2) == 1
        idxDistant = np.abs(planePair1 - planePair2) > 1
        
        # Put them in an iterable list
        idxCategory = [idxDistant, idxSamePlane, idxNeighbor] 
        nameCategory = ['same plane', 'neighbor', 'distant']
        
        # Plotting parameters
        maxDist = max(pwDist)
        xmin, xmax = 0, maxDist
        ymin, ymax = -1, 1
        xx,yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Make figures
        plt.close('all')
        fig, ax = plt.subplots(1,3,figsize=(13,4))
        for idx, (a, idxCat, name) in enumerate(zip(ax, idxCategory, nameCategory)):
            a.scatter(pwDist[idxCat], xcROIs[idxCat], c=ccode[idx], alpha=0.1)
            a.set_title(f"{name}")
            
        plt.show()
        
        if outputFig:
            return fig
        
    
    def planePairHistograms(self, corrCutoff=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]):
        '''Make histogram of number of pairs across specific planes meeting some correlation threshold'''
        
        if len(corrCutoff)>1:
            minCutoff = min(corrCutoff)
            corrCutoff = sorted(corrCutoff) # make sure it goes from smallest to highest cutoff
        else:
            if not(type(corrCutoff)==list): corrCutoff = [corrCutoff]
            minCutoff = corrCutoff[0]
            
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(corrCutoff=minCutoff) # no filtering at the moment
        xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2 = self.filterPairs(pairIdx)
        
        # get full list of possible plane/plane pair names
        ppStr = [str(int(p1))+str(int(p2)) for (p1,p2) in zip(planePair1, planePair2)]
        ppUniq = np.unique(ppStr) 
        
        ppCounts = []
        for cc in corrCutoff:
            # get idx of current correlation cutoff
            cidx = xcROIs > cc
            # make string pair name for planePair indices within this cutoff
            cppstr = np.array([str(int(p1))+str(int(p2)) for (p1,p2) in zip(planePair1[cidx], planePair2[cidx])])
            # append counts to list
            ppCounts.append(np.array([sum(cppstr==puniq) for puniq in ppUniq]))
        
        # Create colormap for each cutoff
        cmap = helpers.ncmap('plasma', 0, len(corrCutoff)-1)
        
        # # Get indices of pairs from specific combinations of planes
        # idx23 = (planePair1==2) & (planePair2==3)
        # idx34 = (planePair1==3) & (planePair2==4)
        # print(f"23 counts: {np.sum(idx23)}")
        # print(f"34 counts: {np.sum(idx34)}")
        # xc_23 = xcROIs[idx23]
        # xc_34 = xcROIs[idx34]
        # npix1High_23 = npixPair1[idx23]
        # npix2High_23 = npixPair2[idx23]
        # npix1High_34 = npixPair1[idx34]
        # npix2High_34 = npixPair2[idx34]
        
        fig = plt.figure()
        for idx, ppc in enumerate(ppCounts):
            plt.bar(x=range(len(ppUniq)), height=ppc, color=cmap(idx), tick_label=ppUniq, label=f"Corr > {corrCutoff[idx]}")
        plt.xlabel('Plane Indices of Pair')
        plt.ylabel('Counts')
        plt.title('Filtering for pair correlations')
        plt.legend()
        plt.show();

    def makeHistograms(self, thresholds=None, withSave=False, npixCutoff=None, keepPlanes=None):
        '''Makes histograms of the correlation coefficients between ROIs within plane or across all planes, filtering for xy - distance'''
        
        # default parameter
        thresholds = thresholds if thresholds is not None else self.thresholds
        
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(npixCutoff=npixCutoff, keepPlanes=keepPlanes)
        xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2 = self.filterPairs(pairIdx)
        
        # same plane information
        samePlaneIdx = planePair1 == planePair2 # boolean for same plane pairs
        planePair = planePair1 * samePlaneIdx # plane for same plane pairs
        planePair[~samePlaneIdx] = -1 # set plane to -1 for any pairs not in the same plane
        
        # Get index of ROIs within a certain distance from each other
        idxClose = [pwDist < th for th in self.thresholds]
        
        # Do it for all data
        fullCounts = np.histogram(xcROIs, bins=self.binEdges)[0]
        closeCounts = [np.histogram(xcROIs[ic], bins=self.binEdges)[0] for ic in idxClose]

        # Then do it for each plane individually
        fcPlane = np.stack([np.histogram(xcROIs[planePair==planeIdx], bins=self.binEdges)[0] for planeIdx in self.vrexp.value['planeIDs']])
        ccPlane = [np.stack([np.histogram(xcROIs[(planePair==planeIdx) & ic], bins=self.binEdges)[0] for planeIdx in self.vrexp.value['planeIDs']]) for ic in idxClose]
        
        # now plot data
        plt.close('all')
        cmap = helpers.ncmap('winter', 0, len(self.thresholds)-1)
        fig,ax = plt.subplots(1,2,figsize=(12,4))
        
        # Plot histograms for full count
        ax[0].bar(self.binCenters, fullCounts, width=self.barWidth, color='k', alpha=1, label='full distribution')
        for idx, counts in enumerate(closeCounts):
            ax[0].bar(self.binCenters, counts, width=self.barWidth, color=cmap(idx), alpha=0.4, label=f"Threshold: {thresholds[idx]}")
        ax[0].set_yscale('log')
        ax[0].set_xlabel('correlation')
        ax[0].set_ylabel('counts')
        ax[0].legend(loc='upper left')
        ax[0].set_title(f"All pairs across planes - {self.onefile}")
        
        ax[1].bar(self.binCenters, np.nanmean(fcPlane,axis=0), width=self.barWidth, alpha=1, color='k', label='full (each plane)')
        for idx, counts in enumerate(ccPlane):
            ax[1].bar(self.binCenters, np.nanmean(counts,axis=0), width=self.barWidth, alpha=0.4, color=cmap(idx), label=f"Threshold: {thresholds[idx]}")
        ax[1].set_yscale('log')
        ax[1].set_xlabel('correlation')
        ax[1].set_ylabel('counts')
        ax[1].legend(loc='upper left')
        ax[1].set_title("Average within-plane distribution")
        
        if withSave:
            print(f"Saving histogram figure for session: {self.vrexp.sessionPrint()}")
            plt.savefig(self.saveDirectory() / str(self.vrexp))
            
        return fig, ax
    
    
    def somaDendritePairs(self, corrCutoff=0.1, npixCutoff=25):
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(npixCutoff=npixCutoff, corrCutoff=corrCutoff)
        xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2 = self.filterPairs(pairIdx)
        
        planeDifference = planePair1 - planePair2
        sizeDifference = npixPair1 - npixPair2
        
        # And plot some results
        plt.close('all')
        
        fig,ax = plt.subplots(1,4,figsize=(16,4))
        ax[0].scatter(planeDifference, xcROIs, c='k', alpha=0.3)
        ax[0].set_xlabel('planediff')
        ax[0].set_ylabel('cross-corr')
        
        ax[1].scatter(planeDifference, sizeDifference, c='k', alpha=0.3)
        ax[1].set_xlabel('planediff')
        ax[1].set_ylabel('roiSize diff')
        
        ax[2].scatter(sizeDifference, xcROIs, c='k', alpha=0.3)
        ax[2].set_xlabel('roiSize diff')
        ax[2].set_ylabel('cross-corr')
        
        plt.show()
        
        return fig, ax
        
    def saveDirectory(self):
        # Define and create target directory
        dirName = analysisDirectory / 'sameCellCandidates' / self.onefile
        if not(dirName.is_dir()): dirName.mkdir(parents=True)
        return dirName
    
        
        
    
    