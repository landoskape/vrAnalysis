import time
import numpy as np
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

import pandas as pd

from . import session
from . import helpers
from . import fileManagement as fm

class sameCellCandidates(standardAnalysis):
    '''Measures cross-correlation of pairs of ROI activity with spatial distance
    
    Takes as required input a vrexp object. Optional inputs define parameters of analysis, 
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).
    
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
    scc.scatterForThresholds(keepPlanes=[1,2,3,4], distanceCutoff=250);
    
    The other makes a cumulative distribution plot for ROIs of two different distance ranges...
    
    Theory:
    -------
    This is used for producing the maximum independent set of nodes that are not the same cell. It's a nice graph theory problem.
    https://www.gcsu.edu/sites/files/page-assets/node-808/attachments/ballardmyer.pdf
    '''
    def __init__(self, vrexp, thresholds=[40, 10, 5, 3, 1], ncorrbins=51, onefile='mpci.roiActivityDeconvolvedOasis', autoload=True):
        self.name = 'sameCellCandidates'
        self.thresholds = thresholds
        self.ncorrbins = ncorrbins
        self.onefile = onefile
        self.vrexp = vrexp
        
        self.binEdges = np.linspace(-1, 1, self.ncorrbins)
        self.binCenters = helpers.edge2center(self.binEdges)
        self.barWidth = np.diff(self.binEdges[:2])
        
        # automatically do measurements
        self.dataloaded = False
        if autoload: self.autoload()
    
    def totalFromPairs(self, pairs):
        assert type(pairs)==int, "pairs is not an integer..."
        n = (1+np.sqrt(1+8*pairs))/2
        assert n.is_integer(), "pairs is not a valid binomial coefficient choose 2..."
        return int(n)
        
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
    
    def getPairFilter(self, npixCutoff=None, keepPlanes=None, corrCutoff=None, distanceCutoff=None, extraFilter=None):
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
        if extraFilter is not None:
            pairIdx &= extraFilter
        return pairIdx
    
    def filterPairs(self, pairIdx):
        xcROIs = self.xcROIs[pairIdx]
        pwDist = self.pwDist[pairIdx]
        planePair1, planePair2 = self.planePair1[pairIdx], self.planePair2[pairIdx]
        npixPair1, npixPair2 = self.npixPair1[pairIdx], self.npixPair2[pairIdx]
        xposPair1, xposPair2 = self.xposPair1[pairIdx], self.xposPair2[pairIdx]
        yposPair1, yposPair2 = self.yposPair1[pairIdx], self.yposPair2[pairIdx]
        return xcROIs, pwDist, planePair1, planePair2, npixPair1, npixPair2, xposPair1, xposPair1, yposPair1, yposPair2
    
    def autoload(self, onefile=None):
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
        self.idxRoi1, self.idxRoi2 = self.pairValFromVec(np.arange(self.numROIs), squareform=True)
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
        pairIdx = self.getPairFilter(keepPlanes=keepPlanes, distanceCutoff=distanceCutoff)
        
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
            a.scatter(pwDist[idxCat], xcROIs[idxCat], c='k', alpha=0.2)
            a.set_title(f"{name}")
            
        plt.show()
        
        if outputFig: return fig
    
    def cdfForThresholds(self, cdfVals=np.linspace(0,1,11), ylim=None, keepPlanes=None, distanceCutoff=50, corrCutoff=None, distanceDistant=(50, 250), withSave=False, withShow=True):
        '''Make color-coded cumulative distribution plots to visualize potential thresholds for distance and planes
        the cutoff inputs and keepPlanes input are all standard - they go into getPairFilter. 
        the distanceDistant input requires a tuple and determines the range of distances to use for the "distant" group
        the "close" group is based purely on 'distanceCutoff'
        This uses a fast approximation of the cdf with prespecified cdfVals, which should be an increasing linspace like array.
        '''
        assert type(distanceDistant)==tuple and len(distanceDistant)==2, "distanceDistant should be a tuple specifying the range"
        
        # filter pairs based on optional cutoffs and plane indices (and more...)
        closeIdx = self.getPairFilter(keepPlanes=keepPlanes, distanceCutoff=distanceCutoff, corrCutoff=corrCutoff)
        farIdx = self.getPairFilter(keepPlanes=keepPlanes, distanceCutoff=distanceDistant[1], corrCutoff=corrCutoff, extraFilter=(self.pwDist>distanceDistant[0])) 
        
        def makeCDF(self, idx):            
            xcROIs = self.xcROIs[idx]
            planePair1 = self.planePair1[idx]
            planePair2 = self.planePair2[idx]
            totalPairs = len(xcROIs) 
            
            out = np.zeros(len(cdfVals))
            for ii, cval in enumerate(cdfVals):
                out[ii] = np.sum(xcROIs < cval) / totalPairs

            return out
        
        nameDistance = [f"{distanceDistant[0]} < distance < {distanceDistant[1]}", f"distance < {distanceCutoff}"]
        colorDistance = ['k', 'b']
        cdfs = [makeCDF(self, farIdx), makeCDF(self, closeIdx)]
        
        # Make figures
        plt.close('all')
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        for name, color, data in zip(nameDistance, colorDistance, cdfs):
            ax[0].plot(cdfVals, data, c=color, marker='.', label=name)
        if ylim is not None:
            ax[0].set_ylim(ylim[0], ylim[1])
        ax[0].set_xlabel("Correlation Coefficient of Pair")
        ax[0].set_ylabel("CDF")
        ax[0].set_title("Full CDF")
        ax[0].legend(loc='lower right')
            
        for name, color, data in zip(nameDistance, colorDistance, cdfs):
            ax[1].plot(cdfVals, data, c=color, marker='.', label=name)
        ax[1].set_xlim(0.2, 1)
        ax[1].set_ylim(0.98, 1)
        ax[1].set_xlabel("Correlation Coefficient of Pair")
        ax[1].set_ylabel("CDF")
        ax[1].set_title("Zoom in")
        ax[1].legend(loc='lower right')
        
        if withSave:
            print(f"Saving cdf plots for session: {self.vrexp.sessionPrint()}")
            plt.savefig(self.saveDirectory('cdfCorrelations') / str(self.vrexp))
        
        plt.show() if withShow else plt.close()
        
    
    def roiCountHandling(self, roiCountCutoffs=np.linspace(0, 1, 11), maxBinConnections=25, keepPlanes=None, distanceCutoff=40, withSave=False, withShow=True):
        '''measures statistics about how many ROIs are removed (and other things about the connection graph)'''
        
        if len(roiCountCutoffs)>11:
            print(f"Note: number of roiCountCutoffs is {len(roiCountCutoffs)}"
                  "this could lead to an extremely long processing time due to the MIS algorithm!")
        
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(keepPlanes=keepPlanes, distanceCutoff=distanceCutoff)
        xcROIs = self.xcROIs[pairIdx]
        idxRoi1 = self.idxRoi1[pairIdx]
        idxRoi2 = self.idxRoi2[pairIdx]
        numROIs = len(set(np.concatenate((idxRoi1, idxRoi2))))
        
        def removeCount_simple(self, cutoff):
            # Count removed by removing any ROI in a pair
            idxMatches = xcROIs > cutoff # True if pair is matched
            t = time.time()
            m1 = idxRoi1[idxMatches]
            m2 = idxRoi2[idxMatches]
            return len(set(np.concatenate((m1,m2)))), time.time()-t
        
        def removeCount_MIS(self, cutoff):
            # Count removed by removing only ROIs necessary to make a (stochastic) maximal independent set
            idxMatches = xcROIs > cutoff # True if pair is matched
            t = time.time()
            removeFull = np.full(len(pairIdx), False) # Start with every pair being valid
            removeFull[pairIdx]=idxMatches # assign matched ROIs to their proper location in the full list
            adjacencyMatrix = sp.spatial.distance.squareform(1*removeFull) # convert pair boolean to adjacency matrix (1 if connected)
            N = adjacencyMatrix.shape[0]
            assert N==self.numROIs, "oops!"
            graph = nx.from_numpy_array(adjacencyMatrix)
            numInMIS = len(nx.maximal_independent_set(graph)) 
            return N-numInMIS, time.time()-t
            
        def countConnections(self, cutoff):
            idxMatches = xcROIs > cutoff # True if pair is matched
            fullPairs = np.full(len(self.xcROIs), False)
            fullPairs[pairIdx] = idxMatches # set any bad pairs to True
            G = sp.spatial.distance.squareform(fullPairs)
            firstOrder = np.sum(G,axis=1) # number of first order connections per ROI
            return firstOrder
        
        filterNames = [f"rmv>{rcc}" for rcc in roiCountCutoffs]
        roiCounts = [removeCount_simple(self, cutoff) for cutoff in roiCountCutoffs]
        roiCountTime = [rc[1] for rc in roiCounts]
        roiCounts = [rc[0] for rc in roiCounts]
        
        misCounts = [removeCount_MIS(self, cutoff) for cutoff in roiCountCutoffs]
        misCountTime = [mc[1] for mc in misCounts]
        misCounts = [mc[0] for mc in misCounts]
        
        filtConnCutoffs = np.round(np.linspace(0.3,0.9,7),1)
        filtConnNames = [f"corr>{fcc}" for fcc in filtConnCutoffs]
        roiConnections = [countConnections(self, cutoff) for cutoff in filtConnCutoffs]
        maxNumConnections = max([max(rc) for rc in roiConnections])
        binCounts = [np.unique(rc, return_counts=True) for rc in roiConnections]
        
        plt.close('all')
        fig, ax = plt.subplots(1,3,figsize=(14,4))
        ax[0].plot(roiCountCutoffs, roiCounts, c='k', marker='.', markersize=12, label='aggressive removal')
        ax[0].plot(roiCountCutoffs, misCounts, c='b', marker='.', markersize=12, label='MIS removal')
        ax[0].axhline(y=numROIs, c='k', linestyle='--', label='Total ROIs')
        ax[0].set_xlabel('Correlation Cutoff')
        ax[0].set_ylabel('ROIs removed')
        ax[0].set_title(f'# ROIs Removed (dist < {distanceCutoff}um)')
        # ax[0].set_yscale('log')
        ax[0].legend()
        
        ax[1].plot(roiCountCutoffs, roiCountTime, c='k', marker='.', markersize=12, label='aggressive removal')
        ax[1].plot(roiCountCutoffs, misCountTime, c='b', marker='.', markersize=12, label='MIS removal')
        ax[1].set_xlabel('Correlation Cutoff')
        ax[1].set_ylabel('Time for removal (s)')
        ax[1].set_title('Speed of algorithm')
        ax[1].legend()
        
        for name, (bins, counts) in zip(filtConnNames, binCounts):
            ax[2].plot(bins, counts, marker='.', markersize=12, label=name)
        ax[2].set_xlabel('Number Pairs Per ROI')
        ax[2].set_ylabel('Counts')
        ax[2].set_xlim(0, maxBinConnections)
        ax[2].set_title('Num "connected" ROIs')
        ax[2].set_yscale('log')
        ax[2].legend()
        
        # Save figure if requested
        if withSave: 
            self.saveFigure(self, fig.number, 'roiCountStats')
        
        # Show figure if requested
        plt.show() if withShow else plt.close()
    
    def planePairHistograms(self, corrCutoff=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], distanceCutoff=None, withSave=False, withShow=True):
        '''Make histogram of number of pairs across specific planes meeting some correlation threshold'''
        
        if len(corrCutoff)>1:
            minCutoff = min(corrCutoff)
            corrCutoff = sorted(corrCutoff) # make sure it goes from smallest to highest cutoff
        else:
            if not(type(corrCutoff)==list): corrCutoff = [corrCutoff]
            minCutoff = corrCutoff[0]
            
        # filter pairs based on optional cutoffs and plane indices (and more...)
        pairIdx = self.getPairFilter(corrCutoff=minCutoff, distanceCutoff=distanceCutoff) # no filtering at the moment
        
        # get full list of possible plane/plane pair names
        ppStr = [str(int(p1))+str(int(p2)) for (p1,p2) in zip(self.planePair1[pairIdx], self.planePair2[pairIdx])]
        ppUniq = np.unique(ppStr) 
        
        ppCounts = []
        for cc in corrCutoff:
            # get idx of current correlation cutoff
            cidx = self.xcROIs > cc
            # make string pair name for planePair indices within this cutoff
            cppstr = np.array([str(int(p1))+str(int(p2)) for (p1,p2) in zip(self.planePair1[pairIdx & cidx], self.planePair2[pairIdx & cidx])])
            # append counts to list
            ppCounts.append(np.array([sum(cppstr==puniq) for puniq in ppUniq]))
        
        # Create colormap for each cutoff
        cmap = helpers.ncmap('plasma', 0, len(corrCutoff)-1)

        # Make plot
        fig = plt.figure()
        for idx, ppc in enumerate(ppCounts):
            plt.bar(x=range(len(ppUniq)), height=ppc, color=cmap(idx), tick_label=ppUniq, label=f"Corr > {corrCutoff[idx]}")
        plt.xlabel('Plane Indices of Pair')
        plt.ylabel('Counts')
        title = 'Pairs exceeding correlation'
        if distanceCutoff is not None:
            title += f' (distance < {distanceCutoff})'
        plt.title(title)
        plt.legend(loc='best')
        plt.rcParams.update({'font.size': 12})
        
        # Save figure if requested
        if withSave: 
            self.saveFigure(self, fig.number, 'planePairHistogram')
        
        # Show figure if requested
        plt.show() if withShow else plt.close()
        

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
        
        # Save figure if requested
        if withSave: 
            self.saveFigure(self, fig.number, f"correlationHistogram_{self.onefile}")
        
        # Show figure if requested
        plt.show() if withShow else plt.close()

        


class piezoConsistency:
    '''
    Analysis class for analyzing the piezo consistency
    
    Measures average +/- std piezo position (& command) and plots it with the plane times overlaid
    
    Standard usage: 
    ---------------
    vrdb = database.vrDatabase() # get database object
    for ses in vrdb.iterSessions(imaging=True): 
        # go through each session with imaging data and make a plot of the average piezo
        pconst = analysis.piezoConsistency(session.vrRegistration(ses))
        # you can save or show the plot as you wish
        pconst.plotAveragePiezo(withSave=True, withShow=False);
    
    # This prints the distribution of frame/plane timing, which needs to be consistent for this analysis to work
    pconst.checkFrameTiming(verbose=True)
    '''
    def __init__(self, vrreg):
        assert isinstance(vrreg, session.vrRegistration), "input must be a vrRegistration object"
        self.vrreg = vrreg
        self.timestamps = self.vrreg.getTimelineVar('timestamps')
        self.piezoCommand = self.vrreg.getTimelineVar('piezoCommand')
        self.piezoPosition = self.vrreg.getTimelineVar('piezoPosition')
        self.neuralFrames = self.vrreg.getTimelineVar('neuralFrames')
        self.changeFrames = np.append(0, np.diff(np.ceil(self.neuralFrames/len(self.vrreg.value['planeIDs']))))==1
        self.frameSamples = np.where(self.changeFrames)[0]
        self.changePlanes = np.append(0, np.diff(self.neuralFrames))==1
        self.planeSamples = np.where(self.changePlanes)[0]
    
    def checkFrameTiming(self, verbose=True, force=False):
        '''Reports the number of samples per frame and plane, and the number of instances
        If frame timing is consistent, then there should be only a few possibilities for samples per frame/plane, 
        and the instances of each number of samples should be well distributed.'''
        if force or not(hasattr(self, 'samplePerFrame') and hasattr(self, 'samplePerPlane')):
            self.samplePerFrame, self.sampleEachFrame, self.frameCounts = np.unique(np.diff(self.frameSamples), return_counts=True, return_inverse=True)
            self.samplePerPlane, self.sampleEachPlane, self.planeCounts = np.unique(np.diff(self.planeSamples), return_counts=True, return_inverse=True)
        if verbose:
            print(pd.DataFrame({'Samples Per Frame':self.samplePerFrame, 'Instances':self.frameCounts}))
            print(pd.DataFrame({'Samples Per Plane':self.samplePerPlane, 'Instances':self.planeCounts}))
        
    def plotAveragePiezo(self, extend=0.1, withSave=False, withShow=True):
        '''Make plot of average piezo command and position for each frame (and return data).
        Extend allows you to see before and after the frame by a proportion of the frame duration'''
        assert 0<=extend<1, "extend must be within [0,1)"
        if not(hasattr(self, 'samplePerFrame') and hasattr(self, 'samplePerPlane')):
            self.checkFrameTiming(verbose=False)
        cycleSamples = min(self.samplePerFrame) # number of samples to use for each frame cycle
        extendSamples = int(cycleSamples*extend) # number of samples to extend window before and after each frame cycle
        numCycles = len(self.frameSamples)-1 # number of cycles to plot
        # Create an index for each cycle
        idx = np.full(self.timestamps.shape, True) # initialize cycle index
        idx[:self.frameSamples[0]]=False # remove any samples before first frame
        idx[self.frameSamples[-1]:]=False # remove any samples after last frame
        # Then remove any samples at the end of a frame cycle that are longer than the minimum
        for fs, sef in zip(self.frameSamples, self.sampleEachFrame):
            idx[fs+cycleSamples:fs+self.samplePerFrame[sef]]=False
        # Now create a new index for the precycle component
        preidx = np.full(self.timestamps.shape, False) # 
        for fs in self.frameSamples[:-1]:
            preidx[fs-extendSamples:fs]=True
        # End by getting each full cycle in stack
        preCycle = self.piezoPosition[preidx].reshape(numCycles, extendSamples)
        eachCycle = self.piezoPosition[idx].reshape(numCycles, cycleSamples)
        postCycle = eachCycle[:,:extendSamples]
        fullCycle = helpers.scale(np.hstack((preCycle, eachCycle, postCycle)))
        
        preCommand = self.piezoCommand[preidx].reshape(numCycles, extendSamples)
        eachCommand = self.piezoCommand[idx].reshape(numCycles, cycleSamples)
        postCommand = eachCommand[:,:extendSamples]
        fullCommand = helpers.scale(np.hstack((preCommand, eachCommand, postCommand)))
        
        # And finally make time series for the cycle
        dt = np.median(np.diff(self.timestamps))
        timeCycle = np.arange(-extendSamples, cycleSamples+extendSamples) * dt
        planeTimes = 1000*np.linspace(0, cycleSamples*dt, len(self.vrreg.value['planeNames'])+1)[:-1]
        cmap = helpers.ncmap('brg',vmin=0,vmax=len(planeTimes)-1) 

        # plot it
        plt.close('all')
        for ii,pt in enumerate(planeTimes):
            plt.axvline(x=pt, c=cmap(ii), label=f"plane{ii}")
        plt.axvline(x=1000*cycleSamples*dt, c='k')
        helpers.errorPlot(timeCycle*1000, fullCycle, axis=0, color='b', alpha=0.5, label='position')
        plt.plot(timeCycle*1000, np.mean(fullCommand,axis=0), color='k', linestyle='--', label='command')
        plt.xlabel('Time (ms)')
        plt.ylabel('Piezo (au)')
        plt.title('Piezo Over 1 Cycle')
        plt.legend(loc='lower right')
        
        if withSave:
            print(f"Saving piezo figure for session: {self.vrreg.sessionPrint()}")
            plt.savefig(self.saveDirectory() / str(self.vrreg))
        
        # Show it if requested, otherwise keep pyplot clean by closing it
        plt.show() if withShow else plt.close()
            
        # return data
        return timeCycle, fullCycle
    
    def saveDirectory(self):
        # Define and create target directory
        dirName = analysisDirectory / 'piezoConsistency'
        if not(dirName.is_dir()): dirName.mkdir(parents=True)
        return dirName