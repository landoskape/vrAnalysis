import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

from .. import session
from .. import helpers
from .. import fileManagement as fm
from .standardAnalysis import standardAnalysis

class piezoConsistency(standardAnalysis):
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
        self.name = 'piezoConsistency'
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
        fig = plt.figure()
        for ii,pt in enumerate(planeTimes):
            plt.axvline(x=pt, c=cmap(ii), label=f"plane{ii}")
        plt.axvline(x=1000*cycleSamples*dt, c='k')
        helpers.errorPlot(timeCycle*1000, fullCycle, axis=0, color='b', alpha=0.5, label='position')
        plt.plot(timeCycle*1000, np.mean(fullCommand,axis=0), color='k', linestyle='--', label='command')
        plt.xlabel('Time (ms)')
        plt.ylabel('Piezo (au)')
        plt.title('Piezo Over 1 Cycle')
        plt.legend(loc='lower right')
        
        # Save figure if requested
        if withSave: 
            self.saveFigure(self, fig.number, 'piezoConsistency')
        
        # Show figure if requested
        plt.show() if withShow else plt.close()
        
        return timeCycle, fullCycle
