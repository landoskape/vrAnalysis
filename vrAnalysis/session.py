# inclusions
import json
import time
from tqdm import tqdm
from pathlib import Path 
import numpy as np
import scipy as sp
import scipy.io as scio
import numba as nb
from numpyencoder import NumpyEncoder
from . import functions
from . import helpers
from . import fileManagement as fm

# Variables that might need to be changed for different users
# if anyone other than me uses this, let me know and I can make it smarter by using a user dictionary or storing a file somewhere else...
dataPath = fm.localDataPath()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------- vrExperiment Management Object -----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class vrSession:
    name = 'vrSession'
    def __init__(self,mouseName,dateString,session):
        # initialize vrSession object -- this is a top level object which can be used to point to folders and identify basic aspects about the session. 
        # it can't do any real data handling
        self.mouseName = mouseName
        self.dateString = dateString
        self.session = session
        
    # path methods
    def dataPath(self):
        # dataPath returns a path object referring to the directory contain data structured in the Alyx format (./mouseName/dateString/session/*data*)
        # the actual dataPath string is defined in the top of this scripts - change this for your machine and keep data in the Alyx format :)
        return dataPath
    
    def sessionPath(self): return self.dataPath()/self.mouseName/self.dateString/self.session
    def onePath(self): return self.sessionPath()/'oneData'
    def rawDataPath(self): return self.sessionPath()/'rawDataPath'
    def suite2pPath(self): return self.sessionPath()/'suite2p'
    
    def getSavedOne(self):
        return self.onePath().glob('*.npy')
    
    def printSavedOne(self):
        # Return all names of one variables stored in this experiment's directory
        return [name.stem for name in self.getSavedOne()] 
    
    def clearOneData(self, oneFileNames=None, certainty=False):
        # clears any oneData in session folder
        # oneFileNames is an optional list of files to clear, otherwise it will clear all of them
        if not(certainty):
            print(f"You have to be certain to clear oneData! (This means set kwarg certainty=True).")
            return None
        oneFiles = self.getSavedOne()
        if oneFileNames: 
            oneFiles = [file for file in oneFiles if file.stem in oneFileNames]
        for file in oneFiles: 
            file.unlink()
        print(f"Cleared oneData from session: {self.sessionPrint()}")
        
    def sessionName(self):
        # function for returning mousename, datestring, and sessionid
        return self.mouseName, self.dateString, self.session
    
    def sessionPrint(self): 
        # useful function for generating string of session name for useful feedback to user 
        mouseName, dateString, sessionid = self.sessionName()
        return "/".join(self.sessionName())

    def __str__(self):
        # use _ to avoid string complications with backslash used in 'sessionPrint()'
        return '_'.join(self.sessionName())
    

class vrExperiment(vrSession):
    """
    The vrExperiment object is a postprocessing object used to load and buffer data, and store highly used methods for analyzing data in vr experiments.
    It accepts three required arguments as strings - the mouseName, dateString, and session, which it uses to identify the directory of the data (w/ Alyx conventions and a method specified local data path)
    vrExperiment is meant to be called after vrRegistration is already run. From the output of vrRegistration, vrExperiment can quickly and efficiently manage a recording session.
    """
    name = 'vrExperiment'
    def __init__(self,*inputs):
        # use createObject to create vrExperiment
        # requires the vrRegistration to have already been run for this session!
        self.createObject(*inputs)
    
    # ------------------------------------------------------- basic meta functions for vrExperiment -----------------------------------------------------------
    def createObject(self, *inputs):
        '''This is used to create a vrExperiment object from an existing object
        In the case of the vrExperiment class itself, it won't make sense to use (except for an alternative to copy.deepcopy())
        But this is very useful for:
        - vrRegistration: populate from existing allows the user to re-register things without starting from scratch
        - redCellProcessing: absolutely necessary because it is literally run during registration
        '''
        if len(inputs)==1:
            assert isinstance(inputs[0], vrExperiment), f"input is a {type(inputs[0])} but it should be a vrExperiment object (or a child thereof)"
            self.mouseName = inputs[0].mouseName
            self.dateString = inputs[0].dateString
            self.session = inputs[0].session
            self.opts = inputs[0].opts
            self.preprocessing = inputs[0].preprocessing
            self.value = inputs[0].value
            self.loadBuffer = inputs[0].loadBuffer
            
        elif len(inputs)==3:
            assert all([isinstance(ip,str) for ip in inputs]), "if three inputs are provided, they must be strings indicating the mouseName, date, and session"
            self.mouseName = inputs[0]
            self.dateString = inputs[1]
            self.session = inputs[2]
            assert self.sessionPath().exists(), "session folder does not exist!"
            self.loadRegisteredExperiment()
            self.loadBuffer = {}
            
        else:
            raise TypeError("input must be either a vrExperiment object or 3 strings indicating the mouseName, date, and session")
        
        # print(f"{self.name} object created for session {self.sessionPrint()} at path {self.sessionPath()}")
        
    def loadRegisteredExperiment(self):
        # load registered experiment -- including options (see below), list of completed preprocessing steps, and any useful values saved to the vrExp object
        assert (self.sessionPath()/'vrExperimentOptions.json').exists(), "session json files were not found! you need to register the session first."
        self.opts = json.load(open(self.sessionPath()/'vrExperimentOptions.json'))
        self.preprocessing = json.load(open(self.sessionPath()/'vrExperimentPreprocessing.json'))
        self.value = json.load(open(self.sessionPath()/'vrExperimentValues.json'))    
    
    def saveParams(self):
        # saveParams saves the parameters stored in vrExperiment objects, usually generated by vrRegistration.
        with open(self.sessionPath()/'vrExperimentOptions.json','w') as file:
            json.dump(self.opts, file, ensure_ascii=False)
        with open(self.sessionPath()/'vrExperimentPreprocessing.json','w') as file:
            json.dump(self.preprocessing, file, ensure_ascii=False)
        with open(self.sessionPath()/'vrExperimentValues.json','w') as file:
            json.dump(self.value, file, ensure_ascii=False, cls=NumpyEncoder)
    
    def registerValue(self,name,value):
        # add a variable to vrExperiment object (these are saved in JSON objects and reloaded each time the vrExp method is generated, best practice is to only add small variables) 
        self.value[name]=value
    
    # ----------------------------------------------------------------- one data handling --------------------------------------------------------------------
    def saveone(self, var, *names):
        # save variable as oneData (names can be an arbitrarily long list of strings, they'll be joined with '.' to make the filename
        # automatically adds variable to the loadBuffer for efficient data handling
        fileName = self.oneFilename(*names)
        self.loadBuffer[fileName] = var
        np.save(self.onePath() / fileName, var)

    def loadone(self, *names, force=False, allow_pickle=True):
        # load one data from vrexp object. if available in loadBuffer, will grab it from there. force=True performs automatic reload, even if already in buffer
        fileName = self.oneFilename(*names)
        if not force and fileName in self.loadBuffer.keys():
            return self.loadBuffer[fileName]
        else:
            if not (self.onePath()/fileName).exists(): 
                print(f"In session {self.sessionPrint()}, the one file {fileName} doesn't exist. Here is a list of saved oneData files:")
                for oneFile in self.printSavedOne(): print(oneFile)
                raise ValueError("oneData requested is not available")
            oneVar = np.load(self.onePath() / fileName, allow_pickle=allow_pickle)
            self.loadBuffer[fileName] = oneVar
            return oneVar

    def oneFilename(self, *names):
        # create one filename given an arbitrary length list of names
        return '.'.join(names)+'.npy'
    
    def clearBuffer(self, *names):
        # clear loadBuffer for data management
        if len(names)==0:
            self.loadBuffer = {}
        else:
            for name in names:
                if name in self.loadBuffer.keys(): 
                    del self.loadBuffer[name]
                if self.oneFilename(name) in self.loadBuffer.keys():
                    del self.loadBuffer[self.oneFilename(name)]
    
    # -------------------------------------------------------------- database communication --------------------------------------------------------------------
    def printSessionNotes(self, vrdb):
        """you must pass a vrdatabase object for this to work"""
        record = vrdb.getRecord(self.mouseName, self.dateString, self.session)
        print(record['sessionNotes'])
    
    # ------------------------------------------ special loading functions for data not stored directly in one format ------------------------------------------
    def loadfcorr(self,meanAdjusted=True,loadFromOne=True):
        # corrected fluorescence requires a special loading function because it isn't saved directly
        if loadFromOne:
            F = self.loadone('mpci.roiActivityF').T
            Fneu = self.loadone('mpci.roiNeuropilActivityF').T
        else:
            F = self.loadS2P('F')
            Fneu = self.loadS2P('Fneu')
        meanFneu = np.mean(Fneu,axis=1,keepdims=True) if meanAdjusted else np.zeros((np.sum(self.value['roiPerPlane']),1))
        return  F - self.opts['neuropilCoefficient']*(Fneu - meanFneu)
    
    def loadS2P(self,varName,concatenate=True,checkVariables=True):
        # load S2P variable from suite2p folders 
        assert varName in self.value['available'], f"{varName} is not available in the suite2p folders for {self.sessionPrint()}"
        if varName=='ops': 
            concatenate=False
            checkVariables=False
            
        var = [np.load(self.suite2pPath()/planeName/f"{varName}.npy",allow_pickle=True) for planeName in self.value['planeNames']]
        if varName=='ops':
            var = [cvar.item() for cvar in var]
            return var
        
        if checkVariables:
            # if check variables is on, then we check the variables for their shapes
            # checkVariables should be "off" for initial registration! (see standard usage in "processImaging")
            
            # first check if ROI variable (if #rows>0)
            isRoiVars = [self.isRoiVar(cvar) for cvar in var]
            assert all(isRoiVars) or not(any(isRoiVars)), f"For {varName}, the suite2p files from planes {[idx for idx,roivar in enumerate(isRoiVars) if roivar]} registered as ROI vars but not the others!"
            isFrameVars = [self.isFrameVar(cvar) for cvar in var]
            assert all(isFrameVars) or not(any(isFrameVars)), f"For {varName}, the suite2p files from planes {[idx for idx,roivar in enumerate(isRoiVars) if roivar]} registered as frame vars but not the others!"
            
            # Check valid shapes across all planes together, then provide complete error message
            if all(isRoiVars): 
                validShapes = [cvar.shape[0]==self.value['roiPerPlane'][planeIdx] for planeIdx,cvar in enumerate(var)]
                assert all(validShapes), f"{self.sessionPrint()}:{varName} has a different number of ROIs than registered in planes: {[pidx for pidx,vs in enumerate(validShapes) if not(vs)]}."
            if all(isFrameVars):
                validShapes = [cvar.shape[1]==self.value['framePerPlane'][planeIdx] for planeIdx,cvar in enumerate(var)]
                assert all(validShapes), f"{self.sessionPrint()}:{varName} has a different number of frames than registered in planes: {[pidx for pidx,vs in enumerate(validShapes) if not(vs)]}."
        
        if concatenate: 
            # if concatenation is requested, then concatenate each plane across the ROIs axis so we have just one ndarray of shape: (allROIs, allFrames)
            if self.isFrameVar(var[0]): var = [v[:,:self.value['numFrames']] for v in var] # trim if necesary so each plane has the same number of frames
            var = np.concatenate(var,axis=0)
            
        return var
    
    # shorthands -- note that these require some assumptions about suite2p variables to be met
    def isRoiVar(self, var): return var.ndim>0 # useful shorthand for determining if suite2p variable includes one element for each ROI
    def isFrameVar(self, var): return var.ndim>1 and var.shape[1]>2 # useful shorthand for determining if suite2p variable includes a column for each frame
    
    def getPlaneIdx(self):
        # produce np array of plane ID associated with each ROI (assuming that the data e.g. spks will be concatenated across planes) 
        return np.concatenate([np.repeat(planeIDs,roiPerPlane) for (planeIDs,roiPerPlane) in zip(self.value['planeIDs'],self.value['roiPerPlane'])]).astype(np.uint8)
    
    def getRoiStackPosition(self, mode='weightedmean'):
        planeIdx = self.getPlaneIdx()
        stat = self.loadS2P('stat')
        lam = [s['lam'] for s in stat]
        ypix = [s['ypix'] for s in stat]
        xpix = [s['xpix'] for s in stat]
        if mode=='weightedmean':
            yc = np.array([np.sum(l*y)/np.sum(l) for l,y in zip(lam,ypix)])
            xc = np.array([np.sum(l*x)/np.sum(l) for l,x in zip(lam,xpix)])
        elif mode=='median':
            yc = np.array([np.median(y) for y in ypix])
            xc = np.array([np.median(x) for x in xpix])
        stackPosition = np.stack((xc,yc,planeIdx)).T
        return stackPosition

    def getNumROIs(self, keepPlanes=None):
        keepPlanes = keepPlanes if keepPlanes is not None else [i for i in range(len(self.value['roiPerPlane']))]
        return sum([self.value['roiPerPlane'][p] for p in keepPlanes])

    def idxToPlanes(self, keepPlanes=None):
        keepPlanes = keepPlanes if keepPlanes is not None else [i for i in range(len(self.value['roiPerPlane']))]
        stackPosition = self.loadone('mpciROIs.stackPosition')
        roiPlaneIdx = stackPosition[:,2].astype(np.int32) # plane index
        return np.any(np.stack([roiPlaneIdx==pidx for pidx in keepPlanes]),axis=0)
        
    def getRedIdx(self, include_manual=True, keepPlanes=None):
        """special loading method for getting red cell index (for handling manual assignment)"""
        assert 'mpciROIs.redCellIdx' in self.printSavedOne(), "mpciROIs.redCellIdx is not a saved one variable, this is required for loading the red index!"
        redidx = self.loadone('mpciROIs.redCellIdx')
        if include_manual:
            assert 'mpciROIs.redCellManualAssignments' in self.printSavedOne(), "mpciROIs.redCellManualAssignments is not a saved one variable, this is required for loading the red index and including manual assignments!"
            redmanual = self.loadone('mpciROIs.redCellManualAssignments')
            redmanual_assignment = redmanual[0]
            redmanual_active = redmanual[1]
            redidx[redmanual_active] = redmanual_assignment[redmanual_active]
        in_plane_idx = self.idxToPlanes(keepPlanes=keepPlanes)
        return redidx[in_plane_idx]
            
    
    # ---------------------------------------- postprocessing functions for translating behavior to imaging time frame -----------------------------------------------------
    def getFrameBehavior(self):
        # convert behavioral data to a timescale that is aligned with imaging data
        trialStartSample = self.loadone('trials.positionTracking') # behavioral sample that each trial starts on
        behaveTimes = self.loadone('positionTracking.times') # time of each positionTracking sample
        behavePosition = self.loadone('positionTracking.position') # virtual position of each behavioral sample 
        behaveTrialIdx = self.getBehaveTrialIdx(trialStartSample) # trial index of each behavioral sample

        frameTimeStamps = self.loadone('mpci.times') # timestamps for each imaging frame
        samplingRate = 1/np.median(np.diff(frameTimeStamps)) # median sampling rate (can use self.value['samplingDeviationMedianPercentError'] to determine if this is smart)
        
        # behave timestamps has higher temporal resolution than frame timestamps, so we need to average over behavioral frames
        idxBehaveToFrame = self.loadone('positionTracking.mpci') # mpci frame index associated with each behavioral frame
        distBehaveToFrame = frameTimeStamps[idxBehaveToFrame] - behaveTimes
        
        # get behavioral variables associated with each imaging frame (preallocate, then use special behaveToFrame numba function)
        framePosition = np.zeros(self.value['numFrames']) 
        frameTrialIdx = np.zeros(self.value['numFrames'])
        count = np.zeros(self.value['numFrames'])
        functions.behaveToFrame(behavePosition,behaveTrialIdx,idxBehaveToFrame,distBehaveToFrame,1/2/samplingRate,framePosition,frameTrialIdx,count)
        framePosition[count==0]=np.nan # behaveToFrame uses a single pass summation method, so count==0 indicates that no behavioral data was available for that frame
        frameTrialIdx[count==0]=np.nan
        assert np.min(frameTrialIdx[count>0])==0 and np.max(frameTrialIdx[count>0])==self.value['numTrials']-1, "frameTrialIdx doesn't have correct number of trials"
        
        # Make sure trial indices are integers -- otherwise (in most cases this means multiple behavioral trials were matched with same neural frame)
        assert np.all([i.is_integer() for i in frameTrialIdx[count>0]]), "some neural frames were associated with multiple trials, non integers found"
        
        # Occasionally some neural frames do not map onto any best behavioral frame (usually for random slow behavioral sample) -- interpolate those samples 
        for trial in range(self.value['numTrials']):
            cTrialIdx = np.where(frameTrialIdx==trial)[0] # find frames associated with particular trial
            trialSlice = slice(cTrialIdx[0], cTrialIdx[-1]+1) # get slice from first to last frame within trial 
            withinTrialNan = np.isnan(frameTrialIdx[trialSlice]) # find nans -- this means the sampling was slow in behavioral computer and requires filling in
            if np.any(withinTrialNan):
                # interpolate missing position data using simple linear interpolation, fill in trial index accordingly
                trialTimeStamps = frameTimeStamps[trialSlice]
                trialPosition = framePosition[trialSlice]
                frameTrialIdx[trialSlice]=trial
                trialPosition[withinTrialNan]=np.interp(trialTimeStamps[withinTrialNan], trialTimeStamps[~withinTrialNan], trialPosition[~withinTrialNan])
                framePosition[trialSlice] = trialPosition
        
        # Once position is fully interpolated on each trial, compute speed (let last sample of each trial have undefined (nan) speed)
        frameSpeed = np.append(np.diff(framePosition)/np.diff(frameTimeStamps), np.nan) # never assume sampling rate is perfect... 
        
        return frameTrialIdx, framePosition, frameSpeed
    
    
    # -------------------- post-processing of behavioral data in one format -----------------
    def getBehaveTrialIdx(self, trialStartFrame):
        # produces a 1-d numpy array where each element indicates the trial index of the behavioral sample onedata arrays
        nspt = np.array([*np.diff(trialStartFrame), self.value['numBehaveTimestamps']-trialStartFrame[-1]]) # num samples per trial
        return np.concatenate([tidx*np.ones(ns) for (tidx,ns) in enumerate(nspt)]).astype(np.uint64)

    def groupBehaveByTrial(self, data, trialStartFrame):
        # returns a list containing trial-separated behavioral data 
        trialIndex = self.getBehaveTrialIdx(trialStartFrame)
        return [data[trialIndex==tidx] for tidx in range(len(trialStartFrame))]
        

    
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- vrExperiment Red Cell Processing Object ------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class redCellProcessing(vrExperiment):
    """
    The redCellProcessing object is devoted to handling red cell processing.
    It accepts as input either a preprocessed vrExperiment object, or three strings indicating the mouseName, dateString, and session.
    This is built to be used as a standalone object, but can be controlled with a Napari based GUI that is currently being written (230526-ATL)
    """
    name = 'redCellProcessing'
    def __init__(self,*inputs, umPerPixel=1.3, autoload=True):
        # Create object
        self.createObject(*inputs)
        
        # Make sure redcell is available...
        assert 'redcell' in self.value['available'], "redcell is not an available suite2p output. That probably means there is no data on the red channel, so you can't do redCellProcessing."
        
        # standard names of the features used to determine red cell criterion
        self.featureNames = ['S2P','dotProduct','pearson','phaseCorrelation']
        
        # load some critical values for easy readable access
        self.numPlanes = len(self.value['planeNames'])
        self.umPerPixel = umPerPixel # store this for generating correct axes and measuring distances
        
        self.data_loaded = False # initialize to false in case data isn't loaded
        if autoload:
            self.loadReferenceAndMasks() # prepare reference images and ROI mask data
            
        
    # ------------------------------
    # -- initialization functions --
    # ------------------------------
    def loadReferenceAndMasks(self):
        # load reference images
        ops = self.loadS2P('ops')
        self.reference = [op['meanImg_chan2'] for op in ops]
        self.lx,self.ly = self.reference[0].shape
        for ref in self.reference: assert (self.lx,self.ly)==ref.shape, "reference images do not all have the same shape"

        # load masks (lam=weight of each pixel, xpix & ypix=index of each pixel in ROI mask)
        stat = self.loadS2P('stat')
        self.lam = [s['lam'] for s in stat]
        self.ypix = [s['ypix'] for s in stat]
        self.xpix = [s['xpix'] for s in stat]
        self.roiPlaneIdx = self.loadone('mpciROIs.stackPosition')[:,2]
        
        # load S2P red cell value
        self.redS2P = self.loadone('mpciROIs.redS2P') # (preloaded, will never change in this function)
        
        # create supporting variables for mapping locations and axes
        self.yBaseRef = np.arange(self.ly)
        self.xBaseRef = np.arange(self.lx)
        self.yDistRef = self.createCenteredAxis(self.ly, self.umPerPixel)
        self.xDistRef = self.createCenteredAxis(self.lx, self.umPerPixel)
        
        # update data_loaded field
        self.data_loaded = True
    
    # ---------------------------------
    # -- updating one data functions --
    # ---------------------------------
    def oneNameFeatureCutoffs(self, name):
        """standard method for naming the features used to define redCellIdx cutoffs"""
        return 'parameters'+'Red'+name[0].upper()+name[1:]+'.minMaxCutoff'
    
    def updateRedIdx(self, s2p_cutoff=None, dotProd_cutoff=None, corrCoef_cutoff=None, pxcValues_cutoff=None):
        """method for updating the red index given new cutoff values"""
        # create initial all true red cell idx
        redCellIdx = np.full(self.loadone('mpciROIs.redCellIdx').shape, True)
        
        # load feature values for each ROI
        redS2P = self.loadone('mpciROIs.redS2P')
        dotProduct = self.loadone('mpciROIs.redDotProduct')
        corrCoef = self.loadone('mpciROIs.redPearson')
        phaseCorr = self.loadone('mpciROIs.redPhaseCorrelation')
        
        # create lists for zipping through each feature/cutoff combination
        features = [redS2P, dotProduct, corrCoef, phaseCorr]
        cutoffs = [s2p_cutoff, dotProd_cutoff, corrCoef_cutoff, pxcValues_cutoff]
        usecutoff = [[False,False] for _ in range(len(cutoffs))]
        
        # check validity of each cutoff and identify whether it should be used
        for name, use, cutoff in zip(self.featureNames, usecutoff, cutoffs):
            assert isinstance(cutoff, np.ndarray), f"{name} cutoff is an numpy ndarray"
            assert len(cutoff)==2, f"{name} cutoff does not have 2 elements"
            if not(np.isnan(cutoff[0])): use[0]=True
            if not(np.isnan(cutoff[1])): use[1]=True
            
        # add feature cutoffs to redCellIdx (sets any to False that don't meet the cutoff)
        for feature, use, cutoff in zip(features, usecutoff, cutoffs):
            if use[0]:
                redCellIdx &= feature >= cutoff[0]
            if use[1]:
                redCellIdx &= feature <= cutoff[1]
        
        # save new red cell index to one data
        self.saveone(redCellIdx, 'mpciROIs.redCellIdx')
        
        # save feature cutoffs to one data 
        for idx,name in enumerate(self.featureNames):
            self.saveone(cutoffs[idx], self.oneNameFeatureCutoffs(name))
        print(f"Red Cell curation choices are saved for session {self.sessionPrint()}")
        
    def updateFromSession(self, redCell, force_update=False):
        """method for updating the red cell cutoffs from another session"""
        assert isinstance(redCell, redCellProcessing), "redCell is not a redCellProcessing object"
        if not(force_update):
            assert redCell.mouseName == self.mouseName, "session to copy from is from a different mouse, this isn't allowed without the force_update=True input"
        cutoffs = [redCell.loadone(redCell.oneNameFeatureCutoffs(name)) for name in self.featureNames]
        self.updateRedIdx(s2p_cutoff=cutoffs[0], dotProd_cutoff=cutoffs[1], corrCoef_cutoff=cutoffs[2], pxcValues_cutoff=cutoffs[3])
        
        
    # ------------------------------
    # -- classification functions --
    # ------------------------------
    def computeFeatures(self, planeIdx=None, width=40, eps=1e6, winFunc=lambda x: np.hamming(x.shape[-1]), lowcut=12, highcut=250, order=3, fs=512):
        """
        This function computes all features related to red cell detection in (hopefully) an optimized manner, such that loading the redSelectionGUI is fast and efficient.
        There's some shortcuts that must be done, including:
        - pre-filtering reference images before computing phase-correlation
        """
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make planeIdx iterable
        if not(self.data_loaded): self.loadReferenceAndMasks()
        
        # start with filtered reference stack (by inspection, the phase correlation is minimally dependent on pre-filtering, and we like these filtering parameters anyway!!!)
        print("Creating centered reference images...")
        refStackNans = self.centeredReferenceStack(planeIdx=planeIdx, width=width, fill=np.nan, filtPrms=(lowcut,highcut,order,fs)) # stack of ref images centered on each ROI
        refStack = np.copy(refStackNans)
        refStack[np.isnan(refStack)]=0
        maskStack = self.centeredMaskStack(planeIdx=planeIdx, width=width) # stack of mask value centered on each ROI
        
        print("Computing phase correlation for each ROI...")
        window = winFunc(refStack)
        pxcStack = np.stack([helpers.phaseCorrelation(ref,mask,eps=eps,window=window) for (ref,mask) in zip(refStack, maskStack)]) # measure phase correlation
        pxcCenterPixel = int((pxcStack.shape[2]-1)/2)
        pxcValues = pxcStack[:,pxcCenterPixel,pxcCenterPixel]
        
        # next, compute the dot product between the filtered reference and masks
        refNorm = np.linalg.norm(refStack, axis=(1,2)) # compute the norm of each centered reference image
        maskNorm = np.linalg.norm(maskStack, axis=(1,2)) 
        dotProd = np.sum(refStack * maskStack, axis=(1,2))/refNorm/maskNorm # compute the dot product for each ROI
        
        # next, compute the correlation coefficients
        refStackNans = np.reshape(refStackNans,(refStackNans.shape[0],-1))
        maskStack = np.reshape(maskStack,(maskStack.shape[0],-1))
        maskStack[np.isnan(refStackNans)]=np.nan # remove the border areas from the masks stack (they are nan in the refStackNans array)
        uRef = np.nanmean(refStackNans,axis=1,keepdims=True)
        uMask = np.nanmean(maskStack,axis=1,keepdims=True)
        sRef = np.nanstd(refStackNans,axis=1)
        sMask = np.nanstd(maskStack,axis=1)
        N = np.sum(~np.isnan(refStackNans),axis=1)
        corrCoef = np.nansum((refStackNans-uRef)*(maskStack-uMask),axis=1)/N/sRef/sMask
        
        # And return
        return dotProd, corrCoef, pxcValues
    
    def croppedPhaseCorrelation(self, planeIdx=None, width=40, eps=1e6, winFunc=lambda x:np.hamming(x.shape[-1])):
        """
        This returns the phase correlation of each (cropped) mask with the (cropped) reference image.
        The default parameters (width=40um, eps=1e6, and a hamming window function) were tested on a few sessions and is purely subjective. 
        I recommend that if you use this function to determine which of your cells are red, you do manual curation and potentially update some of these parameters. 
        """
        if not(self.data_loaded): self.loadReferenceAndMasks()
        if winFunc=='hamming': winFunc = lambda x : np.hamming(x.shape[-1])
        refStack = self.centeredReferenceStack(planeIdx=planeIdx,width=width) # get stack of reference image centered on each ROI
        maskStack = self.centeredMaskStack(planeIdx=planeIdx,width=width) # get stack of mask value centered on each ROI
        window = winFunc(refStack) # create a window function
        pxcStack = np.stack([helpers.phaseCorrelation(ref,mask,eps=eps,window=window) for (ref,mask) in zip(refStack,maskStack)]) # measure phase correlation 
        pxcCenterPixel = int((pxcStack.shape[2]-1)/2)
        return refStack, maskStack, pxcStack, pxcStack[:,pxcCenterPixel,pxcCenterPixel]
    
    def computeDot(self, planeIdx=None, lowcut=12, highcut=250, order=3, fs=512):
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make planeIdx iterable
        if not(self.data_loaded): self.loadReferenceAndMasks()
        
        dotProd = []
        for plane in planeIdx:
            t = time.time()
            cRoiIdx = np.where(self.roiPlaneIdx==plane)[0] # index of ROIs in this plane
            bwReference = helpers.butterworthbpf(self.reference[plane], lowcut, highcut, order=order, fs=fs) # filtered reference image
            bwReference /= np.linalg.norm(bwReference) # adjust to norm for straightforward cosine angle
            # compute normalized dot product for each ROI 
            dotProd.append(np.array([bwReference[self.ypix[roi],self.xpix[roi]]@self.lam[roi]/np.linalg.norm(self.lam[roi]) for roi in cRoiIdx]))
            
        return np.concatenate(dotProd)
    
    def computeCorr(self, planeIdx=None, width=20, lowcut=12, highcut=250, order=3, fs=512):
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make planeIdx iterable
        if not(self.data_loaded): self.loadReferenceAndMasks()
        
        corrCoef = []
        for plane in planeIdx:
            numROIs = self.value['roiPerPlane'][plane]
            cRoiIdx = np.where(self.roiPlaneIdx==plane)[0] # index of ROIs in this plane
            cRefStack = np.reshape(self.centeredReferenceStack(planeIdx=plane,width=width,fill=np.nan,filtPrms=(lowcut,highcut,order,fs)),(numROIs,-1))
            cMaskStack = np.reshape(self.centeredMaskStack(planeIdx=plane,width=width,fill=0),(numROIs,-1))
            cMaskStack[np.isnan(cRefStack)]=np.nan
            
            # Measure mean and standard deviation (and number of non-nan datapoints)
            uRef = np.nanmean(cRefStack,axis=1,keepdims=True)
            uMask = np.nanmean(cMaskStack,axis=1,keepdims=True)
            sRef = np.nanstd(cRefStack,axis=1)
            sMask = np.nanstd(cMaskStack,axis=1)
            N = np.sum(~np.isnan(cRefStack),axis=1)
            # compute correlation coefficient and add to storage variable
            corrCoef.append(np.nansum((cRefStack - uRef) * (cMaskStack - uMask), axis=1) / N / sRef / sMask)
        
        return np.concatenate(corrCoef)
        
    # --------------------------
    # -- supporting functions --
    # --------------------------
    def createCenteredAxis(self, numElements, scale=1):
        return scale*(np.arange(numElements)-(numElements-1)/2)
    
    def getyref(self, yCenter):
        if not(self.data_loaded): self.loadReferenceAndMasks()
        return self.umPerPixel * (self.yBaseRef - xCenter)
    
    def getxref(self, xCenter):
        if not(self.data_loaded): self.loadReferenceAndMasks()
        return self.umPerPixel * (self.xBaseRef - xCenter)
    
    def getRoiCentroid(self,idx,mode='weightedmean'):
        if not(self.data_loaded): self.loadReferenceAndMasks()
        
        if mode=='weightedmean':
            yc = np.sum(self.lam[idx]*self.ypix[idx])/np.sum(self.lam[idx])
            xc = np.sum(self.lam[idx]*self.xpix[idx])/np.sum(self.lam[idx])
        elif mode=='median':
            yc = int(np.median(self.ypix[idx]))
            xc = int(np.median(self.xpix[idx]))
        return yc,xc
    
    def getRoiRange(self,idx):
        if not(self.data_loaded): self.loadReferenceAndMasks()
        # get range of x and y pixels for a particular ROI
        yr = np.ptp(self.ypix[idx])
        xr = np.ptp(self.xpix[idx])
        return yr,xr
    
    def getRoiInPlaneIdx(self,idx): 
        if not(self.data_loaded): self.loadReferenceAndMasks()
        # return index of ROI within it's own plane
        planeIdx = self.roiPlaneIdx[idx]
        return idx - np.sum(self.roiPlaneIdx<planeIdx)
    
    def centeredReferenceStack(self,planeIdx=None,width=15,fill=0.,filtPrms=None):
        # return stack of reference images centered on each ROI (+/- width um around ROI centroid)
        # if planeIdx is none, then returns across all planes
        # fill determines what value to use as the background (should either be 0 or nan...)
        # if filterPrms=None, then just returns centered reference stack. otherwise, filterPrms requires a tuple of 4 parameters which define a butterworth filter
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make planeIdx iterable
        if not(self.data_loaded): self.loadReferenceAndMasks()
        numPixels = int(np.round(width / self.umPerPixel)) # numPixels to each side around the centroid
        refStack = []
        for plane in planeIdx:
            cReference = self.reference[plane]
            if filtPrms is not None: cReference = helpers.butterworthbpf(cReference, filtPrms[0], filtPrms[1], order=filtPrms[2], fs=filtPrms[3]) # filtered reference image
            idxRoiInPlane = np.where(self.roiPlaneIdx==plane)[0]
            refStack.append(np.full((len(idxRoiInPlane), 2*numPixels+1, 2*numPixels+1),fill))
            for idx,idxRoi in enumerate(idxRoiInPlane):
                yc,xc = self.getRoiCentroid(idxRoi,mode='median')
                yUse = (np.maximum(yc-numPixels,0),np.minimum(yc+numPixels+1,self.ly))
                xUse = (np.maximum(xc-numPixels,0),np.minimum(xc+numPixels+1,self.lx))
                yMissing = (-np.minimum(yc-numPixels,0),-np.minimum(self.ly - (yc+numPixels+1),0))
                xMissing = (-np.minimum(xc-numPixels,0),-np.minimum(self.lx - (xc+numPixels+1),0))
                refStack[-1][idx,yMissing[0]:2*numPixels+1-yMissing[1],xMissing[0]:2*numPixels+1-xMissing[1]] = cReference[yUse[0]:yUse[1],xUse[0]:xUse[1]]
        return np.concatenate(refStack,axis=0).astype(np.float32)
    
    def centeredMaskStack(self,planeIdx=None,width=15,fill=0.):
        # return stack of ROI Masks centered on each ROI (+/- width um around ROI centroid)
        # if planeIdx is none, then returns across all planes
        # fill determines what value to use as the background (should either be 0 or nan)
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make planeIdx iterable
        if not(self.data_loaded): self.loadReferenceAndMasks()
        numPixels = int(np.round(width / self.umPerPixel)) # numPixels to each side around the centroid
        maskStack = []
        for plane in planeIdx:
            idxRoiInPlane = np.where(self.roiPlaneIdx==plane)[0]
            maskStack.append(np.full((len(idxRoiInPlane), 2*numPixels+1, 2*numPixels+1),fill))
            for idx,idxRoi in enumerate(idxRoiInPlane):
                yc,xc = self.getRoiCentroid(idxRoi,mode='median')
                # centered y&x pixels of ROI
                cyidx = self.ypix[idxRoi] - yc + numPixels
                cxidx = self.xpix[idxRoi] - xc + numPixels
                # index of pixels still within width of stack
                idxUsePoints = (cyidx>=0) & (cyidx<2*numPixels+1) & (cxidx>=0) & (cxidx<2*numPixels+1)
                maskStack[-1][idx,cyidx[idxUsePoints],cxidx[idxUsePoints]]=self.lam[idxRoi][idxUsePoints]
        return np.concatenate(maskStack,axis=0).astype(np.float32)
    
    def computeVolume(self,planeIdx=None):
        if planeIdx is None: planeIdx = np.arange(self.numPlanes)
        if isinstance(planeIdx,(int,np.integer)): planeIdx=(planeIdx,) # make it iterable
        assert all([0<=plane<self.numPlanes for plane in planeIdx]), f"in session: {self.sessionPrint()}, there are only {self.numPlanes} planes!"
        if not(self.data_loaded): self.loadReferenceAndMasks()
        roiMaskVolume = []
        for plane in planeIdx:
            roiMaskVolume.append(np.zeros((self.value['roiPerPlane'][plane],self.ly,self.lx)))
            idxRoiInPlane = np.where(self.roiPlaneIdx==plane)[0]
            for roi in range(self.value['roiPerPlane'][plane]):
                cRoiIdx = idxRoiInPlane[roi]
                roiMaskVolume[-1][roi,self.ypix[cRoiIdx],self.xpix[cRoiIdx]]=self.lam[cRoiIdx]
        return np.concatenate(roiMaskVolume,axis=0)    
    
    