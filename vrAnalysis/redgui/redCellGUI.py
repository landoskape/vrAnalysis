# Standard modules
from copy import copy
import time
import functools 
import numpy as np
import scipy as sp
from scipy import ndimage as ndi
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# GUI-related modules
import napari
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsProxyWidget, QSlider, QPushButton, QVBoxLayout, QLabel, QLineEdit, QShortcut
from PyQt5.QtGui import QKeySequence

# Special vrAnalysis modules
from .. import session
from .. import helpers
from .. import database

pd.options.display.width = 1000

def compareFeatureCutoffs(*vrexp, roundValue=None):
    features = [
        'parametersRedS2P.minMaxCutoff',
        'parametersRedDotProduct.minMaxCutoff',
        'parametersRedPearson.minMaxCutoff',
        'parametersRedPhaseCorrelation.minMaxCutoff',
    ]
    dfDict = {
        'session': [ses.sessionPrint() for ses in vrexp]
    }
    def getFeatName(name):
        cname = name[name.find('Red')+3:name.find('.')]
        return cname #cname+'_min', cname+'_max'
    
    for feat in features:
        dfDict[getFeatName(feat)]=[None]*len(vrexp)
        
    for idx, ses in enumerate(vrexp):
        for feat in features:
            cdata = ses.loadone(feat)
            if cdata.dtype==object and cdata.item() is None:
                cdata = [None, None]
            else:
                if roundValue is not None: 
                    cdata = np.round(cdata, roundValue)
            dfDict[getFeatName(feat)][idx]=cdata
    
    print(pd.DataFrame(dfDict))
    return None

basicButtonStyle = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}

QPushButton:hover {
    background-color: #45a049;
    font-size: 10px;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 5px 5px;
}
"""

qCheckedStyle = """
QWidget {
    background-color: #1F1F1F;
    color: red;
    font-family: Arial, sans-serif;
}
"""

qNotCheckedStyle = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}
"""


        
class redSelectionGUI:
    def __init__(self, redCellObj, numBins=50):
        assert type(redCellObj)==session.redCellProcessing, "redCellObj must be an instance of the redCellProcessing class inherited from session"
        self.redCell = redCellObj
        self.numPlanes = self.redCell.numPlanes
        self.roiPerPlane = self.redCell.value['roiPerPlane']
        self.numBins = numBins
        self.planeIdx = 0 # keep track of which plane to observe
        
        self.refImage = [None]*self.numPlanes
        self.idxRoi = [None]*self.numPlanes
        self.featureNames = self.redCell.featureNames
        self.numFeatures = len(self.featureNames)
        self.featureActive = [[True, True] for _ in range(self.numFeatures)]
        self.features = [None]*self.numPlanes
        self.hvalues = [None]*self.numPlanes
        self.hvalred = [None]*self.numPlanes
        self.hedges = [None]*self.numFeatures
        
        # process initial plane
        self.controlCellToggle = False # If true, then self.maskImage() will display control cells rather than red cells
        self.redIdx = [np.full(self.roiPerPlane[planeIdx], True) for planeIdx in range(self.numPlanes)] # start with all as red... 
        self.manualLabel = [None]*self.numPlanes 
        self.manualLabelActive = [None]*self.numPlanes
        self.processPlanes() # compute reference / maskVolume / featureArrays for each plane 
        
        # open napari viewer and associated GUI features
        self.showMaskImage = True # if true, will show mask image, if false, will show mask labels
        self.maskVisibility = True # if true, will show either mask image or label, otherwise will not show either!
        self.useManualLabel = True # if true, then will apply manual labels after using features to compute redIdx
        self.onlyManualLabels = False # if true, only show manual labels of selected category...
        self.colorState = 0 # indicates which color to display maskLabels (0:random, 1-4:color by feature)
        self.idxColormap = 0 # which colormap to use for pseudo coloring the masks
        self.listColormaps = ['plasma', 'autumn', 'spring', 'summer', 'winter', 'hot']
        self.initializeNapariViewer()
        
        
    def initializeNapariViewer(self):
        # generate napari viewer
        self.viewer = napari.Viewer(title=f"Red Cell Curation from session: {self.redCell.sessionPrint()}")
        self.reference = self.viewer.add_image(np.stack(self.redCell.reference), name='reference', blending='additive', opacity=0.6)
        self.masks = self.viewer.add_image(self.maskImage(), name='masksImage', blending='additive', colormap='red', visible=self.showMaskImage)
        self.labels = self.viewer.add_labels(self.maskLabels(), name='maskLabels', blending='additive', visible=not(self.showMaskImage))
        self.viewer.dims.current_step = (self.planeIdx, self.viewer.dims.current_step[1], self.viewer.dims.current_step[2])#[0] = self.planeIdx
        
        # create feature and button widget
        self.featureWindow = pg.GraphicsLayoutWidget()
        
        # create components of the feature window (the top row is a sequence of histograms, the bottom row is some buttons and edit fields etc.)
        self.toggleArea = pg.GraphicsLayout()
        self.plotArea = pg.GraphicsLayout()
        self.buttonArea = pg.GraphicsLayout()
        self.featureWindow.addItem(self.toggleArea,row=0,col=0)
        self.featureWindow.addItem(self.plotArea,row=1,col=0)
        self.featureWindow.addItem(self.buttonArea,row=2,col=0)
        
        # start by making a specific layout for the histograms of the features
        self.histLayout = pg.GraphicsLayout()
        self.histGraphs = [None]*self.numFeatures
        self.histReds = [None]*self.numFeatures
        for feature in range(self.numFeatures):
            barWidth = np.diff(self.hedges[feature][:2])
            self.histGraphs[feature] = pg.BarGraphItem(x=helpers.edge2center(self.hedges[feature]), height=self.hvalues[self.planeIdx][feature], width=barWidth)
            self.histReds[feature] = pg.BarGraphItem(x=helpers.edge2center(self.hedges[feature]), height=self.hvalred[self.planeIdx][feature], width=barWidth, brush='r')
        
        # keep y-range of feature plots in useful regime
        def preserveYRange0(): preserveYRange(0)
        def preserveYRange1(): preserveYRange(1)
        def preserveYRange2(): preserveYRange(2)
        def preserveYRange3(): preserveYRange(3)
        preserveMethods = [preserveYRange0, preserveYRange1, preserveYRange2, preserveYRange3]
        
        def preserveYRange(idx):
            #for idx in range(self.numFeatures):
            self.histPlots[idx].getViewBox().sigYRangeChanged.disconnect(preserveMethods[idx])#preserveYRange)
            current_min, current_max = self.histPlots[idx].viewRange()[1]
            current_range = current_max - current_min
            current_max = min(current_range, self.hvaluesMaximum[idx])
            self.histPlots[idx].setYRange(0, current_max)
            self.histPlots[idx].getViewBox().sigYRangeChanged.connect(preserveMethods[idx])#preserveYRange)

        # add bargraphs to plotArea
        self.histPlots = [None]*self.numFeatures
        for feature in range(self.numFeatures):
            self.histPlots[feature] = self.plotArea.addPlot(row=0,col=feature,title=self.featureNames[feature])
            self.histPlots[feature].setMouseEnabled(x=False)
            self.histPlots[feature].setYRange(0, self.hvaluesMaximum[feature])
            self.histPlots[feature].addItem(self.histGraphs[feature])
            self.histPlots[feature].addItem(self.histReds[feature])
            self.histPlots[feature].getViewBox().sigYRangeChanged.connect(preserveMethods[feature])#preserveYRange)
        
        # create cutoffLines (vertical infinite lines) for determining the range within feature values that qualify as red
        def updateCutoffFinished(event, feature):
            cutoffValues = [self.cutoffLines[feature][0].pos()[0], self.cutoffLines[feature][1].pos()[0]]
            minCutoff, maxCutoff = min(cutoffValues), max(cutoffValues)
            self.featureCutoffs[feature][0] = minCutoff
            self.featureCutoffs[feature][1] = maxCutoff
            self.cutoffLines[feature][0].setValue(minCutoff)
            self.cutoffLines[feature][1].setValue(maxCutoff)
            self.updateRedIdx()
        
        self.featureRange = [None]*self.numFeatures
        self.featureCutoffs = [None]*self.numFeatures
        self.cutoffLines = [None]*self.numFeatures
        for feature in range(self.numFeatures):
            self.featureRange[feature] = [np.min(self.hedges[feature]), np.max(self.hedges[feature])]
            self.featureCutoffs[feature] = np.array([np.nan, np.nan])
            # check if feature cutoffs have been created and stored already, if so, use them
            if self.redCell.oneNameFeatureCutoffs(self.featureNames[feature]) in self.redCell.printSavedOne():
                cFeatureCutoff = self.redCell.loadone(self.redCell.oneNameFeatureCutoffs(self.featureNames[feature]))
                self.featureCutoffs[feature] = cFeatureCutoff
                if np.isnan(cFeatureCutoff[0]):
                    self.featureActive[feature][0] = False
                    self.featureCutoffs[feature][0] = self.featureRange[feature][0]
                if np.isnan(cFeatureCutoff[1]):
                    self.featureActive[feature][1] = False
                    self.featureCutoffs[feature][1] = self.featureRange[feature][1]
            self.cutoffLines[feature] = [None]*2 # one for minimum, one for maximum
            for ii in range(2):
                if self.featureActive[feature][ii]:
                    self.cutoffLines[feature][ii] = pg.InfiniteLine(pos=self.featureCutoffs[feature][ii], movable=True)
                else:
                    self.cutoffLines[feature][ii] = pg.InfiniteLine(pos=self.featureRange[feature][ii], movable=False)
                self.cutoffLines[feature][ii].setBounds(self.featureRange[feature])
                self.cutoffLines[feature][ii].sigPositionChangeFinished.connect(functools.partial(updateCutoffFinished, feature=feature))
                self.histPlots[feature].addItem(self.cutoffLines[feature][ii])
                
        # once cutoff lines are established, reset redIdx to prevent silly behavior
        self.updateRedIdx()
        
        # ---------------------
        # -- now add toggles --
        # ---------------------
        minMaxName = ['min','max']
        maxLengthName = max([len(name) for name in self.featureNames])+9
        def toggleFeature(event, name, idx, minmax):
            # set feature active based on whether toggle is checked
            self.featureActive[idx][minmax] = self.useFeatureButtons[idx][minmax].isChecked()
            if self.featureActive[idx][minmax]:
                # if feature is active, set value to cutoffs and make infinite line movable
                text_to_use = f"using {minMaxName[minmax]} {name}".center(maxLengthName, ' ')
                self.cutoffLines[idx][minmax].setValue(self.featureCutoffs[idx][minmax])
                self.cutoffLines[idx][minmax].setMovable(True)
                self.useFeatureButtons[idx][minmax].setText(text_to_use)
                self.useFeatureButtons[idx][minmax].setStyleSheet(qNotCheckedStyle)
            else:
                # if feature isn't active, set value to bounds and make infinite line unmovable
                text_to_use = f"ignore {minMaxName[minmax]} {name}".center(maxLengthName, ' ')
                self.cutoffLines[idx][minmax].setValue(self.featureRange[idx][minmax])
                self.cutoffLines[idx][minmax].setMovable(False)
                self.useFeatureButtons[idx][minmax].setText(text_to_use)
                self.useFeatureButtons[idx][minmax].setStyleSheet(qCheckedStyle)
                
            # then update red idx, which'll replot everything
            self.updateRedIdx()
            
        
        self.useFeatureButtons = [[None,None] for _ in range(self.numFeatures)]
        self.useFeatureButtonsProxy = [None]*(self.numFeatures*2)
        for featidx, featname in enumerate(self.featureNames):
            for i, name in enumerate(minMaxName):
                proxy_idx = 2*featidx + i
                if self.featureActive[featidx][i]:
                    text_to_use = f"using {minMaxName[i]} {featname}".center(maxLengthName, ' ')
                    style_to_use = qNotCheckedStyle
                else:
                    text_to_use = f"ignore {minMaxName[i]} {featname}".center(maxLengthName, ' ')
                    style_to_use = qCheckedStyle
                self.useFeatureButtons[featidx][i] = QPushButton('toggle',text=text_to_use)
                self.useFeatureButtons[featidx][i].setCheckable(True)
                self.useFeatureButtons[featidx][i].setChecked(self.featureActive[featidx][i])
                self.useFeatureButtons[featidx][i].clicked.connect(functools.partial(toggleFeature, name=featname, idx=featidx, minmax=i))
                self.useFeatureButtons[featidx][i].setStyleSheet(style_to_use)
                self.useFeatureButtonsProxy[proxy_idx] = QGraphicsProxyWidget()
                self.useFeatureButtonsProxy[proxy_idx].setWidget(self.useFeatureButtons[featidx][i])
                self.toggleArea.addItem(self.useFeatureButtonsProxy[proxy_idx], row=0, col=proxy_idx)
            
        # ---------------------
        # -- now add buttons --
        # ---------------------
        
        # create save button 
        def saveROIs(event):
            self.saveSelection()
            
        self.saveButton = QPushButton('button',text='save red selection')
        self.saveButton.clicked.connect(saveROIs)
        self.saveButton.setStyleSheet(basicButtonStyle)
        self.saveButtonProxy = QGraphicsProxyWidget()
        self.saveButtonProxy.setWidget(self.saveButton)
        
        # create update database button
        def updateDatabase(event):
            self.updateDatabase(True)
            
        self.updateDatabaseButton = QPushButton('button',text='update database (QC=True)')
        self.updateDatabaseButton.clicked.connect(updateDatabase)
        self.updateDatabaseButton.setStyleSheet(basicButtonStyle)
        self.updateDatabaseButtonProxy = QGraphicsProxyWidget()
        self.updateDatabaseButtonProxy.setWidget(self.updateDatabaseButton)
        
        # create update database button
        def updateDatabaseFalse(event):
            self.updateDatabase(False)
            
        self.updateDatabaseFalseButton = QPushButton('button',text='update database (QC=False)')
        self.updateDatabaseFalseButton.clicked.connect(updateDatabaseFalse)
        self.updateDatabaseFalseButton.setStyleSheet(basicButtonStyle)
        self.updateDatabaseFalseButtonProxy = QGraphicsProxyWidget()
        self.updateDatabaseFalseButtonProxy.setWidget(self.updateDatabaseFalseButton)
        
        
        # add toggle control/red cell button
        def toggleCellsToView(inputArgument):
            # changes whether to plot control or red cells (maybe add a textbox and update it so as to not depend on looking at the print outputs...)
            self.controlCellToggle = not(self.controlCellToggle)
            self.toggleCellButton.setText('control cells' if self.controlCellToggle else 'red cells')
            self.masks.data = self.maskImage()
            self.labels.data = self.maskLabels()
        
        self.toggleCellButton = QPushButton(text='control cells' if self.controlCellToggle else 'red cells')
        self.toggleCellButton.clicked.connect(toggleCellsToView)
        self.toggleCellButton.setStyleSheet(basicButtonStyle)
        self.toggleCellButtonProxy = QGraphicsProxyWidget()
        self.toggleCellButtonProxy.setWidget(self.toggleCellButton)
        
        # add button to toggle whether to include manual labels in mask plot
        def useManualLabel(event):
            self.useManualLabel = not(self.useManualLabel)
            self.useManualLabelButton.setText('using manual labels' if self.useManualLabel else 'ignoring manual labels')
            # update replot masks and recompute histograms
            self.regenerateMaskData()
            
        self.useManualLabelButton = QPushButton(text='using manual labels' if self.useManualLabel else 'ignoring manual labels')
        self.useManualLabelButton.clicked.connect(useManualLabel)
        self.useManualLabelButton.setStyleSheet(basicButtonStyle)
        self.useManualLabelProxy = QGraphicsProxyWidget()
        self.useManualLabelProxy.setWidget(self.useManualLabelButton)
        
        # add button to clear all manual labels
        def clearManualLabels(event):
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers==QtCore.Qt.ControlModifier:
                for plane in range(self.numPlanes): self.manualLabelActive[plane][:] = False
                self.regenerateMaskData()
            else:
                print('clearing manual labels requires a control click')
                
        self.clearManualLabelButton = QPushButton(text='clear manual labels')
        self.clearManualLabelButton.clicked.connect(clearManualLabels)
        self.clearManualLabelButton.setStyleSheet(basicButtonStyle)
        self.clearManualLabelProxy = QGraphicsProxyWidget()
        self.clearManualLabelProxy.setWidget(self.clearManualLabelButton)
        
        # add show manual labels only button
        def showManualLabels(event):
            self.onlyManualLabels = not(self.onlyManualLabels)
            if self.onlyManualLabels: self.useManualLabel = True 
            self.showManualLabelButton.setText('only manual labels' if self.onlyManualLabels else 'all labels')
            self.regenerateMaskData()
            
        self.showManualLabelButton = QPushButton(text='all labels')
        self.showManualLabelButton.clicked.connect(showManualLabels)
        self.showManualLabelButton.setStyleSheet(basicButtonStyle)
        self.showManualLabelProxy = QGraphicsProxyWidget()
        self.showManualLabelProxy.setWidget(self.showManualLabelButton)
        
        # add colormap selection button 
        def nextColorState(event):
            self.colorState = np.mod(self.colorState+1, len(self.colorButtonNames))
            self.colorButton.setText(self.colorButtonNames[self.colorState])
            self.updateLabelColors()
            
        self.colorButtonNames = ['random',*self.featureNames]
        self.colorButton = QPushButton(text=self.colorButtonNames[self.colorState])
        self.colorButton.setCheckable(False)
        self.colorButton.clicked.connect(nextColorState)
        self.colorButton.setStyleSheet(basicButtonStyle)
        self.colorButtonProxy = QGraphicsProxyWidget()
        self.colorButtonProxy.setWidget(self.colorButton)
        
        # add colormap selection button 
        def nextColormap(event):
            self.idxColormap = np.mod(self.idxColormap+1, len(self.listColormaps))
            self.colormapSelection.setText(self.listColormaps[self.idxColormap])
            self.updateLabelColors()
            
        self.colormapSelection = QPushButton(text=self.listColormaps[self.idxColormap])
        self.colormapSelection.clicked.connect(nextColormap)
        self.colormapSelection.setStyleSheet(basicButtonStyle)
        self.colormapSelectionProxy = QGraphicsProxyWidget()
        self.colormapSelectionProxy.setWidget(self.colormapSelection)
                                             
        self.buttonArea.addItem(self.saveButtonProxy, row=0, col=0)
        self.buttonArea.addItem(self.updateDatabaseButtonProxy, row=0, col=1)
        self.buttonArea.addItem(self.updateDatabaseFalseButtonProxy, row=0, col=2)
        self.buttonArea.addItem(self.toggleCellButtonProxy, row=0, col=3)
        self.buttonArea.addItem(self.useManualLabelProxy, row=0, col=4)
        self.buttonArea.addItem(self.showManualLabelProxy, row=0, col=5)
        self.buttonArea.addItem(self.clearManualLabelProxy, row=0, col=6)
        self.buttonArea.addItem(self.colorButtonProxy, row=0, col=7)
        self.buttonArea.addItem(self.colormapSelectionProxy, row=0, col=8)
        
        # add feature plots to napari window
        self.dockWindow = self.viewer.window.add_dock_widget(self.featureWindow, name='ROI Features', area='bottom')
        
        def switchImageLabel(viewer):
            self.showMaskImage = not(self.showMaskImage)
            self.updateVisibility()
            
        def updateMaskVisibility(viewer):
            self.maskVisibility = not(self.maskVisibility)
            self.updateVisibility()
            
        def updateReferenceVisibility(viewer):
            self.reference.visible = not(self.reference.visible)
            
        self.viewer.bind_key('t', toggleCellsToView, overwrite=True)
        self.viewer.bind_key('s', switchImageLabel, overwrite=True)
        self.viewer.bind_key('v', updateMaskVisibility, overwrite=True)
        self.viewer.bind_key('r', updateReferenceVisibility, overwrite=True)
        self.viewer.bind_key('c', nextColorState, overwrite=True)
        self.viewer.bind_key('a', nextColormap, overwrite=True)
        
        # create single-click callback for printing data about ROI features
        def singleClickLabel(layer, event):
            if not(self.labels.visible):
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return 
            
            # get click data
            planeIdx, yidx, xidx = [int(pos) for pos in event.position]
            labelIdx = self.labels.data[planeIdx, yidx, xidx]
            if labelIdx==0:
                self.viewer.status = "single-click on background, no ROI selected"
                return 
            
            # get ROI data
            roiIdx = labelIdx-1  # oh napari, oh napari
            inPlaneIdx = np.where(self.idxRoi[planeIdx]==(roiIdx))[0][0]
            featurePrint = [f"{featname}={featdata[inPlaneIdx]:.3f}" for featname, featdata in zip(self.featureNames, self.features[planeIdx])]
            
            stringToPrint = f"ROI: {roiIdx}, Plane Idx: {planeIdx}, (inPlane)ROI: {inPlaneIdx}, " + ' '.join(featurePrint)
            
            # only print single click data if alt is held down
            if 'Alt' in event.modifiers:
                print(stringToPrint)
            
            # always show message in viewer status
            self.viewer.status = stringToPrint
            
            
        def doubleClickLabel(layer, event):
            self.viewer.status = "you just double clicked!"
            
            # if not looking at labels, then don't allow manual selection (it would be random!)
            if not(self.labels.visible): 
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return 
            
            # if not looking at manual annotations, don't allow manual selection...
            if not(self.useManualLabel):
                self.viewer.status = "can only manually select cells when the manual labels are being used!"
                return 
            
            planeIdx, yidx, xidx = [int(pos) for pos in event.position]
            labelIdx = self.labels.data[planeIdx, yidx, xidx]
            if labelIdx==0:
                self.viewer.status = "double-click on background, no ROI identity toggled"
            else:
                if 'Alt' in event.modifiers:
                    self.viewer.status = "Alt was used, assuming you are trying to single click and not doing a manual label!"
                else:
                    roiIdx = labelIdx-1
                    inPlaneIdx = np.where(self.idxRoi[planeIdx]==(roiIdx))[0][0]
                    if 'Control' in event.modifiers: 
                        if self.onlyManualLabels:
                            self.manualLabelActive[planeIdx][inPlaneIdx] = False
                            self.viewer.status = f"you just removed the manual label from roi: {roiIdx}"
                        else:
                            self.viewer.status = f"you can only remove a label if you are only looking at manualLabels!"
                    else:
                        # manual annotation: if plotting control cells, then annotate as red (1), if plotting red cells, annotate as control (0)
                        newLabel = copy(self.controlCellToggle) 
                        self.manualLabel[planeIdx][inPlaneIdx] = newLabel
                        self.manualLabelActive[planeIdx][inPlaneIdx] = True
                        self.viewer.status = f"you just labeled roi: {roiIdx} with the identity: {newLabel}"
                    self.regenerateMaskData()
        
        self.labels.mouse_drag_callbacks.append(singleClickLabel)
        self.labels.mouse_double_click_callbacks.append(doubleClickLabel)
        self.masks.mouse_double_click_callbacks.append(doubleClickLabel)
        self.reference.mouse_double_click_callbacks.append(doubleClickLabel)
        
        # add callback for dimension slider
        def updatePlaneIdx(event):
            self.planeIdx = event.source.current_step[0]
            self.updateFeaturePlots()
        
        self.viewer.dims.events.connect(updatePlaneIdx)
        
    def updateVisibility(self):
        self.masks.visible = self.showMaskImage and self.maskVisibility
        self.labels.visible = not(self.showMaskImage) and self.maskVisibility
        
    def updateFeaturePlots(self):
        for feature in range(self.numFeatures):
            self.histGraphs[feature].setOpts(height=self.hvalues[self.planeIdx][feature])
            self.histReds[feature].setOpts(height=self.hvalred[self.planeIdx][feature])
    
    def updateLabelColors(self):
        if self.colorState==0:
            # then use random colors -- what I encoded here is the default 
            colormap = dict(zip([0,None],[np.array([0.,0.,0.,0.],dtype=np.single),np.array([0.,0.,0.,1.],dtype=np.single)]))
        else:
            # good colormaps: 
            norm = mpl.colors.Normalize(vmin=self.featureRange[self.colorState-1][0], vmax=self.featureRange[self.colorState-1][1])
            colors = plt.colormaps[self.listColormaps[self.idxColormap]](norm(np.concatenate([feat[self.colorState-1] for feat in self.features])))
            colormap = dict(zip(np.concatenate(self.idxRoi)+1, colors))
            colormap[0] = np.array([0.,0.,0.,0.],dtype=np.single) # add transparent background
        # Update colors of the labels
        self.labels.color = colormap
        
    def processPlanes(self):
        redS2P = self.redCell.loadone('mpciROIs.redS2P')
        redDot = self.redCell.loadone('mpciROIs.redDotProduct')
        redCorr = self.redCell.loadone('mpciROIs.redPearson')
        redPhase = self.redCell.loadone('mpciROIs.redPhaseCorrelation')
        manualLabels = self.redCell.loadone('mpciROIs.redCellManualAssignments')
        for planeIdx in range(self.numPlanes):
            self.refImage[planeIdx] = self.redCell.reference[planeIdx]
            self.idxRoi[planeIdx] = np.where(self.redCell.roiPlaneIdx==planeIdx)[0]
            self.manualLabel[planeIdx] = manualLabels[0][self.redCell.roiPlaneIdx==planeIdx]
            self.manualLabelActive[planeIdx] = manualLabels[1][self.redCell.roiPlaneIdx==planeIdx]
            self.features[planeIdx] = [None]*self.numFeatures
            self.features[planeIdx][0] = redS2P[self.redCell.roiPlaneIdx==planeIdx]
            self.features[planeIdx][1] = redDot[self.redCell.roiPlaneIdx==planeIdx]
            self.features[planeIdx][2] = redCorr[self.redCell.roiPlaneIdx==planeIdx]
            self.features[planeIdx][3] = redPhase[self.redCell.roiPlaneIdx==planeIdx]
            self.hvalues[planeIdx] = [None]*self.numFeatures
            self.hvalred[planeIdx] = [None]*self.numFeatures
            
        # use the same edges across planes
        for feature in range(self.numFeatures):
            featureAcrossPlanes = np.concatenate([featureData[feature] for featureData in self.features]) 
            self.hedges[feature] = np.histogram(featureAcrossPlanes, bins=self.numBins)[1]
                
        for planeIdx in range(self.numPlanes):
            for feature in range(self.numFeatures):
                self.hvalues[planeIdx][feature] = np.histogram(self.features[planeIdx][feature], bins=self.hedges[feature])[0]
                self.hvalred[planeIdx][feature] = np.histogram(self.features[planeIdx][feature][self.redIdx[planeIdx]], bins=self.hedges[feature])[0]
        
        # establish maximum for the yranges
        maxValue = [[max(hval) for hval in hvalue] for hvalue in self.hvalues]
        self.hvaluesMaximum = [max(x) for x in zip(*maxValue)]
        
    def maskLabels(self):
        # note that labelData handles indices in a complicated way so that it's easy to interface with Napari. Key points:
        # 1. ROIs are assigned an index that is unique across all ROIs independent of plane (the first ROI in plane 1 isn't ROI 0, it's 1 + the number of ROIs in plane 0)
        # 2. ROI indices are incremented by 1 when they are added to the "label" layer of the napari viewer. This is because the label layer uses "0" to indicate "no label"
        # 3. ROIs are only presented if they are True in "self.idxMasksToPlot", which is a boolean array of size (numROIsPerPlane,). (Hence the enumerated for loop...)
        labelData = np.zeros((self.numPlanes,self.redCell.ly,self.redCell.lx), dtype=int)
        for planeIdx in range(self.numPlanes):
            plotIdx = self.idxMasksToPlot(planeIdx)
            for idx,roi in enumerate(self.idxRoi[planeIdx]):
                if plotIdx[idx]:
                    labelData[planeIdx,self.redCell.ypix[roi],self.redCell.xpix[roi]] = roi+1 # 0 is transparent for a labels layer in napari, so 1 index the ROIs!
        return labelData
    
    def maskImage(self):
        imageData = np.zeros((self.numPlanes,self.redCell.ly,self.redCell.lx))
        for planeIdx in range(self.numPlanes):
            plotIdx = self.idxMasksToPlot(planeIdx)
            for idx,roi in enumerate(self.idxRoi[planeIdx]):
                if plotIdx[idx]:
                    imageData[planeIdx,self.redCell.ypix[roi],self.redCell.xpix[roi]] = self.redCell.lam[roi]
        return imageData
    
    def idxMasksToPlot(self, planeIdx):
        # standard function for determining which masks to plot for each plane
        if self.onlyManualLabels:
            # if only showing manual labels, initialize plot index as all false, then update as usual
            plotIdx = np.full(self.redIdx[planeIdx].shape, False)
        else:
            # if showing all labels, then initialize plotIdx with whatever is currently passing the feature rules
            plotIdx = np.copy(self.redIdx[planeIdx] if not(self.controlCellToggle) else ~self.redIdx[planeIdx])
        if self.useManualLabel:
            plotIdx[self.manualLabelActive[planeIdx]] = (self.manualLabel[planeIdx][self.manualLabelActive[planeIdx]]!=self.controlCellToggle)
        return plotIdx
    
    def updateRedIdx(self):
        for planeIdx in range(self.numPlanes):
            self.redIdx[planeIdx] = np.full(self.roiPerPlane[planeIdx], True) # start with all as red... 
            for feature in range(self.numFeatures):
                if not(np.isnan(self.featureCutoffs[feature][0])):
                    self.redIdx[planeIdx] &= self.features[planeIdx][feature] >= self.featureCutoffs[feature][0] # only keep in redIdx if above minimum 
                if not(np.isnan(self.featureCutoffs[feature][1])):
                    self.redIdx[planeIdx] &= self.features[planeIdx][feature] <= self.featureCutoffs[feature][1] # only keep in redIdx if below maximum
        
        # now that the red idx has been updated, we need new mask data and new histograms
        self.regenerateMaskData()
    
    def regenerateMaskData(self):
        self.masks.data = self.maskImage()
        self.labels.data = self.maskLabels()
        for planeIdx in range(self.numPlanes):
            for feature in range(self.numFeatures):
                if self.onlyManualLabels:
                    cRedIdx = np.full(self.redIdx[planeIdx].shape, False)
                else:
                    cRedIdx = np.copy(self.redIdx[planeIdx])
                if self.useManualLabel:
                    # if using manual label, any manual labels will overwrite red idx if manual label is active 
                    cRedIdx[self.manualLabelActive[planeIdx]] = self.manualLabel[planeIdx][self.manualLabelActive[planeIdx]]
                self.hvalred[planeIdx][feature] = np.histogram(self.features[planeIdx][feature][cRedIdx], bins=self.hedges[feature])[0]
                
        # regenerate histograms
        for feature in range(self.numFeatures):
            self.histReds[feature].setOpts(height=self.hvalred[self.planeIdx][feature])
        
    def saveSelection(self):
        fullRedIdx = np.concatenate(self.redIdx)
        fullManualLabels = np.stack((np.concatenate(self.manualLabel),np.concatenate(self.manualLabelActive)))
        self.redCell.saveone(fullRedIdx, 'mpciROIs.redCellIdx')
        self.redCell.saveone(fullManualLabels, 'mpciROIs.redCellManualAssignments')
        for idx,name in enumerate(self.featureNames):
            cFeatureCutoffs = self.featureCutoffs[idx]
            if not(self.featureActive[idx][0]): cFeatureCutoffs[0]=np.nan
            if not(self.featureActive[idx][1]): cFeatureCutoffs[1]=np.nan
            self.redCell.saveone(self.featureCutoffs[idx], self.redCell.oneNameFeatureCutoffs(name))
            
        print(f"Red Cell curation choices are saved for session {self.redCell.sessionPrint()}")
        
    def updateDatabase(self, state):
        vrdb = database.vrDatabase()
        success = vrdb.setRedCellQC(self.redCell.mouseName, self.redCell.dateString, self.redCell.session, state=state)
        if success:
            print(f"Successfully updated the redCellQC field of the database to {state} for session {self.redCell.sessionPrint()}")
        else:
            print(f"Failed to update the redCellQC field of the database for session {self.redCell.sessionPrint()}")
            
    # def oneNameFeatureCutoffs(self, name):
    #     return 'parameters'+'Red'+name[0].upper()+name[1:]+'.minMaxCutoff'
    
    
        
        
        
        
        
# converting uiPlottingFunctions.scrollMatchedImages into a redSelection GUI made to be similar to the same named function in Matlab
def redCellViewer(stacks, features, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    # supporting class for storing and updating the ROI displayed in redCellViewer()
    class currentROI:
        def __init__(self,minroi=0,maxroi=None):
            self.minroi=minroi
            self.maxroi=maxroi if maxroi is not None else np.inf
            assert self.minroi < self.maxroi, "minimum roi value must be less than maximum roi value"
            self.value = 0
        def update(self,value):
            self.value=np.minimum(np.maximum(value,self.minroi),self.maxroi)
    
    # handle inputs
    assert len(stacks)==3, "stacks should be a length 3 iterable containing 3-d centered stacks of the reference image, the mask, and the phase correlation plots"
    assert len(features)==4, "features should be a length 4 iterable containing the suite2p red probability, dot product, correlation coefficient, and central pxc point for each ROI in stacks"
    assert type(enableMouse)==bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    numFeatures = len(features)
    for stack in range(numStacks): assert stacks[stack].ndim==3, "stacks are not all 3-dimensional"
    for stack in range(numStacks): assert stacks[stack].shape[0]==numImages,  "number of ROIs to look through are not the same in each stack"
    for feature in features: assert isinstance(feature, np.ndarray), "features need to be numpy arrays"
    for feature in features: assert feature.size==numImages, "number of ROIs in each features array must be same as the number of ROIs in the stacks"
    stackTitles = ["reference", "mask", "phase-correlation"]
    featureTitles = ["S2P","dot(ref,mask)","corr(ref,mask)","pxc"]
    
    # keep track of current ROI (I have  no idea why I can't do this with a python int...
    roi = currentROI(minroi=0,maxroi=numImages-1)
    
    # measure minimum and maximum of each stack
    if preserveScale:  imLevels = [(np.min(stack),np.max(stack)) for stack in stacks]
    else: imLevels = [None]*numStacks # allocate list for simple code later on
        
    def updateStackIndex():
        # whenever the ROI is changed, update the images and the labels (and keep scale the same if necessary)
        for stack,image,imLevel,view in zip(stacks,imageItems,imLevels,views):
            image.setImage(stack[roi.value])
            label.setText(f"ROI {roi.value+1}/{numImages}")
            if preserveScale:
                image.setLevels(imLevel)
        # whenever the ROI is changed, update the infiniteLine position indicating the value of that particular ROI
        for feature,cvalROI in zip(features,currentValueROI):
            cvalROI.setValue(feature[roi.value])
    
    # create image items for each stack
    imageItems = [pg.ImageItem(image=stacks[stack][0],axisOrder='row-major') for stack in range(numStacks)] 
    if preserveScale:
        for imLevel,image in zip(imLevels,imageItems):
            image.setLevels(imLevel)
    
    # infLines are drawn over the stacks to help find the same position across stacks, they are linked across stacks. 
    if infLines:
        def updateLinePosX(event):
            for ixLine in ixLineItems: ixLine.setValue(event.x())
        def updateLinePosY(event):
            for iyLine in iyLineItems: iyLine.setValue(event.y())
        # start with the lines in the center (0,0) position
        xPosition = stacks[0].shape[2]/2 
        yPosition = stacks[0].shape[1]/2
        # create the lines, and add callbacks
        ixLineItems = [pg.InfiniteLine(pos=xPosition,angle=90,movable=True,pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        iyLineItems = [pg.InfiniteLine(pos=yPosition,angle=0,movable=True,pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        for ixLine,iyLine in zip(ixLineItems,iyLineItems):
            ixLine.sigPositionChangeFinished.connect(updateLinePosX)
            iyLine.sigPositionChangeFinished.connect(updateLinePosY)
    
    # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
    window = pg.GraphicsLayoutWidget(size=(1200,800))
    
    # Create graphics layout with viewboxes for the image stacks
    stackLayout = pg.GraphicsLayout()
    window.addItem(stackLayout, row=1, col=0)
    # create a viewbox for each stack, add the appropriate image to it, link images so they all move together
    views = [stackLayout.addViewBox(row=0,col=stack,enableMouse=enableMouse,lockAspect=lockAspect,invertY=True,name=stackTitles[stack]) for stack in range(numStacks)]
    for image,view in zip(imageItems, views): view.addItem(image)
    for view in views[1:]: 
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])
    # add infinite lines to mark positions if requested
    if infLines:
        for (ixLine,iyLine,view) in zip(ixLineItems,iyLineItems,views):
            view.addItem(ixLine)
            view.addItem(iyLine)
    
    # Create barplots for each feature
    histCenters, histValues, histRange = [],[],[]
    for feature in features: 
        # make histogram of each feature
        cHist,cEdges = np.histogram(feature,bins=50)
        histRange.append((cEdges[0],cEdges[-1])) # min/max of histogram for each feature
        histCenters.append(helpers.edge2center(cEdges)) # center of histogram bin for each feature
        histValues.append(cHist) # histogram value for each bin for each feature
    featureHistograms = [pg.BarGraphItem(x=histCenter,height=histValue,width=histCenter[1]-histCenter[0]) for histCenter,histValue in zip(histCenters,histValues)]
    featRedHistograms = [pg.BarGraphItem(x=histCenter,height=histValue/2,width=histCenter[1]-histCenter[0],brush='r') for histCenter,histValue in zip(histCenters,histValues)]

    # Create a graphics layout with bar graph plots for the features
    featureLayout = pg.GraphicsLayout()
    window.addItem(featureLayout, row=2, col=0)
    featurePlots = [featureLayout.addPlot(row=0,col=feature,enableMouse=False,title=featureTitles[feature]) for feature in range(numFeatures)]
    for featurePlot in featurePlots: featurePlot.setMouseEnabled(x=False, y=False)
    #featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    for featureHistogram,featurePlot in zip(featureHistograms, featurePlots): featurePlot.addItem(featureHistogram)
    # for featureHistogram,featurePlot in zip(featRedHistograms, featurePlots): featurePlot.addItem(featureHistogram)
    
    # Create vertical lines indicating the value of the currently presented cell
    currentValueROI = [pg.InfiniteLine(pos=features[feature][0],angle=90,movable=False,pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    for fplot,cv in zip(featurePlots,currentValueROI): fplot.addItem(cv)
    
    # Create a slider label for indicating which ROI is being presented    
    sliderNameProxy = QGraphicsProxyWidget()
    label = QLabel(f"ROI {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy,row=3,col=0)
    
    # Create a slider with prev/next buttons and an edit field to change which ROI is being presented
    def updateSlider(value):
        roi.update(value) # first try updating roi value
        slider.setValue(roi.value) # if it clipped, reset slider appropriately
        editField.setText(str(roi.value)) # update textfield
        updateStackIndex() # update which ROI is presented
        
    def prevROI():
        roi.update(roi.value-1) # try updating roi value 
        slider.setValue(roi.value) # update slider 
        editField.setText(str(roi.value)) # update textfield
        updateStackIndex() # update which ROI is presented
        
    def nextROI():
        roi.update(roi.value+1) # try updating roi value
        slider.setValue(roi.value) # update slider
        editField.setText(str(roi.value)) # update textfield
        updateStackIndex() # update which ROI is presented
    
    def gotoROI():
        if not editField.text().isdigit():
            editField.setText('invalid ROI')
            return
        textValue = int(editField.text())
        if (textValue < roi.minroi) or (textValue > roi.maxroi):
            editField.setText('invalid ROI')
            return
        # otherwise text is valid ROI
        roi.update(textValue)
        editField.setText(str(roi.value))
        slider.setValue(roi.value) # update slider
        updateStackIndex()
    
    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages-1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages/10))
    slider.setValue(roi.value)
    slider.valueChanged.connect(updateSlider)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)
        
    prevButtonProxy = QGraphicsProxyWidget()
    prevButton = QPushButton('button',text='Prev ROI')
    prevButton.clicked.connect(prevROI)
    prevButtonProxy.setWidget(prevButton)
    
    nextButtonProxy = QGraphicsProxyWidget()
    nextButton = QPushButton('button',text='Next ROI')
    nextButton.clicked.connect(nextROI)
    nextButtonProxy.setWidget(nextButton)
    
    editFieldProxy = QGraphicsProxyWidget()
    editField = QLineEdit()
    editField.setText('0')
    editFieldProxy.setWidget(editField)
    
    gotoEditProxy = QGraphicsProxyWidget()
    gotoButton = QPushButton('button',text='go to ROI')
    gotoButton.clicked.connect(gotoROI)
    gotoEditProxy.setWidget(gotoButton)
    
    # add shortcut for going to ROI without pressing the button...
    shortcut = QShortcut(QKeySequence("G"), window)
    shortcut.activated.connect(gotoROI)
        
    roiSelectionLayout = pg.GraphicsLayout()
    roiSelectionLayout.addItem(prevButtonProxy,row=0,col=0)
    roiSelectionLayout.addItem(sliderProxy,row=0,col=1)
    roiSelectionLayout.addItem(nextButtonProxy,row=0,col=2)
    roiSelectionLayout.addItem(editFieldProxy,row=0,col=3)
    roiSelectionLayout.addItem(gotoEditProxy,row=0,col=4)
    window.addItem(roiSelectionLayout,row=4,col=0)
    
    # show GUI and return window for programmatic interaction
    window.show()
    return window




# converting uiPlottingFunctions.scrollMatchedImages into a redSelection GUI made to be similar to the same named function in Matlab
def redSelectionAmorphous(stacks, features, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    assert isinstance(stacks, (list, tuple)) and len(stacks)>1, "stacks must be a tuple of at least 2 3-d image stacks"
    assert isinstance(features, (list, tuple)), "features must be a list or tuple (even if it only has one element...)"
    assert type(enableMouse)==bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    numFeatures = len(features)
    for stack in range(numStacks): assert stacks[stack].ndim==3, "stacks are not all 3-dimensional"
    for stack in range(numStacks): assert stacks[stack].shape[0]==numImages,  "number of images to look through are not the same in each stack"
    for feature in features: assert isinstance(feature, np.ndarray), "features need to be numpy arrays"
    for feature in features: assert feature.size==numImages, "number of values in each features must be same as number of images in each stack"
    
    # measure minimum and maximum of each stack
    if preserveScale: 
        imLevels = [(np.min(stack),np.max(stack)) for stack in stacks]
    else:
        imLevels = [None]*numStacks # allocate list for simple code later on
        
    def updateStackIndex(value):
        for stack,image,imLevel,view in zip(stacks,imageItems,imLevels,views):
            image.setImage(stack[value])
            label.setText(f"Image {value+1}/{numImages}")
            if preserveScale:
                image.setLevels(imLevel)
        for feature,cvalROI in zip(features,currentValueROI):
            cvalROI.setValue(feature[value])
            
            
    # create image items for each stack
    imageItems = [pg.ImageItem(image=stacks[stack][0],axisOrder='row-major') for stack in range(numStacks)] 
    if preserveScale:
        for imLevel,image in zip(imLevels,imageItems):
            image.setLevels(imLevel)
    
    if infLines:
        def updateLinePosX(event):
            for ixLine in ixLineItems: ixLine.setValue(event.x())
        def updateLinePosY(event):
            for iyLine in iyLineItems: iyLine.setValue(event.y())
        xPosition = stacks[0].shape[2]/2
        yPosition = stacks[0].shape[1]/2
        ixLineItems = [pg.InfiniteLine(pos=xPosition,angle=90,movable=True,pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        iyLineItems = [pg.InfiniteLine(pos=yPosition,angle=0,movable=True,pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        for ixLine,iyLine in zip(ixLineItems,iyLineItems):
            ixLine.sigPositionChangeFinished.connect(updateLinePosX)
            iyLine.sigPositionChangeFinished.connect(updateLinePosY)
    
    # Create graphics layout with viewboxes for the image stacks
    window = pg.GraphicsLayoutWidget()
    stackLayout = pg.GraphicsLayout()
    window.addItem(stackLayout, row=0, col=0)
    views = [stackLayout.addViewBox(row=0,col=stack,enableMouse=enableMouse,lockAspect=lockAspect,invertY=True) for stack in range(numStacks)]
    for image,view in zip(imageItems, views): view.addItem(image)
    for view in views[1:]: 
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])
        
    if infLines:
        for (ixLine,iyLine,view) in zip(ixLineItems,iyLineItems,views):
            view.addItem(ixLine)
            view.addItem(iyLine)
    
    # Create barplots for each feature
    histCenters, histValues, histRange = [],[],[]
    for feature in features: 
        # make histogram of each feature
        cHist,cEdges = np.histogram(feature,bins=100)
        histRange.append((cEdges[0],cEdges[-1])) # min/max of histogram for each feature
        histCenters.append(helpers.edge2center(cEdges)) # center of histogram bin for each feature
        histValues.append(cHist) # histogram value for each bin for each feature
    featureHistograms = [pg.BarGraphItem(x=histCenter,height=histValue,width=histCenter[1]-histCenter[0]) for histCenter,histValue in zip(histCenters,histValues)]
        
    # Create a graphics layout with bar graph plots for the features
    featureLayout = pg.GraphicsLayout()
    window.addItem(featureLayout, row=1, col=0)
    featurePlots = [featureLayout.addPlot(row=0,col=feature,enableMouse=False) for feature in range(numFeatures)]
    for featurePlot in featurePlots: featurePlot.setMouseEnabled(x=False, y=False)
    #featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    for featureHistogram,featurePlot in zip(featureHistograms, featurePlots): featurePlot.addItem(featureHistogram)
    
    # Create vertical lines indicating the value of the currently presented cell
    currentValueROI = [pg.InfiniteLine(pos=features[feature][0],angle=90,movable=False,pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    for fplot,cv in zip(featurePlots,currentValueROI): fplot.addItem(cv)
    
    print(hasattr(featurePlots[0],'getAxis'))
    
    # Create a slider for selecting which slice of the image stacks to look at    
    sliderNameProxy = QGraphicsProxyWidget()
    sliderName = QVBoxLayout()
    label = QLabel(f"Image {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy,row=2,col=0)
    
    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages-1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages/10))
    slider.setValue(0)
    slider.valueChanged.connect(updateStackIndex)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)
    window.addItem(sliderProxy,row=3,col=0)
    
    window.show()
    
    return window


