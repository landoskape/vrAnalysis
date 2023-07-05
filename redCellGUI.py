import time
import functools 
import numpy as np
import scipy as sp
import napari
from scipy import ndimage as ndi
import pyqtgraph as pg
import basicFunctions as bf
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsProxyWidget, QSlider, QPushButton, QVBoxLayout, QLabel, QLineEdit, QShortcut
from PyQt5.QtGui import QKeySequence
import vrExperiment
import basicFunctions as bf

class redSelectionGUI:
    def __init__(self, redCellObj, planeIdx=0, numBins=50):
        assert type(redCellObj)==vrExperiment.redCellProcessing, "redCellObj must be an instance of the redCellProcessing class inherited from vrExperiment"
        self.redCell = redCellObj
        self.numPlanes = self.redCell.numPlanes
        self.roiPerPlane = self.redCell.value['roiPerPlane']
        self.numBins = numBins
        self.planeIdx = planeIdx # keep track of which plane to observe
        self.planeProcessed = [False]*self.numPlanes
        
        self.refImage = [None]*self.numPlanes
        self.maskStack = [None]*self.numPlanes
        self.features = [None]*self.numPlanes
        # self.redS2P = [self.redCell.redS2P[self.redCell.roiPlaneIdx==planeIdx] for planeIdx in range(self.numPlanes)]
        # self.redDot = [None]*self.numPlanes
        # self.redCorr = [None]*self.numPlanes
        # self.redPxc = [None]*self.numPlanes
        self.hvalues = [None]*self.numPlanes
        self.hvalred = [None]*self.numPlanes
        self.hedges = [None]*self.numPlanes
        self.featureNames = ['S2P','dot(ref,mask)','corr(ref,mask)','phase-correlation']
        self.numFeatures = len(self.featureNames)
        
        # process initial plane
        self.roiIdx = np.full(self.roiPerPlane[self.planeIdx], True) # start with all as red... 
        self.processPlane(self.planeIdx) # compute reference / maskVolume / featureArrays for plane 
        
        # open napari viewer and associated GUI features
        self.initializeNapariViewer()
        
        
        
    def initializeNapariViewer(self):
        # generate napari viewer
        self.viewer = napari.Viewer()
        self.reference = self.viewer.add_image(self.refImage[self.planeIdx], name='reference', blending='additive')
        self.masks = self.viewer.add_image(self.maskImage(), name='masks', blending='additive', colormap='red')
        
        # create feature widget
        self.featureWindow = pg.GraphicsLayoutWidget()
        
        # start by making a specific layout for the histograms of the features
        self.histLayout = pg.GraphicsLayout()
        self.histGraphs = [None]*self.numFeatures
        self.histReds = [None]*self.numFeatures
        for feature in range(self.numFeatures):
            barWidth = np.diff(self.hedges[self.planeIdx][feature][:2])
            self.histGraphs[feature] = pg.BarGraphItem(x=bf.edge2center(self.hedges[self.planeIdx][feature]), height=self.hvalues[self.planeIdx][feature], width=barWidth)
            self.histReds[feature] = pg.BarGraphItem(x=bf.edge2center(self.hedges[self.planeIdx][feature]), height=self.hvalred[self.planeIdx][feature], width=barWidth, brush='r')
        
        # add bargraphs to featureWindow
        self.histPlots = [None]*self.numFeatures
        for feature in range(self.numFeatures):
            self.histPlots[feature] = self.featureWindow.addPlot(row=0,col=feature,enableMouse=False,title=self.featureNames[feature])
            self.histPlots[feature].addItem(self.histGraphs[feature])
            self.histPlots[feature].addItem(self.histReds[feature])
        
        # add feature plots to napari window
        self.dockWindow = self.viewer.window.add_dock_widget(self.featureWindow, name='ROI Features', area='bottom')
        
        # create some keystrokes for the viewer (currently the hello1 is just an example of how to do it...)
        def hello1(viewer):
            print('helloworld')
            x = np.random.randint(0,100)
            self.viewer.status = f'hello {x}'
            yield
            self.viewer.status = 'goodbye'
            
        self.viewer.bind_key('r', hello1, overwrite=True) 
        
        
        
        
    def processPlane(self, planeIdx, forceProcess=False):
        if forceProcess or not self.planeProcessed[planeIdx]:
            t = time.time()
            print(f"Processing plane {planeIdx} for session {self.redCell.sessionPrint()}...")
            self.refImage[planeIdx] = self.redCell.reference[planeIdx]
            self.maskStack[planeIdx] = self.redCell.computeVolume(planeIdx=planeIdx).transpose(1,2,0)
            self.features[planeIdx] = [None]*self.numFeatures
            self.features[planeIdx][0] = self.redCell.redS2P[self.redCell.roiPlaneIdx==planeIdx]
            self.features[planeIdx][1] = self.redCell.computeDot(planeIdx=planeIdx)
            self.features[planeIdx][2] = self.redCell.computeCorr(planeIdx=planeIdx)
            self.features[planeIdx][3] = self.redCell.croppedPhaseCorrelation(planeIdx=planeIdx)[3]
            self.hvalues[planeIdx] = [None]*self.numFeatures
            self.hvalred[planeIdx] = [None]*self.numFeatures
            self.hedges[planeIdx] = [None]*self.numFeatures
            
            for feature in range(self.numFeatures):
                self.hvalues[planeIdx][feature], self.hedges[planeIdx][feature] = np.histogram(self.features[planeIdx][feature], bins=self.numBins)
                self.hvalred[planeIdx][feature], self.hedges[planeIdx][feature] = np.histogram(self.features[planeIdx][feature][self.roiIdx], bins=self.numBins)
                
            self.planeProcessed[planeIdx] = True
            print(f"Finished in {time.time()-t} seconds.")
   
    def maskImage(self):
        return np.sum(self.maskStack[self.planeIdx] * self.roiIdx, axis=2)
    
    def extras(self):
        # Create barplots for each feature
#         histCenters, histValues, histRange = [],[],[]
#         for feature in features: 
#             # make histogram of each feature
#             cHist,cEdges = np.histogram(feature,bins=50)
#             histRange.append((cEdges[0],cEdges[-1])) # min/max of histogram for each feature
#             histCenters.append(bf.edge2center(cEdges)) # center of histogram bin for each feature
#             histValues.append(cHist) # histogram value for each bin for each feature
#         featureHistograms = [pg.BarGraphItem(x=histCenter,height=histValue,width=histCenter[1]-histCenter[0]) for histCenter,histValue in zip(histCenters,histValues)]
#         featRedHistograms = [pg.BarGraphItem(x=histCenter,height=histValue/2,width=histCenter[1]-histCenter[0],brush='r') for histCenter,histValue in zip(histCenters,histValues)]

#         # Create a graphics layout with bar graph plots for the features
#         featureLayout = pg.GraphicsLayout()
#         window.addItem(featureLayout, row=2, col=0)
#         featurePlots = [featureLayout.addPlot(row=0,col=feature,enableMouse=False,title=featureTitles[feature]) for feature in range(numFeatures)]
#         for featurePlot in featurePlots: featurePlot.setMouseEnabled(x=False, y=False)
#         #featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
#         for featureHistogram,featurePlot in zip(featureHistograms, featurePlots): featurePlot.addItem(featureHistogram)
#         # for featureHistogram,featurePlot in zip(featRedHistograms, featurePlots): featurePlot.addItem(featureHistogram)

#         # Create vertical lines indicating the value of the currently presented cell
#         currentValueROI = [pg.InfiniteLine(pos=features[feature][0],angle=90,movable=False,pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
#         for fplot,cv in zip(featurePlots,currentValueROI): fplot.addItem(cv)
        
        
        
        
#         histPlot = window.addPlot(title="cutoff",row=0,col=0)
#         histPlot.addLegend()
#         histPlot.setRange(xRange=xRange,yRange=(0,1.1*np.max(histCurve)), update=False)
#         histPlot.setMouseEnabled(x=False, y=False)
#         cutoffLine = histPlot.plot()
#         cutoffLine.setData(np.arange(ND), histCurve) # create profile of cell masks along x axis 
#         cutoffVal1 = pg.InfiniteLine(pos=0,movable=True)
#         cutoffVal1.setBounds(xRange)
#         cutoffVal1.sigPositionChanged.connect(updateCutoff)
#         histPlot.addItem(cutoffVal1)
#         cutoffVal2 = pg.InfiniteLine(pos=ND-1,movable=True)
#         cutoffVal2.setBounds(xRange)
#         cutoffVal2.sigPositionChanged.connect(updateCutoff)
#         histPlot.addItem(cutoffVal2)

#         # try push buttons
#         def helloFromPush(event):
#             reference.data = np.random.normal(0,1,(ND,ND))
#             print('hello from push')

#         buttonProxy = QGraphicsProxyWidget()
#         button = QPushButton('button')
#         button.setDefault(True)
#         button.setCheckable(True)
#         button.setChecked(True)
#         button.clicked.connect(helloFromPush)
#         buttonProxy.setWidget(button)
#         window.addItem(buttonProxy,row=1,col=0)    

#         dockWidget = viewer.window.add_dock_widget(window, name='cutoffSelection', area='bottom')
        return None
    

def napariWithSlider(ND=512,NC=50,cellWidth=2):
    refImage = np.random.normal(0,1,(ND,ND))
    xCellLocations = np.random.randint(0,ND,NC)
    yCellLocations = np.random.randint(0,ND,NC)
    cellGrid = np.zeros((ND,ND,NC))
    cellGrid[yCellLocations,xCellLocations,np.arange(NC)] = 1
    cellMask = ndi.gaussian_filter(cellGrid, (cellWidth, cellWidth, 0))

    def convertToImage(cellMask, cellIdx):
        return np.sum(cellMask * cellIdx, axis=2)

    def updateCutoff(event):
        cutoffValues = (cutoffVal1.pos()[0], cutoffVal2.pos()[0])
        minCutoff = min(cutoffValues)
        maxCutoff = max(cutoffValues)
        masks.data = convertToImage(cellMask, (xCellLocations>=minCutoff) & (xCellLocations<maxCutoff))

    viewer = napari.Viewer()
    reference = viewer.add_image(refImage, name='reference', blending='additive')
    masks = viewer.add_image(convertToImage(cellMask, np.full(NC,True)), name='masks', blending='additive', colormap='red') 
    reference.metadata['count']=1

    histCurve = np.sum(convertToImage(cellMask, np.full(NC,True)),axis=0)
    xRange = (0,ND-1)
    window = pg.GraphicsLayoutWidget(title='hello world')
    histPlot = window.addPlot(title="cutoff",row=0,col=0)
    histPlot.addLegend()
    histPlot.setRange(xRange=xRange,yRange=(0,1.1*np.max(histCurve)), update=False)
    histPlot.setMouseEnabled(x=False, y=False)
    cutoffLine = histPlot.plot()
    cutoffLine.setData(np.arange(ND), histCurve) # create profile of cell masks along x axis 
    cutoffVal1 = pg.InfiniteLine(pos=0,movable=True)
    cutoffVal1.setBounds(xRange)
    cutoffVal1.sigPositionChanged.connect(updateCutoff)
    histPlot.addItem(cutoffVal1)
    cutoffVal2 = pg.InfiniteLine(pos=ND-1,movable=True)
    cutoffVal2.setBounds(xRange)
    cutoffVal2.sigPositionChanged.connect(updateCutoff)
    histPlot.addItem(cutoffVal2)

    # try push buttons
    def helloFromPush(event):
        reference.data = np.random.normal(0,1,(ND,ND))
        print('hello from push')

    buttonProxy = QGraphicsProxyWidget()
    button = QPushButton('button')
    button.setDefault(True)
    button.setCheckable(True)
    button.setChecked(True)
    button.clicked.connect(helloFromPush)
    buttonProxy.setWidget(button)
    window.addItem(buttonProxy,row=1,col=0)    

    dockWidget = viewer.window.add_dock_widget(window, name='cutoffSelection', area='bottom')

    @viewer.bind_key('r')
    def hello1(viewer):
        x = np.random.randint(0,100)
        viewer.status = f'hello {x}'
        yield
        viewer.status = 'goodbye'

    @reference.mouse_drag_callbacks.append
    def update_layer(layer, event):
        reference.metadata['count']+=1
        print(reference.metadata['count'])
        #layer.data = np.random.normal(0,reference.metadata['count'],layer.data.shape)

    return viewer, reference


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
        histCenters.append(bf.edge2center(cEdges)) # center of histogram bin for each feature
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
        histCenters.append(bf.edge2center(cEdges)) # center of histogram bin for each feature
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






# ------------------------------------------------------------------------------------------
# -- I copied everything from testingNapari.py below because it'll probably be useful!
# ------------------------------------------------------------------------------------------

def testHistSlider(cutoff=0):
    ND = 512
    def genBackground(ND,cutoff=0):
        background = np.repeat(np.arange(ND).reshape(1,-1),ND,axis=0)
        return background * (np.arange(ND)>=cutoff)
    window = pg.GraphicsLayoutWidget()
    view = window.addViewBox(title='shadeLeftToRight',row=0,col=0)
    cutoffPlot = window.Plot(title='cutoffValue',row=1,col=0)
    
    view.addItem(genBackground(ND,cutoff=cutoff))
    cutoffLine = cutoffPlot.plot()
    cutoffLine.addData(np.arange(ND),np.arange(ND))
    
    return window,view,cutoffPlot


def napariWithSlider(ND=512,NC=50,cellWidth=2):
    refImage = np.random.normal(0,1,(ND,ND))
    xCellLocations = np.random.randint(0,ND,NC)
    yCellLocations = np.random.randint(0,ND,NC)
    cellGrid = np.zeros((ND,ND,NC))
    cellGrid[yCellLocations,xCellLocations,np.arange(NC)] = 1
    cellMask = ndi.gaussian_filter(cellGrid, (cellWidth, cellWidth, 0))
    
    def convertToImage(cellMask, cellIdx):
        return np.sum(cellMask * cellIdx, axis=2)
    
    def updateCutoff(event):
        cutoffValues = (cutoffVal1.pos()[0], cutoffVal2.pos()[0])
        minCutoff = min(cutoffValues)
        maxCutoff = max(cutoffValues)
        masks.data = convertToImage(cellMask, (xCellLocations>=minCutoff) & (xCellLocations<maxCutoff))
        
    viewer = napari.Viewer()
    reference = viewer.add_image(refImage, name='reference', blending='additive')
    masks = viewer.add_image(convertToImage(cellMask, np.full(NC,True)), name='masks', blending='additive', colormap='red') 
    reference.metadata['count']=1
    
    histCurve = np.sum(convertToImage(cellMask, np.full(NC,True)),axis=0)
    xRange = (0,ND-1)
    window = pg.GraphicsLayoutWidget(title='hello world')
    histPlot = window.addPlot(title="cutoff",row=0,col=0)
    histPlot.addLegend()
    histPlot.setRange(xRange=xRange,yRange=(0,1.1*np.max(histCurve)), update=False)
    histPlot.setMouseEnabled(x=False, y=False)
    cutoffLine = histPlot.plot()
    cutoffLine.setData(np.arange(ND), histCurve) # create profile of cell masks along x axis 
    cutoffVal1 = pg.InfiniteLine(pos=0,movable=True)
    cutoffVal1.setBounds(xRange)
    cutoffVal1.sigPositionChanged.connect(updateCutoff)
    histPlot.addItem(cutoffVal1)
    cutoffVal2 = pg.InfiniteLine(pos=ND-1,movable=True)
    cutoffVal2.setBounds(xRange)
    cutoffVal2.sigPositionChanged.connect(updateCutoff)
    histPlot.addItem(cutoffVal2)
    
    # try push buttons
    def helloFromPush(event):
        reference.data = np.random.normal(0,1,(ND,ND))
        print('hello from push')
        
    buttonProxy = QGraphicsProxyWidget()
    button = QPushButton('button')
    button.setDefault(True)
    button.setCheckable(True)
    button.setChecked(True)
    button.clicked.connect(helloFromPush)
    buttonProxy.setWidget(button)
    window.addItem(buttonProxy,row=1,col=0)    
        
    dockWidget = viewer.window.add_dock_widget(window, name='cutoffSelection', area='bottom')
    
    @viewer.bind_key('r')
    def hello1(viewer):
        x = np.random.randint(0,100)
        viewer.status = f'hello {x}'
        yield
        viewer.status = 'goodbye'

    @reference.mouse_drag_callbacks.append
    def update_layer(layer, event):
        reference.metadata['count']+=1
        print(reference.metadata['count'])
        #layer.data = np.random.normal(0,reference.metadata['count'],layer.data.shape)
    
    return viewer, reference


class locPlot(object):
    def __init__(self, ND=500, NC=50, cellWidth=10):
        self.left = 10
        self.top = 10
        self.title = 'hello world'
        self.width = 320
        self.height = 200
        
        self.refImage = np.random.normal(0,1,(ND,ND))
        self.cellLocations = np.random.randint(0,ND,(2,NC))
        self.cellGrid = np.zeros((NC,ND,ND))
        self.cellGrid[np.arange(NC),self.cellLocations[0],self.cellLocations[1]] = 1
        self.cellMask = ndi.gaussian_filter(self.cellGrid, (0, cellWidth, cellWidth))
        self.tt = np.arange(ND)
        
        self.win = pg.GraphicsLayoutWidget(title=self.title) # create a window
        self.plots = [None] * 2
        self.plots[0] = self.win.addPlot(title="left plot", row=0, col=0)
        self.plots[1] = self.win.addPlot(title="right plot", row=0, col=1)
        
        self.idxPlot = [None] * 2
        for i in range(2):
            self.idxPlot[i] = self.plots[i].plot()
            #self.idxPlot[i].setShadowPen(pg.mkPen((255,255,255), width=2, cosmetic=True))
        
        self.plot(0)
        self.win.show()
        
        
    def plot(self, idx):
        xv = np.zeros_like(self.tt)
        yv = np.zeros_like(self.tt)
        xv[self.cellLocations[0,idx]]=1
        yv[self.cellLocations[1,idx]]=1
        self.idxPlot[0].setData(self.tt, xv)
        self.idxPlot[1].setData(self.tt, yv)

def testLocationPlot():
    viewer = napari.Viewer()

    lp = locPlot()
        
    reference = viewer.add_image(lp.refImage, name='reference', blending='additive')
    masks = viewer.add_image(lp.cellMask, name='masks', blending='additive', colormap='red') 
    
    def update_slider(event):
        idx = event.source.current_step[0]
        lp.plot(idx)
        viewer.status = f"Idx: {idx}"
        
    @viewer.bind_key('q')
    def exit(viewer):
        lp.win.close()
        viewer.close()

    viewer.dims.events.connect(update_slider)
        
    return viewer

def napariTest():
    blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
    viewer = napari.view_image(blobs.astype(float), name='blobs')
    labeled = ndi.label(blobs)[0]
    viewer.add_labels(labeled, name='blob ID')
    return viewer


def basicNapari(ND=500,NC=50,cellWidth=10):
    refImage = np.random.normal(0,1,(ND,ND))
    cellLocations = np.random.randint(0,ND,(2,NC))
    cellGrid = np.zeros((NC,ND,ND))
    cellGrid[np.arange(NC),cellLocations[0],cellLocations[1]] = 1
    cellMask = ndi.gaussian_filter(cellGrid, (0, cellWidth, cellWidth))
    
    viewer = napari.Viewer()
    reference = viewer.add_image(refImage, name='reference', blending='additive')
    masks = viewer.add_image(cellMask, name='masks', blending='additive', colormap='red') 

    widg_dict = {}
    widg_dict['plot_widget1'] = pg.PlotWidget()
    widg_dict['plot_widget1'].addLegend()
    widg_dict['xline1'] = widg_dict['plot_widget1'].plot([0],[0],pen='b',name='xval')
    widg_dict['yline1'] = widg_dict['plot_widget1'].plot([0],[0],pen='r',name='yval') 
    
    widg_dict['plot_widget2'] = pg.PlotWidget()
    widg_dict['plot_widget2'].addLegend()
    widg_dict['xline2'] = widg_dict['plot_widget2'].plot([0],[0],pen='b',name='xval')
    widg_dict['yline2'] = widg_dict['plot_widget2'].plot([0],[0],pen='r',name='yval') 
    
    widg_dict['dock_widget1'] = viewer.window.add_dock_widget(widg_dict['plot_widget1'], name='plot', area='bottom')
    widg_dict['dock_widget2'] = viewer.window.add_dock_widget(widg_dict['plot_widget2'], name='plot', area='bottom')
    
    
    def plotNewLines(idx):
        tt = np.arange(ND)
        xv = np.zeros_like(tt)
        yv = np.zeros_like(tt)
        xv[cellLocations[0,idx]]=1
        yv[cellLocations[1,idx]]=1
        widg_dict['xline1'].setData(tt,xv)
        widg_dict['yline1'].setData(tt,yv)
        widg_dict['xline2'].setData(tt,xv)
        widg_dict['yline2'].setData(tt,yv)
    
    @viewer.bind_key('r')
    def hello1(viewer):
        x = np.random.randint(0,10)
        viewer.status = f'hello {x}'
        yield
        viewer.status = 'goodbye'

    def update_slider(event):
        idx = event.source.current_step[0]
        plotNewLines(idx)
        viewer.status = f"Idx: {idx}"
        
#     def helloWorld():
#         x = np.random.randint(0,1000)
#         viewer.status = f"Hello! {x}"
        
    plotNewLines(0)
    
    viewer.dims.events.connect(update_slider)
    # viewer.dims.events.connect(helloWorld)

    return viewer

#     @cell_layer.mouse_drag_callbacks.append
#     def on_click(cell_labels, event):
#         value = cell_labels.get_value(
#             position=event.position,
#             view_direction=event.view_direction,
#             dims_displayed=event.dims_displayed,
#             world=True)
#         print(value)
#         if value is not None and value > 0:
#             cell_idx = value - 1
#             if event.button == 1:
#                 update_plot(widg_dict, cell_idx)
#             # if event.button == 2:
#             #     mark_cell(
#             #         cell_idx, 0, outputs['iscell'], cell_layer, not_cell_layer)

#     @not_cell_layer.mouse_drag_callbacks.append
#     def on_click(not_cell_labels, event):
#         value = not_cell_labels.get_value(
#             position=event.position,
#             view_direction=event.view_direction,
#             dims_displayed=event.dims_displayed,
#             world=True)
#         print('Not cell,', value)
#         if value is not None and value > 0:
#             cell_idx = value - 1
#             if event.button == 1:
#                 update_plot(widg_dict, cell_idx)
#             # if event.button == 2:
#             #     mark_cell(
#             #         cell_idx, 1, outputs['iscell'], cell_layer, not_cell_layer)

#     return v



# def create_napari_ui(outputs, lam_thresh=0.3, title='3D Viewer', use_patch_coords=False, scale=(15,4,4), theme='dark', extra_cells=None, extra_cells_names=None, vmap_limits=None,
#                      extra_images = None, extra_images_names = None, cell_label_name='cells', vmap_name='corr map', use_filtered_iscell=True):
#     if use_patch_coords:
#         vmap = outputs['vmap_patch']
#     else: 
#         vmap = outputs['vmap']
#     if use_filtered_iscell and 'iscell_filtered' in outputs.keys():
#         iscell = outputs['iscell_filtered']
#     else:
#         iscell = outputs['iscell']
#     if len(iscell.shape) > 1:
#         iscell = iscell[:,0]
#     cell_labels = make_cell_label_vol(outputs['stats'], iscell, vmap.shape,
#                                          lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
#     not_cell_labels = make_cell_label_vol(outputs['stats'], 1-iscell, vmap.shape,
#                                              lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
#     v = napari.view_image(
#         vmap, title=title, name=vmap_name, opacity=1.0, scale=scale, contrast_limits=vmap_limits)
#     if extra_images is not None:
#         for i, extra_image in enumerate(extra_images):
#             v.add_image(
#                 extra_image, name=extra_images_names[i], opacity=1.0, scale=scale)

#     if 'im3d' in outputs.keys():
#         v.add_image(outputs['im3d'], name='Image', scale=scale)
#     cell_layer = v.add_labels(cell_labels, name=cell_label_name, opacity=0.5, scale=scale)

#     if extra_cells is not None:
#         for i, extra_cell in enumerate(extra_cells):
#             extra_cell_labels = make_cell_label_vol(extra_cell, n.ones(len(extra_cell)), vmap.shape,
#                                                     lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
#             v.add_labels(extra_cell_labels,
#                          name=extra_cells_names[i], scale=scale, opacity=0.5)

#     not_cell_layer = v.add_labels(
#         not_cell_labels, name='not-' +cell_label_name, opacity=0.5, scale=scale)
    
#     if 'F' in outputs.keys():
#         if outputs['F'].shape[0] != len(iscell):
#             assert outputs['F'].shape[0] == iscell.sum()
#             trace_idxs = n.cumsum(iscell) - 1
#         else:
#             trace_idxs = n.arange(len(iscell))

#     v.theme = theme
#     widg_dict = {}
#     widg_dict['plot_widget'] = pg.PlotWidget()
#     widg_dict['plot_widget'].addLegend()
#     widg_dict['f_line'] = widg_dict['plot_widget'].plot(
#         [0], [0], pen='b', name='F')
#     widg_dict['fneu_line'] = widg_dict['plot_widget'].plot(
#         [0], [0], pen='r', name='Npil')
#     widg_dict['spks_line'] = widg_dict['plot_widget'].plot(
#         [0], [0], pen='w', name='Deconv')
#     widg_dict['dock_widget'] = v.window.add_dock_widget(
#         widg_dict['plot_widget'], name='activity', area='bottom')



#     def get_traces(cell_idx):
#         trace_idx = trace_idxs[cell_idx]
#         fx = outputs['F'][trace_idx]
#         fn = outputs['Fneu'][trace_idx]
#         ss = outputs['spks'][trace_idx]
#         return outputs['ts'], fx, fn, ss

#     def update_plot(widg_dict, cell_idx):
#         ts, fx, fn, ss = get_traces(cell_idx)
#         widg_dict['f_line'].setData(ts, fx)
#         widg_dict['fneu_line'].setData(ts, fn)
#         widg_dict['spks_line'].setData(ts, ss)

#     @cell_layer.mouse_drag_callbacks.append
#     def on_click(cell_labels, event):
#         value = cell_labels.get_value(
#             position=event.position,
#             view_direction=event.view_direction,
#             dims_displayed=event.dims_displayed,
#             world=True)
#         print(value)
#         if value is not None and value > 0:
#             cell_idx = value - 1
#             if event.button == 1:
#                 update_plot(widg_dict, cell_idx)
#             # if event.button == 2:
#             #     mark_cell(
#             #         cell_idx, 0, outputs['iscell'], cell_layer, not_cell_layer)

#     @not_cell_layer.mouse_drag_callbacks.append
#     def on_click(not_cell_labels, event):
#         value = not_cell_labels.get_value(
#             position=event.position,
#             view_direction=event.view_direction,
#             dims_displayed=event.dims_displayed,
#             world=True)
#         print('Not cell,', value)
#         if value is not None and value > 0:
#             cell_idx = value - 1
#             if event.button == 1:
#                 update_plot(widg_dict, cell_idx)
#             # if event.button == 2:
#             #     mark_cell(
#             #         cell_idx, 1, outputs['iscell'], cell_layer, not_cell_layer)

#     return v


def testHistSlider(ND=512,minCutoff=0,maxCutoff=None):
    # generate image and a cutoff control with infinite lines that are draggable and update which part of the image are shown
    maxCutoff = ND
    def genBackground(ND,minCutoff=0,maxCutoff=maxCutoff):
        background = np.repeat(np.arange(ND).reshape(1,-1),ND,axis=0)
        return (background * (minCutoff<=np.arange(ND)) * (np.arange(ND)<maxCutoff)).T
    
    def mouseClickEvent(event):
        if event.button()==1 and cutoffPlot.sceneBoundingRect().contains(event.scenePos()):
            mouse_point = cutoffPlot.vb.mapSceneToView(event.scenePos())
            print(f"X:{mouse_point.x()}, Y:{mouse_point.y()}")
            selectionPoint.setData([mouse_point.x()],[mouse_point.y()])
            
    def viewClickEvent(event):
        if event.button()==1 and view.sceneBoundingRect().contains(event.scenePos()):
            print('hello world')
            
    def checkLineEvents(event):
        cutoffValues = (cutVal1.pos()[0], cutVal2.pos()[0])
        minCutoff = min(cutoffValues)
        maxCutoff = max(cutoffValues)
        img.setImage(genBackground(ND,minCutoff=minCutoff,maxCutoff=maxCutoff))
        img.setLevels([0,ND], update=False)
    
    xRange = (0,ND)
    yRange = (0,ND)
    window = pg.GraphicsLayoutWidget()
    view = window.addViewBox(row=0,col=0,enableMouse=False)
    cutoffPlot = window.addPlot(title='cutoffValue',row=1,col=0)
    cutoffPlot.setRange(xRange=xRange,yRange=yRange)
    cutoffPlot.setMouseEnabled(x=False, y=False)
    cutoffPlot.scene().sigMouseClicked.connect(mouseClickEvent)
    view.scene().sigMouseClicked.connect(viewClickEvent)
    
    img = pg.ImageItem(image=genBackground(ND,minCutoff=minCutoff,maxCutoff=maxCutoff))
    img.setLevels([0,ND], update=False)
    
    view.addItem(img)
    cutoffLine = cutoffPlot.plot()
    cutoffLine.setData(np.arange(ND),np.arange(ND))
    cutVal1 = pg.InfiniteLine(pos=0,movable=True)
    cutVal1.setBounds(xRange)
    cutVal1.sigPositionChanged.connect(checkLineEvents)
    cutoffPlot.addItem(cutVal1)
    cutVal2 = pg.InfiniteLine(pos=ND-1,movable=True)
    cutVal2.setBounds(xRange)
    cutVal2.sigPositionChanged.connect(checkLineEvents)
    cutoffPlot.addItem(cutVal2)
    
    window.show()
    return window,view,cutoffPlot,cutVal1