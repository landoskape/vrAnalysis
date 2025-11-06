# Standard modules
import numpy as np
import pandas as pd

# GUI-related modules
import napari
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QGraphicsProxyWidget,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence

# Special vrAnalysis modules
from .. import helpers

pd.options.display.width = 1000


# converting uiPlottingFunctions.scrollMatchedImages into a redSelection GUI made to be similar to the same named function in Matlab
def redCellViewer(stacks, features, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    # supporting class for storing and updating the ROI displayed in redCellViewer()
    class currentROI:
        def __init__(self, minroi=0, maxroi=None):
            self.minroi = minroi
            self.maxroi = maxroi if maxroi is not None else np.inf
            assert self.minroi < self.maxroi, "minimum roi value must be less than maximum roi value"
            self.value = 0

        def update(self, value):
            self.value = np.minimum(np.maximum(value, self.minroi), self.maxroi)

    # handle inputs
    msg = "stacks should be a length 3 iterable containing 3-d centered stacks of the reference image, the mask, and the phase correlation plots"
    assert len(stacks) == 3, msg
    msg = "features should be a length 4 iterable containing the suite2p red probability, dot product, correlation coefficient, and central pxc point for each ROI in stacks"
    assert len(features) == 4, msg
    assert type(enableMouse) == bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    numFeatures = len(features)
    for stack in range(numStacks):
        assert stacks[stack].ndim == 3, "stacks are not all 3-dimensional"
    for stack in range(numStacks):
        assert stacks[stack].shape[0] == numImages, "number of ROIs to look through are not the same in each stack"
    for feature in features:
        assert isinstance(feature, np.ndarray), "features need to be numpy arrays"
    for feature in features:
        assert feature.size == numImages, "number of ROIs in each features array must be same as the number of ROIs in the stacks"
    stackTitles = ["reference", "mask", "phase-correlation"]
    featureTitles = ["S2P", "dot(ref,mask)", "corr(ref,mask)", "pxc"]

    # keep track of current ROI (I have  no idea why I can't do this with a python int...
    roi = currentROI(minroi=0, maxroi=numImages - 1)

    # measure minimum and maximum of each stack
    if preserveScale:
        imLevels = [(np.min(stack), np.max(stack)) for stack in stacks]
    else:
        imLevels = [None] * numStacks  # allocate list for simple code later on

    def updateStackIndex():
        # whenever the ROI is changed, update the images and the labels (and keep scale the same if necessary)
        for stack, image, imLevel, view in zip(stacks, imageItems, imLevels, views):
            image.setImage(stack[roi.value])
            label.setText(f"ROI {roi.value+1}/{numImages}")
            if preserveScale:
                image.setLevels(imLevel)
        # whenever the ROI is changed, update the infiniteLine position indicating the value of that particular ROI
        for feature, cvalROI in zip(features, currentValueROI):
            cvalROI.setValue(feature[roi.value])

    # create image items for each stack
    imageItems = [pg.ImageItem(image=stacks[stack][0], axisOrder="row-major") for stack in range(numStacks)]
    if preserveScale:
        for imLevel, image in zip(imLevels, imageItems):
            image.setLevels(imLevel)

    # infLines are drawn over the stacks to help find the same position across stacks, they are linked across stacks.
    if infLines:

        def updateLinePosX(event):
            for ixLine in ixLineItems:
                ixLine.setValue(event.x())

        def updateLinePosY(event):
            for iyLine in iyLineItems:
                iyLine.setValue(event.y())

        # start with the lines in the center (0,0) position
        xPosition = stacks[0].shape[2] / 2
        yPosition = stacks[0].shape[1] / 2
        # create the lines, and add callbacks
        ixLineItems = [pg.InfiniteLine(pos=xPosition, angle=90, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        iyLineItems = [pg.InfiniteLine(pos=yPosition, angle=0, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        for ixLine, iyLine in zip(ixLineItems, iyLineItems):
            ixLine.sigPositionChangeFinished.connect(updateLinePosX)
            iyLine.sigPositionChangeFinished.connect(updateLinePosY)

    # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
    window = pg.GraphicsLayoutWidget(size=(1200, 800))

    # Create graphics layout with viewboxes for the image stacks
    stackLayout = pg.GraphicsLayout()
    window.addItem(stackLayout, row=1, col=0)
    # create a viewbox for each stack, add the appropriate image to it, link images so they all move together
    views = [
        stackLayout.addViewBox(
            row=0,
            col=stack,
            enableMouse=enableMouse,
            lockAspect=lockAspect,
            invertY=True,
            name=stackTitles[stack],
        )
        for stack in range(numStacks)
    ]
    for image, view in zip(imageItems, views):
        view.addItem(image)
    for view in views[1:]:
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])
    # add infinite lines to mark positions if requested
    if infLines:
        for ixLine, iyLine, view in zip(ixLineItems, iyLineItems, views):
            view.addItem(ixLine)
            view.addItem(iyLine)

    # Create barplots for each feature
    histCenters, histValues, histRange = [], [], []
    for feature in features:
        # make histogram of each feature
        cHist, cEdges = np.histogram(feature, bins=50)
        histRange.append((cEdges[0], cEdges[-1]))  # min/max of histogram for each feature
        histCenters.append(helpers.edge2center(cEdges))  # center of histogram bin for each feature
        histValues.append(cHist)  # histogram value for each bin for each feature
    featureHistograms = [
        pg.BarGraphItem(x=histCenter, height=histValue, width=histCenter[1] - histCenter[0]) for histCenter, histValue in zip(histCenters, histValues)
    ]
    featRedHistograms = [
        pg.BarGraphItem(x=histCenter, height=histValue / 2, width=histCenter[1] - histCenter[0], brush="r")
        for histCenter, histValue in zip(histCenters, histValues)
    ]

    # Create a graphics layout with bar graph plots for the features
    featureLayout = pg.GraphicsLayout()
    window.addItem(featureLayout, row=2, col=0)
    featurePlots = [featureLayout.addPlot(row=0, col=feature, enableMouse=False, title=featureTitles[feature]) for feature in range(numFeatures)]
    for featurePlot in featurePlots:
        featurePlot.setMouseEnabled(x=False, y=False)
    # featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    for featureHistogram, featurePlot in zip(featureHistograms, featurePlots):
        featurePlot.addItem(featureHistogram)
    # for featureHistogram,featurePlot in zip(featRedHistograms, featurePlots): featurePlot.addItem(featureHistogram)

    # Create vertical lines indicating the value of the currently presented cell
    currentValueROI = [pg.InfiniteLine(pos=features[feature][0], angle=90, movable=False, pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    for fplot, cv in zip(featurePlots, currentValueROI):
        fplot.addItem(cv)

    # Create a slider label for indicating which ROI is being presented
    sliderNameProxy = QGraphicsProxyWidget()
    label = QLabel(f"ROI {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy, row=3, col=0)

    # Create a slider with prev/next buttons and an edit field to change which ROI is being presented
    def updateSlider(value):
        roi.update(value)  # first try updating roi value
        slider.setValue(roi.value)  # if it clipped, reset slider appropriately
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def prevROI():
        roi.update(roi.value - 1)  # try updating roi value
        slider.setValue(roi.value)  # update slider
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def nextROI():
        roi.update(roi.value + 1)  # try updating roi value
        slider.setValue(roi.value)  # update slider
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def gotoROI():
        if not editField.text().isdigit():
            editField.setText("invalid ROI")
            return
        textValue = int(editField.text())
        if (textValue < roi.minroi) or (textValue > roi.maxroi):
            editField.setText("invalid ROI")
            return
        # otherwise text is valid ROI
        roi.update(textValue)
        editField.setText(str(roi.value))
        slider.setValue(roi.value)  # update slider
        updateStackIndex()

    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages - 1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages / 10))
    slider.setValue(roi.value)
    slider.valueChanged.connect(updateSlider)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)

    prevButtonProxy = QGraphicsProxyWidget()
    prevButton = QPushButton("button", text="Prev ROI")
    prevButton.clicked.connect(prevROI)
    prevButtonProxy.setWidget(prevButton)

    nextButtonProxy = QGraphicsProxyWidget()
    nextButton = QPushButton("button", text="Next ROI")
    nextButton.clicked.connect(nextROI)
    nextButtonProxy.setWidget(nextButton)

    editFieldProxy = QGraphicsProxyWidget()
    editField = QLineEdit()
    editField.setText("0")
    editFieldProxy.setWidget(editField)

    gotoEditProxy = QGraphicsProxyWidget()
    gotoButton = QPushButton("button", text="go to ROI")
    gotoButton.clicked.connect(gotoROI)
    gotoEditProxy.setWidget(gotoButton)

    # add shortcut for going to ROI without pressing the button...
    shortcut = QShortcut(QKeySequence("G"), window)
    shortcut.activated.connect(gotoROI)

    roiSelectionLayout = pg.GraphicsLayout()
    roiSelectionLayout.addItem(prevButtonProxy, row=0, col=0)
    roiSelectionLayout.addItem(sliderProxy, row=0, col=1)
    roiSelectionLayout.addItem(nextButtonProxy, row=0, col=2)
    roiSelectionLayout.addItem(editFieldProxy, row=0, col=3)
    roiSelectionLayout.addItem(gotoEditProxy, row=0, col=4)
    window.addItem(roiSelectionLayout, row=4, col=0)

    # show GUI and return window for programmatic interaction
    window.show()
    return window


# converting uiPlottingFunctions.scrollMatchedImages into a redSelection GUI made to be similar to the same named function in Matlab
def redSelectionAmorphous(stacks, features, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    assert isinstance(stacks, (list, tuple)) and len(stacks) > 1, "stacks must be a tuple of at least 2 3-d image stacks"
    assert isinstance(features, (list, tuple)), "features must be a list or tuple (even if it only has one element...)"
    assert type(enableMouse) == bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    numFeatures = len(features)
    for stack in range(numStacks):
        assert stacks[stack].ndim == 3, "stacks are not all 3-dimensional"
    for stack in range(numStacks):
        assert stacks[stack].shape[0] == numImages, "number of images to look through are not the same in each stack"
    for feature in features:
        assert isinstance(feature, np.ndarray), "features need to be numpy arrays"
    for feature in features:
        assert feature.size == numImages, "number of values in each features must be same as number of images in each stack"

    # measure minimum and maximum of each stack
    if preserveScale:
        imLevels = [(np.min(stack), np.max(stack)) for stack in stacks]
    else:
        imLevels = [None] * numStacks  # allocate list for simple code later on

    def updateStackIndex(value):
        for stack, image, imLevel, view in zip(stacks, imageItems, imLevels, views):
            image.setImage(stack[value])
            label.setText(f"Image {value+1}/{numImages}")
            if preserveScale:
                image.setLevels(imLevel)
        for feature, cvalROI in zip(features, currentValueROI):
            cvalROI.setValue(feature[value])

    # create image items for each stack
    imageItems = [pg.ImageItem(image=stacks[stack][0], axisOrder="row-major") for stack in range(numStacks)]
    if preserveScale:
        for imLevel, image in zip(imLevels, imageItems):
            image.setLevels(imLevel)

    if infLines:

        def updateLinePosX(event):
            for ixLine in ixLineItems:
                ixLine.setValue(event.x())

        def updateLinePosY(event):
            for iyLine in iyLineItems:
                iyLine.setValue(event.y())

        xPosition = stacks[0].shape[2] / 2
        yPosition = stacks[0].shape[1] / 2
        ixLineItems = [pg.InfiniteLine(pos=xPosition, angle=90, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        iyLineItems = [pg.InfiniteLine(pos=yPosition, angle=0, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        for ixLine, iyLine in zip(ixLineItems, iyLineItems):
            ixLine.sigPositionChangeFinished.connect(updateLinePosX)
            iyLine.sigPositionChangeFinished.connect(updateLinePosY)

    # Create graphics layout with viewboxes for the image stacks
    window = pg.GraphicsLayoutWidget()
    stackLayout = pg.GraphicsLayout()
    window.addItem(stackLayout, row=0, col=0)
    views = [stackLayout.addViewBox(row=0, col=stack, enableMouse=enableMouse, lockAspect=lockAspect, invertY=True) for stack in range(numStacks)]
    for image, view in zip(imageItems, views):
        view.addItem(image)
    for view in views[1:]:
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])

    if infLines:
        for ixLine, iyLine, view in zip(ixLineItems, iyLineItems, views):
            view.addItem(ixLine)
            view.addItem(iyLine)

    # Create barplots for each feature
    histCenters, histValues, histRange = [], [], []
    for feature in features:
        # make histogram of each feature
        cHist, cEdges = np.histogram(feature, bins=100)
        histRange.append((cEdges[0], cEdges[-1]))  # min/max of histogram for each feature
        histCenters.append(helpers.edge2center(cEdges))  # center of histogram bin for each feature
        histValues.append(cHist)  # histogram value for each bin for each feature
    featureHistograms = [
        pg.BarGraphItem(x=histCenter, height=histValue, width=histCenter[1] - histCenter[0]) for histCenter, histValue in zip(histCenters, histValues)
    ]

    # Create a graphics layout with bar graph plots for the features
    featureLayout = pg.GraphicsLayout()
    window.addItem(featureLayout, row=1, col=0)
    featurePlots = [featureLayout.addPlot(row=0, col=feature, enableMouse=False) for feature in range(numFeatures)]
    for featurePlot in featurePlots:
        featurePlot.setMouseEnabled(x=False, y=False)
    # featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    for featureHistogram, featurePlot in zip(featureHistograms, featurePlots):
        featurePlot.addItem(featureHistogram)

    # Create vertical lines indicating the value of the currently presented cell
    currentValueROI = [pg.InfiniteLine(pos=features[feature][0], angle=90, movable=False, pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    for fplot, cv in zip(featurePlots, currentValueROI):
        fplot.addItem(cv)

    print(hasattr(featurePlots[0], "getAxis"))

    # Create a slider for selecting which slice of the image stacks to look at
    sliderNameProxy = QGraphicsProxyWidget()
    sliderName = QVBoxLayout()
    label = QLabel(f"Image {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy, row=2, col=0)

    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages - 1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages / 10))
    slider.setValue(0)
    slider.valueChanged.connect(updateStackIndex)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)
    window.addItem(sliderProxy, row=3, col=0)

    window.show()

    return window
