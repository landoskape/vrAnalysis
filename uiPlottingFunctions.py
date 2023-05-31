import numpy as np
import scipy as sp
import napari
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsProxyWidget, QSlider, QPushButton, QVBoxLayout, QLabel

def scrollMatchedImages(stacks, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    assert isinstance(stacks, (list, tuple)) and len(stacks)>1, "stacks must be a tuple of at least 2 3-d image stacks"
    assert type(enableMouse)==bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    for stack in range(numStacks): assert stacks[stack].ndim==3, "stacks are not all 3-dimensional"
    for stack in range(numStacks): assert stacks[stack].shape[0]==numImages,  "number of images to look through are not the same in each stack"
    
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
        
    # Create graphics layout with two viewboxes for the image stacks
    window = pg.GraphicsLayoutWidget()
    views = [window.addViewBox(row=0,col=stack,enableMouse=enableMouse,lockAspect=lockAspect,invertY=True) for stack in range(numStacks)]
    for image,view in zip(imageItems, views): view.addItem(image)
    for view in views[1:]: 
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])
        
    if infLines:
        for (ixLine,iyLine,view) in zip(ixLineItems,iyLineItems,views):
            view.addItem(ixLine)
            view.addItem(iyLine)
    
    sliderNameProxy = QGraphicsProxyWidget()
    sliderName = QVBoxLayout()
    label = QLabel(f"Image {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy,row=1,col=0,colspan=numStacks)
    
    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages-1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages/10))
    slider.setValue(0)
    slider.valueChanged.connect(updateStackIndex)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)
    window.addItem(sliderProxy,row=2,col=0,colspan=numStacks)
    
    window.show()
    
    return window
    