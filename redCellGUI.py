import numpy as np
from scipy import ndimage as ndi
from skimage import data
import napari
import pyqtgraph as pg
import matplotlib.pyplot as pl



# -- I copied everything from testingNapari.py below because it'll probably be useful!



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