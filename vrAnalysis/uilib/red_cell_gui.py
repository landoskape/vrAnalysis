# Standard modules
from copy import copy
import functools
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# GUI-related modules
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QGraphicsProxyWidget,
    QPushButton,
)
import napari
from napari.utils.colormaps import label_colormap, direct_colormap
import pyqtgraph as pg

# Special vrAnalysis modules
from ..registration.redcell import RedCellProcessing
from .. import helpers
from .. import database

pd.options.display.width = 1000


def compare_feature_cutoffs(*rcps: RedCellProcessing, round_value: Optional[int] = None) -> None:
    """
    Compare feature cutoffs across multiple sessions.

    Parameters
    ----------
    *rcps : RedCellProcessing
        Variable number of RedCellProcessing objects to compare.
    round_value : Optional[int], optional
        Number of decimal places to round cutoff values. If None, no rounding
        is performed.
    """
    features = [
        "parametersRedS2P.minMaxCutoff",
        "parametersRedDotProduct.minMaxCutoff",
        "parametersRedPearson.minMaxCutoff",
        "parametersRedPhaseCorrelation.minMaxCutoff",
    ]
    df_dict = {"session": [red_cell.b2session.session_print() for red_cell in rcps]}

    def get_feat_name(name):
        cname = name[name.find("Red") + 3 : name.find(".")]
        return cname

    for feat in features:
        df_dict[get_feat_name(feat)] = [None] * len(rcps)

    for idx, red_cell in enumerate(rcps):
        for feat in features:
            cdata = red_cell.b2session.loadone(feat)
            if cdata.dtype == object and cdata.item() is None:
                cdata = [None, None]
            else:
                if round_value is not None:
                    cdata = np.round(cdata, round_value)
            df_dict[get_feat_name(feat)][idx] = cdata

    print(pd.DataFrame(df_dict))


BASIC_BUTTON_STYLE = """
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

Q_CHECKED_STYLE = """
QWidget {
    background-color: #1F1F1F;
    color: red;
    font-family: Arial, sans-serif;
}
"""

Q_NOT_CHECKED_STYLE = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}
"""


class RedSelectionGUI:
    """
    GUI for red cell selection and curation using napari.

    This class provides an interactive interface for curating red cell
    identifications based on multiple features (S2P, dot product, Pearson
    correlation, phase correlation). Users can adjust feature cutoffs,
    manually label cells, and visualize results in napari.

    Parameters
    ----------
    red_cell : RedCellProcessing
        RedCellProcessing object containing the session data and red cell
        features.
    num_bins : int, optional
        Number of bins for feature histograms. Default is 50.
    init_yzoom : float, optional
        Initial y-axis zoom level for feature histograms. If None, uses
        maximum histogram value. Default is None.

    Attributes
    ----------
    red_cell : RedCellProcessing
        The RedCellProcessing object containing session data.
    num_planes : int
        Number of imaging planes in the session.
    roi_per_plane : np.ndarray
        Number of ROIs per plane.
    num_bins : int
        Number of bins for histograms.
    plane_idx : int
        Currently selected plane index.
    feature_names : list of str
        Names of features used for red cell identification.
    """

    def __init__(self, red_cell: RedCellProcessing, num_bins: int = 50, init_yzoom: float = None):
        assert isinstance(red_cell, RedCellProcessing), "red_cell must be an instance of RedCellProcessing"
        self.red_cell = red_cell
        self.num_planes = self.red_cell.num_planes
        self.roi_per_plane = self.red_cell.b2session.get_value("roiPerPlane")
        self.num_bins = num_bins
        self.init_yzoom = init_yzoom
        # keep track of which plane to observe
        self.plane_idx = 0

        self.ref_image = [None] * self.num_planes
        self.idx_roi = [None] * self.num_planes
        self.feature_names = self.red_cell.feature_names
        self.num_features = len(self.feature_names)
        self.feature_active = [[True, True] for _ in range(self.num_features)]
        self.features = [None] * self.num_planes
        self.hvalues = [None] * self.num_planes
        self.hvalred = [None] * self.num_planes
        self.hedges = [None] * self.num_features

        # process initial plane
        # If true, then self.mask_image() will display control cells rather than red cells
        self.control_cell_toggle = False
        # start with all as red...
        self.red_idx = [np.full(self.roi_per_plane[plane_idx], True) for plane_idx in range(self.num_planes)]
        self.manual_label = [None] * self.num_planes
        self.manual_label_active = [None] * self.num_planes
        # compute reference / maskVolume / featureArrays for each plane
        self.process_planes()

        # open napari viewer and associated GUI features
        # if true, will show mask image, if false, will show mask labels
        self.show_mask_image = False
        # if true, will show either mask image or label, otherwise will not show either!
        self.mask_visibility = True
        # if true, then will apply manual labels after using features to compute red_idx
        self.use_manual_label = True
        # if true, only show manual labels of selected category...
        self.only_manual_labels = False
        # indicates which color to display mask_labels (0:random, 1-4:color by feature)
        self.color_state = 0
        # which colormap to use for pseudo coloring the masks
        self.idx_colormap = 0
        self.list_colormaps = ["plasma", "autumn", "spring", "summer", "winter", "hot"]
        self.initialize_napari_viewer()

    def initialize_napari_viewer(self):
        """
        Initialize the napari viewer and all GUI components.

        Creates the napari viewer with reference images, mask images, and labels.
        Sets up feature histograms, cutoff lines, toggle buttons, and control
        buttons for red cell curation.
        """
        # generate napari viewer
        self.viewer = napari.Viewer(title=f"Red Cell Curation from session: {self.red_cell.b2session.session_print()}")
        self.reference = self.viewer.add_image(np.stack(self.red_cell.reference), name="reference", blending="additive", opacity=0.6)
        self.masks = self.viewer.add_image(
            self.mask_image(),
            name="masksImage",
            blending="additive",
            colormap="red",
            visible=self.show_mask_image,
        )
        self.labels = self.viewer.add_labels(
            self.mask_labels(),
            name="maskLabels",
            blending="additive",
            visible=not (self.show_mask_image),
        )
        # [0] = self.plane_idx
        self.viewer.dims.current_step = (
            self.plane_idx,
            self.viewer.dims.current_step[1],
            self.viewer.dims.current_step[2],
        )

        # create feature and button widget
        self.feature_window = pg.GraphicsLayoutWidget()

        # create components of the feature window (the top row is a sequence of histograms, the bottom row is some buttons and edit fields etc.)
        self.toggle_area = pg.GraphicsLayout()
        self.plot_area = pg.GraphicsLayout()
        self.button_area = pg.GraphicsLayout()
        self.feature_window.addItem(self.toggle_area, row=0, col=0)
        self.feature_window.addItem(self.plot_area, row=1, col=0)
        self.feature_window.addItem(self.button_area, row=2, col=0)

        # start by making a specific layout for the histograms of the features
        self.hist_layout = pg.GraphicsLayout()
        self.hist_graphs = [None] * self.num_features
        self.hist_reds = [None] * self.num_features
        for feature in range(self.num_features):
            bar_width = np.diff(self.hedges[feature][:2])
            self.hist_graphs[feature] = pg.BarGraphItem(
                x=helpers.edge2center(self.hedges[feature]),
                height=self.hvalues[self.plane_idx][feature],
                width=bar_width,
            )
            self.hist_reds[feature] = pg.BarGraphItem(
                x=helpers.edge2center(self.hedges[feature]),
                height=self.hvalred[self.plane_idx][feature],
                width=bar_width,
                brush="r",
            )

        # keep y-range of feature plots in useful regime
        def preserve_y_range_0():
            preserve_y_range(0)

        def preserve_y_range_1():
            preserve_y_range(1)

        def preserve_y_range_2():
            preserve_y_range(2)

        def preserve_y_range_3():
            preserve_y_range(3)

        # make independent callbacks for each so it's possible to disconnect and reconnnect them
        # (this doesn't work with functools.partial, although that would be more elegant)
        preserve_methods = [preserve_y_range_0, preserve_y_range_1, preserve_y_range_2, preserve_y_range_3]

        def preserve_y_range(idx):
            # remove callback so we can update the yrange without a recursive call
            self.hist_plots[idx].getViewBox().sigYRangeChanged.disconnect(preserve_methods[idx])
            # then figure out the current y range (this is after a user update)
            current_min, current_max = self.hist_plots[idx].viewRange()[1]
            # set the new max to not exceed the current maximum
            current_range = current_max - current_min
            current_max = min(current_range, self.hvalues_maximum[idx])
            # range is from 0 to the max, therefore the y=0 line always stays in the same place
            self.hist_plots[idx].setYRange(0, current_max)
            # reconnect callback for next update
            self.hist_plots[idx].getViewBox().sigYRangeChanged.connect(preserve_methods[idx])

        # add bargraphs to plotArea
        self.hist_plots = [None] * self.num_features
        for feature in range(self.num_features):
            self.hist_plots[feature] = self.plot_area.addPlot(row=0, col=feature, title=self.feature_names[feature])
            self.hist_plots[feature].setMouseEnabled(x=False)
            # allow user to optionally initialize y zoom to be zoomed
            if self.init_yzoom is not None:
                self.hist_plots[feature].setYRange(0, self.init_yzoom)
            else:
                self.hist_plots[feature].setYRange(0, self.hvalues_maximum[feature])
            self.hist_plots[feature].addItem(self.hist_graphs[feature])
            self.hist_plots[feature].addItem(self.hist_reds[feature])
            # preserve_y_range
            self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(preserve_methods[feature])

        # create cutoffLines (vertical infinite lines) for determining the range within feature values that qualify as red
        def update_cutoff_finished(event, feature):
            cutoff_values = [
                self.cutoff_lines[feature][0].pos()[0],
                self.cutoff_lines[feature][1].pos()[0],
            ]
            min_cutoff, max_cutoff = min(cutoff_values), max(cutoff_values)
            self.feature_cutoffs[feature][0] = min_cutoff
            self.feature_cutoffs[feature][1] = max_cutoff
            self.cutoff_lines[feature][0].setValue(min_cutoff)
            self.cutoff_lines[feature][1].setValue(max_cutoff)
            self.update_red_idx()

        self.feature_range = [None] * self.num_features
        self.feature_cutoffs = [None] * self.num_features
        self.cutoff_lines = [None] * self.num_features
        for feature in range(self.num_features):
            self.feature_range[feature] = [
                np.min(self.hedges[feature]),
                np.max(self.hedges[feature]),
            ]
            # initialize to range
            self.feature_cutoffs[feature] = copy(self.feature_range[feature])
            # check if feature cutoffs have been created and stored already, if so, use them
            cutoff_name = self.red_cell.one_name_feature_cutoffs(self.feature_names[feature])
            if cutoff_name in self.red_cell.b2session.print_saved_one():
                c_feature_cutoff = self.red_cell.b2session.loadone(cutoff_name)
                self.feature_cutoffs[feature] = c_feature_cutoff
                if np.isnan(c_feature_cutoff[0]):
                    self.feature_active[feature][0] = False
                    self.feature_cutoffs[feature][0] = self.feature_range[feature][0]
                if np.isnan(c_feature_cutoff[1]):
                    self.feature_active[feature][1] = False
                    self.feature_cutoffs[feature][1] = self.feature_range[feature][1]
            # one for minimum, one for maximum
            self.cutoff_lines[feature] = [None] * 2
            for ii in range(2):
                if self.feature_active[feature][ii]:
                    self.cutoff_lines[feature][ii] = pg.InfiniteLine(pos=self.feature_cutoffs[feature][ii], movable=True)
                else:
                    self.cutoff_lines[feature][ii] = pg.InfiniteLine(pos=self.feature_range[feature][ii], movable=False)
                self.cutoff_lines[feature][ii].setBounds(self.feature_range[feature])
                self.cutoff_lines[feature][ii].sigPositionChangeFinished.connect(functools.partial(update_cutoff_finished, feature=feature))
                self.hist_plots[feature].addItem(self.cutoff_lines[feature][ii])

        # once cutoff lines are established, reset red_idx to prevent silly behavior
        self.update_red_idx()

        # ---------------------
        # -- now add toggles --
        # ---------------------
        min_max_name = ["min", "max"]
        max_length_name = max([len(name) for name in self.feature_names]) + 9

        def toggle_feature(event, name, idx, minmax):
            # set feature active based on whether toggle is checked
            self.feature_active[idx][minmax] = self.use_feature_buttons[idx][minmax].isChecked()
            if self.feature_active[idx][minmax]:
                # if feature is active, set value to cutoffs and make infinite line movable
                text_to_use = f"using {min_max_name[minmax]} {name}".center(max_length_name, " ")
                self.cutoff_lines[idx][minmax].setValue(self.feature_cutoffs[idx][minmax])
                self.cutoff_lines[idx][minmax].setMovable(True)
                self.use_feature_buttons[idx][minmax].setText(text_to_use)
                self.use_feature_buttons[idx][minmax].setStyleSheet(Q_NOT_CHECKED_STYLE)
            else:
                # if feature isn't active, set value to bounds and make infinite line unmovable
                text_to_use = f"ignore {min_max_name[minmax]} {name}".center(max_length_name, " ")
                self.cutoff_lines[idx][minmax].setValue(self.feature_range[idx][minmax])
                self.cutoff_lines[idx][minmax].setMovable(False)
                self.use_feature_buttons[idx][minmax].setText(text_to_use)
                self.use_feature_buttons[idx][minmax].setStyleSheet(Q_CHECKED_STYLE)

            # then update red idx, which'll replot everything
            self.update_red_idx()

        self.use_feature_buttons = [[None, None] for _ in range(self.num_features)]
        self.use_feature_buttons_proxy = [None] * (self.num_features * 2)
        for featidx, featname in enumerate(self.feature_names):
            for i, name in enumerate(min_max_name):
                proxy_idx = 2 * featidx + i
                if self.feature_active[featidx][i]:
                    text_to_use = f"using {min_max_name[i]} {featname}".center(max_length_name, " ")
                    style_to_use = Q_NOT_CHECKED_STYLE
                else:
                    text_to_use = f"ignore {min_max_name[i]} {featname}".center(max_length_name, " ")
                    style_to_use = Q_CHECKED_STYLE
                self.use_feature_buttons[featidx][i] = QPushButton("toggle", text=text_to_use)
                self.use_feature_buttons[featidx][i].setCheckable(True)
                self.use_feature_buttons[featidx][i].setChecked(self.feature_active[featidx][i])
                self.use_feature_buttons[featidx][i].clicked.connect(functools.partial(toggle_feature, name=featname, idx=featidx, minmax=i))
                self.use_feature_buttons[featidx][i].setStyleSheet(style_to_use)
                self.use_feature_buttons_proxy[proxy_idx] = QGraphicsProxyWidget()
                self.use_feature_buttons_proxy[proxy_idx].setWidget(self.use_feature_buttons[featidx][i])
                self.toggle_area.addItem(self.use_feature_buttons_proxy[proxy_idx], row=0, col=proxy_idx)

        # ---------------------
        # -- now add buttons --
        # ---------------------

        # create save button
        def save_rois(event):
            self.save_selection()

        self.save_button = QPushButton("button", text="save red selection")
        self.save_button.clicked.connect(save_rois)
        self.save_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.save_button_proxy = QGraphicsProxyWidget()
        self.save_button_proxy.setWidget(self.save_button)

        # create update database button
        def update_database(event):
            self.update_database(True)

        self.update_database_button = QPushButton("button", text="update database (QC=True)")
        self.update_database_button.clicked.connect(update_database)
        self.update_database_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.update_database_button_proxy = QGraphicsProxyWidget()
        self.update_database_button_proxy.setWidget(self.update_database_button)

        # create update database button
        def update_database_false(event):
            self.update_database(False)

        self.update_database_false_button = QPushButton("button", text="update database (QC=False)")
        self.update_database_false_button.clicked.connect(update_database_false)
        self.update_database_false_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.update_database_false_button_proxy = QGraphicsProxyWidget()
        self.update_database_false_button_proxy.setWidget(self.update_database_false_button)

        # add toggle control/red cell button
        def toggle_cells_to_view(input_argument):
            # changes whether to plot control or red cells (maybe add a textbox and update it so as to not depend on looking at the print outputs...)
            self.control_cell_toggle = not (self.control_cell_toggle)
            self.toggle_cell_button.setText("control cells" if self.control_cell_toggle else "red cells")
            self.masks.data = self.mask_image()
            self.labels.data = self.mask_labels()

        self.toggle_cell_button = QPushButton(text="control cells" if self.control_cell_toggle else "red cells")
        self.toggle_cell_button.clicked.connect(toggle_cells_to_view)
        self.toggle_cell_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.toggle_cell_button_proxy = QGraphicsProxyWidget()
        self.toggle_cell_button_proxy.setWidget(self.toggle_cell_button)

        # add button to toggle whether to include manual labels in mask plot
        def use_manual_label(event):
            self.use_manual_label = not (self.use_manual_label)
            self.use_manual_label_button.setText("using manual labels" if self.use_manual_label else "ignoring manual labels")
            # update replot masks and recompute histograms
            self.regenerate_mask_data()

        self.use_manual_label_button = QPushButton(text="using manual labels" if self.use_manual_label else "ignoring manual labels")
        self.use_manual_label_button.clicked.connect(use_manual_label)
        self.use_manual_label_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.use_manual_label_proxy = QGraphicsProxyWidget()
        self.use_manual_label_proxy.setWidget(self.use_manual_label_button)

        # add button to clear all manual labels
        def clear_manual_labels(event):
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ControlModifier:
                for plane in range(self.num_planes):
                    self.manual_label_active[plane][:] = False
                self.regenerate_mask_data()
            else:
                print("clearing manual labels requires a control click")

        self.clear_manual_label_button = QPushButton(text="clear manual labels")
        self.clear_manual_label_button.clicked.connect(clear_manual_labels)
        self.clear_manual_label_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.clear_manual_label_proxy = QGraphicsProxyWidget()
        self.clear_manual_label_proxy.setWidget(self.clear_manual_label_button)

        # add show manual labels only button
        def show_manual_labels(event):
            self.only_manual_labels = not (self.only_manual_labels)
            if self.only_manual_labels:
                self.use_manual_label = True
            self.show_manual_label_button.setText("only manual labels" if self.only_manual_labels else "all labels")
            self.regenerate_mask_data()

        self.show_manual_label_button = QPushButton(text="all labels")
        self.show_manual_label_button.clicked.connect(show_manual_labels)
        self.show_manual_label_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.show_manual_label_proxy = QGraphicsProxyWidget()
        self.show_manual_label_proxy.setWidget(self.show_manual_label_button)

        # add colormap selection button
        def next_color_state(event):
            self.color_state = np.mod(self.color_state + 1, len(self.color_button_names))
            self.color_button.setText(self.color_button_names[self.color_state])
            self.update_label_colors()

        self.color_button_names = ["random", *self.feature_names]
        self.color_button = QPushButton(text=self.color_button_names[self.color_state])
        self.color_button.setCheckable(False)
        self.color_button.clicked.connect(next_color_state)
        self.color_button.setStyleSheet(BASIC_BUTTON_STYLE)
        self.color_button_proxy = QGraphicsProxyWidget()
        self.color_button_proxy.setWidget(self.color_button)

        # add colormap selection button
        def next_colormap(event):
            self.idx_colormap = np.mod(self.idx_colormap + 1, len(self.list_colormaps))
            self.colormap_selection.setText(self.list_colormaps[self.idx_colormap])
            self.update_label_colors()

        self.colormap_selection = QPushButton(text=self.list_colormaps[self.idx_colormap])
        self.colormap_selection.clicked.connect(next_colormap)
        self.colormap_selection.setStyleSheet(BASIC_BUTTON_STYLE)
        self.colormap_selection_proxy = QGraphicsProxyWidget()
        self.colormap_selection_proxy.setWidget(self.colormap_selection)

        self.button_area.addItem(self.save_button_proxy, row=0, col=0)
        self.button_area.addItem(self.update_database_button_proxy, row=0, col=1)
        self.button_area.addItem(self.update_database_false_button_proxy, row=0, col=2)
        self.button_area.addItem(self.toggle_cell_button_proxy, row=0, col=3)
        self.button_area.addItem(self.use_manual_label_proxy, row=0, col=4)
        self.button_area.addItem(self.show_manual_label_proxy, row=0, col=5)
        self.button_area.addItem(self.clear_manual_label_proxy, row=0, col=6)
        self.button_area.addItem(self.color_button_proxy, row=0, col=7)
        self.button_area.addItem(self.colormap_selection_proxy, row=0, col=8)

        # add feature plots to napari window
        self.dock_window = self.viewer.window.add_dock_widget(self.feature_window, name="ROI Features", area="bottom")

        def switch_image_label(viewer):
            self.show_mask_image = not (self.show_mask_image)
            self.update_visibility()

        def update_mask_visibility(viewer):
            self.mask_visibility = not (self.mask_visibility)
            self.update_visibility()

        def update_reference_visibility(viewer):
            self.reference.visible = not (self.reference.visible)

        def save_curation_update_database(viewer):
            self.save_selection()
            self.update_database(True)

        self.viewer.bind_key("t", toggle_cells_to_view, overwrite=True)
        self.viewer.bind_key("s", switch_image_label, overwrite=True)
        self.viewer.bind_key("v", update_mask_visibility, overwrite=True)
        self.viewer.bind_key("r", update_reference_visibility, overwrite=True)
        self.viewer.bind_key("c", next_color_state, overwrite=True)
        self.viewer.bind_key("a", next_colormap, overwrite=True)
        self.viewer.bind_key("Control-c", save_curation_update_database, overwrite=False)

        # create single-click callback for printing data about ROI features
        def single_click_label(layer, event):
            if not (self.labels.visible):
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return

            # get click data
            plane_idx, yidx, xidx = [int(pos) for pos in event.position]
            label_idx = self.labels.data[plane_idx, yidx, xidx]
            if label_idx == 0:
                self.viewer.status = "single-click on background, no ROI selected"
                return

            # get ROI data
            # oh napari, oh napari
            roi_idx = label_idx - 1
            in_plane_idx = np.where(self.idx_roi[plane_idx] == (roi_idx))[0][0]
            feature_print = [f"{featname}={featdata[in_plane_idx]:.3f}" for featname, featdata in zip(self.feature_names, self.features[plane_idx])]

            string_to_print = f"ROI: {roi_idx}, Plane Idx: {plane_idx}, (inPlane)ROI: {in_plane_idx}, " + " ".join(feature_print)

            # only print single click data if alt is held down
            if "Alt" in event.modifiers:
                print(string_to_print)

            # always show message in viewer status
            self.viewer.status = string_to_print

        def double_click_label(layer, event):
            self.viewer.status = "you just double clicked!"

            # if not looking at labels, then don't allow manual selection (it would be random!)
            if not (self.labels.visible):
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return

            # if not looking at manual annotations, don't allow manual selection...
            if not (self.use_manual_label):
                self.viewer.status = "can only manually select cells when the manual labels are being used!"
                return

            plane_idx, yidx, xidx = [int(pos) for pos in event.position]
            label_idx = self.labels.data[plane_idx, yidx, xidx]
            if label_idx == 0:
                self.viewer.status = "double-click on background, no ROI identity toggled"
            else:
                if "Alt" in event.modifiers:
                    self.viewer.status = "Alt was used, assuming you are trying to single click and not doing a manual label!"
                else:
                    roi_idx = label_idx - 1
                    in_plane_idx = np.where(self.idx_roi[plane_idx] == (roi_idx))[0][0]
                    if "Control" in event.modifiers:
                        if self.only_manual_labels:
                            self.manual_label_active[plane_idx][in_plane_idx] = False
                            self.viewer.status = f"you just removed the manual label from roi: {roi_idx}"
                        else:
                            self.viewer.status = f"you can only remove a label if you are only looking at manualLabels!"
                    else:
                        # manual annotation: if plotting control cells, then annotate as red (1), if plotting red cells, annotate as control (0)
                        new_label = copy(self.control_cell_toggle)
                        self.manual_label[plane_idx][in_plane_idx] = new_label
                        self.manual_label_active[plane_idx][in_plane_idx] = True
                        self.viewer.status = f"you just labeled roi: {roi_idx} with the identity: {new_label}"
                    self.regenerate_mask_data()

        self.labels.mouse_drag_callbacks.append(single_click_label)
        self.masks.mouse_drag_callbacks.append(single_click_label)
        self.reference.mouse_drag_callbacks.append(single_click_label)
        self.labels.mouse_double_click_callbacks.append(double_click_label)
        self.masks.mouse_double_click_callbacks.append(double_click_label)
        self.reference.mouse_double_click_callbacks.append(double_click_label)

        # add callback for dimension slider
        def update_plane_idx(event):
            self.plane_idx = event.source.current_step[0]
            self.update_feature_plots()

        self.viewer.dims.events.connect(update_plane_idx)

    def update_visibility(self):
        """Update visibility of mask and label layers in napari viewer."""
        self.masks.visible = self.show_mask_image and self.mask_visibility
        self.labels.visible = not (self.show_mask_image) and self.mask_visibility

    def update_feature_plots(self):
        """Update feature histogram plots for the current plane."""
        for feature in range(self.num_features):
            self.hist_graphs[feature].setOpts(height=self.hvalues[self.plane_idx][feature])
            self.hist_reds[feature].setOpts(height=self.hvalred[self.plane_idx][feature])

    def update_label_colors(self):
        """
        Update the colors of the labels in the napari viewer.

        Colors can be random or based on feature values, depending on the
        current color_state setting.
        """
        if self.color_state == 0:
            # this is inherited from the default random colormap in napari
            colormap = label_colormap(49, 0.5, background_value=0)
        else:
            # assign colors based on the feature values for every ROI
            norm = mpl.colors.Normalize(
                vmin=self.feature_range[self.color_state - 1][0],
                vmax=self.feature_range[self.color_state - 1][1],
            )
            colors = plt.colormaps[self.list_colormaps[self.idx_colormap]](
                norm(np.concatenate([feat[self.color_state - 1] for feat in self.features]))
            )
            color_dict = dict(zip(np.concatenate(self.idx_roi) + 1, colors))
            # transparent background (or default)
            color_dict[None] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.single)
            colormap = direct_colormap(color_dict)
        # Update colors of the labels
        self.labels.color = colormap

    def process_planes(self):
        """
        Process all planes to extract features and compute histograms.

        Loads red cell features (S2P, dot product, Pearson correlation,
        phase correlation) and manual labels for each plane, then computes
        histograms for visualization.
        """
        red_s2p = self.red_cell.b2session.loadone("mpciROIs.redS2P")
        red_dot = self.red_cell.b2session.loadone("mpciROIs.redDotProduct")
        red_corr = self.red_cell.b2session.loadone("mpciROIs.redPearson")
        red_phase = self.red_cell.b2session.loadone("mpciROIs.redPhaseCorrelation")
        manual_labels = self.red_cell.b2session.loadone("mpciROIs.redCellManualAssignments")
        for plane_idx in range(self.num_planes):
            self.ref_image[plane_idx] = self.red_cell.reference[plane_idx]
            self.idx_roi[plane_idx] = np.where(self.red_cell.roi_plane_idx == plane_idx)[0]
            self.manual_label[plane_idx] = manual_labels[0][self.red_cell.roi_plane_idx == plane_idx]
            self.manual_label_active[plane_idx] = manual_labels[1][self.red_cell.roi_plane_idx == plane_idx]
            self.features[plane_idx] = [None] * self.num_features
            self.features[plane_idx][0] = red_s2p[self.red_cell.roi_plane_idx == plane_idx]
            self.features[plane_idx][1] = red_dot[self.red_cell.roi_plane_idx == plane_idx]
            self.features[plane_idx][2] = red_corr[self.red_cell.roi_plane_idx == plane_idx]
            self.features[plane_idx][3] = red_phase[self.red_cell.roi_plane_idx == plane_idx]
            self.hvalues[plane_idx] = [None] * self.num_features
            self.hvalred[plane_idx] = [None] * self.num_features

        # use the same edges across planes
        for feature in range(self.num_features):
            feature_across_planes = np.concatenate([feature_data[feature] for feature_data in self.features])
            self.hedges[feature] = np.histogram(feature_across_planes, bins=self.num_bins)[1]

        for plane_idx in range(self.num_planes):
            for feature in range(self.num_features):
                self.hvalues[plane_idx][feature] = np.histogram(self.features[plane_idx][feature], bins=self.hedges[feature])[0]
                self.hvalred[plane_idx][feature] = np.histogram(
                    self.features[plane_idx][feature][self.red_idx[plane_idx]],
                    bins=self.hedges[feature],
                )[0]

        # establish maximum for the yranges
        max_value = [[max(hval) for hval in hvalue] for hvalue in self.hvalues]
        self.hvalues_maximum = [max(x) for x in zip(*max_value)]

    def mask_labels(self):
        """
        Create label data for napari labels layer.

        Returns
        -------
        np.ndarray
            Label data array of shape (num_planes, ly, lx) with ROI indices.
            Note that labelData handles indices in a complicated way so that
            it's easy to interface with Napari. Key points:
            1. ROIs are assigned an index that is unique across all ROIs
               independent of plane (the first ROI in plane 1 isn't ROI 0,
               it's 1 + the number of ROIs in plane 0)
            2. All ROI indices are incremented by 1 when they are added to
               the "label" layer of the napari viewer. This is because the
               label layer uses "0" to indicate "no label"
            3. ROIs are only presented if they are True in "self.idx_masks_to_plot",
               which is a boolean array of size (numROIsPerPlane,).
        """
        label_data = np.zeros((self.num_planes, self.red_cell.ly, self.red_cell.lx), dtype=int)
        for plane_idx in range(self.num_planes):
            plot_idx = self.idx_masks_to_plot(plane_idx)
            for idx, roi in enumerate(self.idx_roi[plane_idx]):
                if plot_idx[idx]:
                    # 0 is transparent for a labels layer in napari, so 1 index the ROIs!
                    label_data[plane_idx, self.red_cell.ypix[roi], self.red_cell.xpix[roi]] = roi + 1
        return label_data

    def mask_image(self):
        """
        Create mask image data for napari image layer.

        Returns
        -------
        np.ndarray
            Mask image data array of shape (num_planes, ly, lx) with ROI mask
            weights (lam values).
        """
        image_data = np.zeros((self.num_planes, self.red_cell.ly, self.red_cell.lx))
        for plane_idx in range(self.num_planes):
            plot_idx = self.idx_masks_to_plot(plane_idx)
            for idx, roi in enumerate(self.idx_roi[plane_idx]):
                if plot_idx[idx]:
                    image_data[plane_idx, self.red_cell.ypix[roi], self.red_cell.xpix[roi]] = self.red_cell.lam[roi]
        return image_data

    def idx_masks_to_plot(self, plane_idx):
        """
        Determine which masks to plot for a given plane.

        Parameters
        ----------
        plane_idx : int
            Index of the plane.

        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs in the plane should be plotted.
        """
        # standard function for determining which masks to plot for each plane
        if self.only_manual_labels:
            # if only showing manual labels, initialize plot index as all false, then update as usual
            plot_idx = np.full(self.red_idx[plane_idx].shape, False)
        else:
            # if showing all labels, then initialize plotIdx with whatever is currently passing the feature rules
            plot_idx = np.copy(self.red_idx[plane_idx] if not (self.control_cell_toggle) else ~self.red_idx[plane_idx])
        if self.use_manual_label:
            plot_idx[self.manual_label_active[plane_idx]] = (
                self.manual_label[plane_idx][self.manual_label_active[plane_idx]] != self.control_cell_toggle
            )
        return plot_idx

    def update_red_idx(self):
        """
        Update red cell index based on current feature cutoffs.

        Applies minimum and maximum cutoffs for each active feature to
        determine which ROIs should be classified as red cells.
        """
        for plane_idx in range(self.num_planes):
            self.red_idx[plane_idx] = np.full(self.roi_per_plane[plane_idx], True)  # start with all as red...
            for feature in range(self.num_features):
                if self.feature_active[feature][0]:
                    # only keep in red_idx if above minimum
                    self.red_idx[plane_idx] &= self.features[plane_idx][feature] >= self.feature_cutoffs[feature][0]
                if self.feature_active[feature][1]:
                    # only keep in red_idx if below maximum
                    self.red_idx[plane_idx] &= self.features[plane_idx][feature] <= self.feature_cutoffs[feature][1]

        # now that the red idx has been updated, we need new mask data and new histograms
        self.regenerate_mask_data()

    def regenerate_mask_data(self):
        """
        Regenerate mask and label data and update histograms.

        Updates the napari viewer layers and recomputes histograms for
        the currently selected red cells.
        """
        self.masks.data = self.mask_image()
        self.labels.data = self.mask_labels()
        for plane_idx in range(self.num_planes):
            for feature in range(self.num_features):
                if self.only_manual_labels:
                    c_red_idx = np.full(self.red_idx[plane_idx].shape, False)
                else:
                    c_red_idx = np.copy(self.red_idx[plane_idx])
                if self.use_manual_label:
                    # if using manual label, any manual labels will overwrite red idx if manual label is active
                    c_red_idx[self.manual_label_active[plane_idx]] = self.manual_label[plane_idx][self.manual_label_active[plane_idx]]
                self.hvalred[plane_idx][feature] = np.histogram(self.features[plane_idx][feature][c_red_idx], bins=self.hedges[feature])[0]

        # regenerate histograms
        for feature in range(self.num_features):
            self.hist_reds[feature].setOpts(height=self.hvalred[self.plane_idx][feature])

    def save_selection(self):
        """
        Save red cell selection and feature cutoffs to onedata.

        Saves the current red cell index, manual labels, and feature cutoff
        values to the session's onedata storage.
        """
        full_red_idx = np.concatenate(self.red_idx)
        full_manual_labels = np.stack((np.concatenate(self.manual_label), np.concatenate(self.manual_label_active)))
        self.red_cell.b2session.saveone(full_red_idx, "mpciROIs.redCellIdx")
        self.red_cell.b2session.saveone(full_manual_labels, "mpciROIs.redCellManualAssignments")
        for idx, name in enumerate(self.feature_names):
            c_feature_cutoffs = self.feature_cutoffs[idx]
            if not (self.feature_active[idx][0]):
                c_feature_cutoffs[0] = np.nan
            if not (self.feature_active[idx][1]):
                c_feature_cutoffs[1] = np.nan
            self.red_cell.b2session.saveone(self.feature_cutoffs[idx], self.red_cell.one_name_feature_cutoffs(name))

        print(f"Red Cell curation choices are saved for session {self.red_cell.b2session.session_print()}")

    def update_database(self, state: bool):
        """
        Update red cell QC status in the database.

        Parameters
        ----------
        state : bool
            QC status to set (True or False).
        """
        vrdb = database.SessionDatabase()
        success = vrdb.set_red_cell_qc(
            self.red_cell.b2session.mouse_name, self.red_cell.b2session.date, self.red_cell.b2session.session_id, state=state
        )
        if success:
            print(f"Successfully updated the redCellQC field of the database to {state} for session {self.red_cell.b2session.session_print()}")
        else:
            print(f"Failed to update the redCellQC field of the database for session {self.red_cell.b2session.session_print()}")
