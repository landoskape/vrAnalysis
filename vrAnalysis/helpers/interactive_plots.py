import numpy as np

# GUI-related modules
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QGraphicsProxyWidget,
    QGridLayout,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence


# supporting class for storing and updating the indices for the example viewer
class CurrentSelection:
    def __init__(self, value=None, minval=0, maxval=None):
        self.minval = minval
        self.maxval = maxval if maxval is not None else np.inf
        assert self.minval < self.maxval, "minimum value must be less than maximum value"
        self.value = value if value is not None else minval

    def update(self, value):
        """update with bounds"""
        self.value = np.minimum(np.maximum(value, self.minval), self.maxval)
        self._callback()

    def __call__(self):
        """make getting the value simple"""
        return self.value

    def _callback(self):
        print(self.value)


class SliderSelector:
    def __init__(self, window, selection, name, callback=None, callback_requires_input=True, row=None, col=None, shortcut_key=None):
        self.window = window
        self.selection = selection
        self.name = name
        self.row = row
        self.col = col
        self.callback = callback
        self.callback_requires_input = callback_requires_input
        self.shortcut_key = shortcut_key
        self._build_widgets()
        self._add_shortcut()

    def _build_widgets(self):
        # build slider label
        # self.slider_name_proxy = QGraphicsProxyWidget()
        self.slider_name = QLabel(f"{self.name} {1}/{self.selection.maxval}")
        self.slider_name.setAlignment(QtCore.Qt.AlignCenter)
        # self.slider_name_proxy.setWidget(self.label)

        # build edit field
        # self.edit_field_proxy = QGraphicsProxyWidget()
        self.edit_field = QLineEdit()
        self.edit_field.setText("0")
        # self.edit_field_proxy.setWidget(self.edit_field)

        # build slider
        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.selection.maxval - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(int(self.selection.maxval / 10))
        self.slider.setValue(self.selection.value)
        self.slider.valueChanged.connect(self.update_slider)
        # self.slider_proxy = QGraphicsProxyWidget()
        # self.slider_proxy.setWidget(self.slider)

        # build previous button
        # self.prev_button_proxy = QGraphicsProxyWidget()
        self.prev_button = QPushButton("button", text="Prev")
        self.prev_button.clicked.connect(self.prev_value)
        # self.prev_button_proxy.setWidget(self.prev_button)

        # build previous button
        # self.next_button_proxy = QGraphicsProxyWidget()
        self.next_button = QPushButton("button", text="Next")
        self.next_button.clicked.connect(self.next_value)
        # self.next_button_proxy.setWidget(self.next_button)

        # go to edit proxy
        # self.go_to_value_proxy = QGraphicsProxyWidget()
        self.go_to_value_button = QPushButton("button", text="go to value")
        self.go_to_value_button.clicked.connect(self.go_to_value)
        # self.go_to_value_proxy.setWidget(self.go_to_value_button)

        # layout
        selection_layout = QGridLayout()
        selection_layout.addWidget(self.slider_name, 0, 0)
        selection_layout.addWidget(self.prev_button, 0, 1)
        selection_layout.addWidget(self.slider, 0, 2)
        selection_layout.addWidget(self.next_button, 0, 3)
        selection_layout.addWidget(self.edit_field, 0, 4)
        selection_layout.addWidget(self.go_to_value_button, 0, 5)

        selection_layout.setColumnStretch(0, 1)
        selection_layout.setColumnStretch(1, 0)
        selection_layout.setColumnStretch(2, 3)
        selection_layout.setColumnStretch(3, 0)
        selection_layout.setColumnStretch(4, 0)
        selection_layout.setColumnStretch(5, 0)

        # add slider to main window

        selection_layout_proxy = QGraphicsProxyWidget()
        selection_layout_proxy.setWidget(selection_layout)
        self.window.addItem(selection_layout_proxy, row=self.row, col=self.col)

    def _add_shortcut(self):
        # add shortcut for going to ROI without pressing the button...
        if self.shortcut_key is not None:
            self.shortcut = QShortcut(QKeySequence(self.shortcut_key), self.window)
            self.shortcut.activated.connect(self.go_to_value)

    def _get_value(self):
        return self.selection.value

    def _do_callback(self):
        if self.callback is not None:
            if self.callback_requires_input:
                self.callback(self._get_value())
            else:
                self.callback()

    def _set_text(self):
        self.slider_name.setText(f"{self.name} {self._get_value()}/{self.selection.maxval}")
        self.edit_field.setText(str(self._get_value()))
        self._do_callback()

    def update_slider(self, value):
        self.selection.update(value)
        self.slider.setValue(self._get_value())
        self._set_text()
        self._do_callback()

    def prev_value(self):
        self.selection.update(self._get_value() - 1)  # try updating roi value
        self.slider.setValue(self._get_value())  # update slider
        self._set_text()
        self._do_callback()

    def next_value(self):
        self.selection.update(self._get_value() + 1)  # try updating roi value
        self.slider.setValue(self._get_value())  # update slider
        self._set_text()
        self._do_callback()

    def go_to_value(self):
        if not self.edit_field.text().isdigit():
            self.edit_field.setText("invalid ROI")
            return
        text_value = int(self.edit_field.text())
        if (text_value < self.selection.minval) or (text_value > self.selection.maxval):
            self.edit_field.setText("ROI out of range")
            return

        # otherwise text is valid value
        self.selection.update(text_value)
        self.slider.setValue(self._get_value())  # update slider
        self._set_text()
        self._do_callback()
