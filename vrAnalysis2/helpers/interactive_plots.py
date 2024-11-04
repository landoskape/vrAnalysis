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

    def __call__(self):
        """make getting the value simple"""
        return self.value


class Slider:
    def __init__(self, selection, name, callback=None, callback_requires_input=True):
        self.selection = selection
        self.name = name
        self.callback = callback
        self.callback_requires_input = callback_requires_input
        self._build_slider()
        self._build_extras()
        self._build_layout()
        self._add_shortcut()

    def _build_slider(self):
        # build slider label
        self.slider_name = QLabel(f"{self.name}")
        self.slider_name.setAlignment(QtCore.Qt.AlignCenter)

        # build slider
        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.selection.minval)
        self.slider.setMaximum(self.selection.maxval)
        self.slider.setSingleStep(1)
        self.slider.setPageStep((self.selection.maxval - self.selection.minval) // 10)
        self.slider.setValue(self.selection.value)
        self.slider.valueChanged.connect(self.update_slider)

    def _build_extras(self):
        """overwrite if need more to the slider bar, then overwrite _build_layout"""
        pass

    def _build_layout(self):
        # layout
        selection_layout = QGridLayout()
        selection_layout.addWidget(self.slider_name, 0, 0)
        selection_layout.addWidget(self.slider, 0, 1)

        selection_layout.setColumnStretch(0, 1)
        selection_layout.setColumnStretch(1, 8)

        self.selection_layout = selection_layout

    def _add_shortcut(self):
        # print("WARNING: I removed the window attribute so shortcuts have to be added manually by the caller")
        # self.shortcut = QShortcut(QKeySequence(self.shortcut_key), self.window)
        # self.shortcut.activated.connect(self.go_to_value)
        pass

    def _get_value(self):
        return self.selection.value

    def update_slider(self, value):
        self.selection.update(value)
        self.slider.setValue(self._get_value())
        self._update_extras(self._get_value())
        self._do_callback()

    def _update_extras(self, value):
        """overwrite if need more to the slider bar"""
        pass

    def _do_callback(self):
        if self.callback is not None:
            if self.callback_requires_input:
                self.callback(self._get_value())
            else:
                self.callback()


class SliderSelector(Slider):
    def _build_extras(self):
        # build edit field
        self.edit_field = QLineEdit()
        self.edit_field.setText("0")

        # build previous button
        self.prev_button = QPushButton("button", text="Prev")
        self.prev_button.clicked.connect(self._prev_value)

        # build previous button
        self.next_button = QPushButton("button", text="Next")
        self.next_button.clicked.connect(self._next_value)

        # go to edit proxy
        self.go_to_value_button = QPushButton("button", text="go to value")
        self.go_to_value_button.clicked.connect(self._go_to_value)

    def _build_layout(self):
        selection_layout = QGridLayout()
        selection_layout.addWidget(self.slider_name, 0, 0)
        selection_layout.addWidget(self.prev_button, 0, 1)
        selection_layout.addWidget(self.slider, 0, 2)
        selection_layout.addWidget(self.next_button, 0, 3)
        selection_layout.addWidget(self.edit_field, 0, 4)
        selection_layout.addWidget(self.go_to_value_button, 0, 5)

        selection_layout.setColumnStretch(0, 2)
        selection_layout.setColumnStretch(1, 1)
        selection_layout.setColumnStretch(2, 8)
        selection_layout.setColumnStretch(3, 1)
        selection_layout.setColumnStretch(4, 0)
        selection_layout.setColumnStretch(5, 1)

        self.selection_layout = selection_layout

    def _add_shortcut(self):
        # print("WARNING: I removed the window attribute so shortcuts have to be added manually by the caller")
        # self.shortcut = QShortcut(QKeySequence(self.shortcut_key), self.window)
        # self.shortcut.activated.connect(self.go_to_value)
        pass

    def _get_value(self):
        return self.selection.value

    def _do_callback(self):
        if self.callback is not None:
            if self.callback_requires_input:
                self.callback(self._get_value())
            else:
                self.callback()

    def _update_extras(self, value):
        self.edit_field.setText(str(value))

    def _prev_value(self):
        self.update_slider(self._get_value() - 1)

    def _next_value(self):
        self.update_slider(self._get_value() + 1)

    def _go_to_value(self):
        if not self.edit_field.text().isdigit():
            self.edit_field.setText("invalid selection")
            return
        text_value = int(self.edit_field.text())
        if (text_value < self.selection.minval) or (text_value > self.selection.maxval):
            self.edit_field.setText("selection out of range")
            return

        # otherwise text is valid value
        self.update_slider(text_value)
