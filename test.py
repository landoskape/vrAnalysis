import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QPushButton
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5 import QtCore


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the dark mode color scheme
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)

        # Set up the window frame style
        self.setStyleSheet(
            "QWidget { background-color: #353535; }"
            "QMenuBar { background-color: #4285f4; color: white; }"
            "QMenuBar::item { background-color: transparent; }"
            "QMenuBar::item:selected { background-color: #3073d6; }"
        )

        layout = QVBoxLayout()

        # Create a blue-accented slider
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { background-color: #1e1e1e; height: 8px; border-radius: 4px; }"
            "QSlider::handle:horizontal { background-color: #4285f4; width: 18px; height: 18px; border-radius: 9px; margin: -5px 0; }"
        )
        layout.addWidget(self.slider)

        # Create a blue-accented button
        self.button = QPushButton("Print Slider Value")
        self.button.setStyleSheet("QPushButton { background-color: #4285f4; color: white; padding: 10px 20px; border-radius: 5px; }")
        self.button.clicked.connect(self.print_slider_value)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("Slider Example")

    def print_slider_value(self):
        slider_value = self.slider.value()
        print(f"Slider value: {slider_value}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
