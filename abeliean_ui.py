import pyqtgraph
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPalette, QColor, QFont
import abliean_ui_class
from PyQt5.QtCore import *
from functools import partial
import cv2
import numpy as np
import sandpile_elite


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        title_font = QFont('Arial', 20)
        config_area = QVBoxLayout()
        title_label = QLabel(self)
        title_label.setText("Abliean Sandpile Simulation")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        config_area.addWidget(title_label)
        input_field = abliean_ui_class.input_layout(self)
        config_area.addWidget(input_field)
        self.plot = abliean_ui_class.simulation_layout(self)
        config_area.addWidget(self.plot)

        self.worker = abliean_ui_class.WorkThread()

        input_field.preview.clicked.connect(lambda: self.plot.update_preview(
            int(input_field.config[0].text()),
            int(input_field.config[1].text()),
            int(input_field.config[2].text()),
            int(input_field.config[3].text()),
            float(input_field.config[4].text())))
        input_field.simulation_start.clicked.connect(self.execute)
        input_field.simulation_stop.clicked.connect(input_field.stop)
        input_field.prime_step.clicked.connect(
            self.plot.change_prime)
        self.setLayout(config_area)
        self.setWindowTitle("Abliean Simulation Software")

    def execute(self):
        self.worker.start()
        self.worker.trigger.connect(self.display_sandpile_state_map)
        self.worker.cascade_trigger.connect(self.display_cascade_changes)
        self.worker.ce_loss_trigger.connect(self.display_loss_changes)
        self.worker.cum_vmap_trigger.connect(self.total_volume_heatmap_volume)
        self.worker.fat_tail_trigger.connect(self.display_fat_tail)
        self.worker.black_red_ratio_trigger.connect(self.display_ratio)
        self.worker.falloff_bits_trigger.connect(self.display_falloff)

    def display_sandpile_state_map(self, image):
        image = ((image - np.min(image))/(np.max(image)-np.min(image)))*255
        image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
        self.plot.heatmap_item.updateImage(image, cmap='jet')

    def display_cascade_changes(self, cascades):
        x = np.linspace(0, cascades.shape[0], cascades.shape[0])
        self.plot.cascades_plot.setData(
            x, cascades, pen={'color': (255, 0, 0), 'width': 2})

    def total_volume_heatmap_volume(self, image):
        image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_HOT)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.plot.cum_vmap_item.updateImage(image, cmap='jet')

    def display_loss_changes(self, loss):
        x = np.linspace(0, loss.shape[0], loss.shape[0])
        self.plot.ce_plot.setData(
            x, loss, pen={'color': (0, 255, 255), 'width': 2})

    def display_fat_tail(self, freq):
        fr_x, fr_y, dp = sandpile_elite.fat_tail_curve(freq)
        self.plot.fat_tail_scatter.setData(fr_x, fr_y, pen={
            'color': 'g'}, brush='g')

        # self.plot.fat_tail_curve.setData(
        # fr_x, dp, pen={'color': (255, 140, 0), 'width': 2})
        if len(fr_y):
            self.plot.fat_tail_graph.setYRange(
                np.min(fr_y)-0.1, np.max(fr_y)+0.1)
            self.plot.fat_tail_graph.setXRange(
                np.min(fr_x), np.max(fr_x))

    def display_ratio(self, ratios):
        red, black = ratios[0], ratios[1]
        self.plot.red_ratio_plot.setData(np.arange(len(red)), red, pen={
                                         'color': (255, 0, 0), 'width': 2})
        self.plot.black_ratio_plot.setData(np.arange(len(black)), black, pen={
            'color': (0, 0, 0), 'width': 2})

    def display_falloff(self, falloff):
        self.plot.fallout_bits_graph.setData(np.arange(len(falloff)), falloff, pen={
            'color': (0, 0, 255), 'width': 2})


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
