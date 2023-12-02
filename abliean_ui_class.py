import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPalette, QColor, QFont
import PyQt5.QtGui
import numpy as np
import pyqtgraph as pg
import cv2
import sandpile_elite
from threading import Thread
from PyQt5.QtCore import QThread, pyqtSignal
import time
matrix = []
cascades = []
iterations = 0
stop = False
isPrimed = False
black_ratio = []
red_ratio = []
red_config_ratio = 0.0


class WorkThread(QThread):
    trigger = pyqtSignal(np.ndarray)
    cascade_trigger = pyqtSignal(np.ndarray)
    ce_loss_trigger = pyqtSignal(np.ndarray)
    cum_vmap_trigger = pyqtSignal(np.ndarray)
    fat_tail_trigger = pyqtSignal(dict)
    black_red_ratio_trigger = pyqtSignal(list)

    def __int__(self):
        super(WorkThread, self).__init__()

    def black_red_ratio(self, matrix):
        global black_ratio, red_ratio
        total_count = 0
        red = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v, counts = np.unique(matrix[i][j], return_counts=True)
                if len(v) == 1:
                    if v[0] == '0':
                        red += counts[0]
                        total_count += (counts[0])
                    else:
                        total_count += (counts[0])
                elif len(v) == 2:
                    red += counts[0]
                    total_count += (counts[0]+counts[1])
                else:
                    continue
        red_ratio.append(red/total_count)
        black_ratio.append((total_count-red)/total_count)
        return [red_ratio, black_ratio]

    def run(self):
        global matrix, cascades, stop, iterations, red_config_ratio
        stop = False

        frequency_dict = dict()
        information_transmitted = {
            "a": 1e-3,
            "b": 1e-3,
            "c": 1e-3,
            "d": 1e-3,
            "e": 1e-3,
            "f": 1e-3,
            "g": 1e-3,
            "h": 1e-3,
            "i": 1e-3,
            "j": 1e-3,
            "k": 1e-3,
            "l": 1e-3,
            "m": 1e-3,
            "n": 1e-3,
            "o": 1e-3,
            "p": 1e-3
        }
        information_retained = {
            "a": 1e-3,
            "b": 1e-3,
            "c": 1e-3,
            "d": 1e-3,
            "e": 1e-3,
            "f": 1e-3,
            "g": 1e-3,
            "h": 1e-3,
            "i": 1e-3,
            "j": 1e-3,
            "k": 1e-3,
            "l": 1e-3,
            "m": 1e-3,
            "n": 1e-3,
            "o": 1e-3,
            "p": 1e-3
        }
        cascades = []
        area = []
        volume = []

        loss_history = []
        if isPrimed:
            iterations += (matrix.shape[0] ** 2 * 10)

        for i in range(iterations):
            if stop:
                break
            msg_sand = None
            if isPrimed:
                if i > (matrix.shape[0] ** 2 * 10):
                    random_choice = np.random.uniform()
                    if random_choice < red_config_ratio:
                        # print("red")
                        msg_sand = sandpile_elite.msg_cipher[0]
                    else:
                        # print("black")
                        msg_sand = sandpile_elite.msg_cipher[-1]
                    # msg_sand = np.random.choice(
                        # sandpile_elite.msg_cipher, size=1)[0]
                else:
                    msg_sand = sandpile_elite.msg_cipher[0]
            else:
                msg_sand = np.random.choice(
                    sandpile_elite.msg_cipher, size=1)[0]

            information_transmitted[msg_sand] = information_transmitted[msg_sand] + 1
            matrix = sandpile_elite.drop_sand(matrix, msg_sand)
            cas_count, area_affected, cascade_magnitude, mag_area = sandpile_elite.regulate_pile(
                matrix, i, self.trigger)

            sandpile_elite.update_information(information_retained, matrix)
            ce = sandpile_elite.entropy_loss(list(information_transmitted.values()), list(
                information_retained.values()))

            loss_history.append(ce)
            cascades.append(cas_count)
            area.append(area_affected)
            volume.append(cascade_magnitude)
            # emit signal
            self.trigger.emit(mag_area)
            self.cascade_trigger.emit(np.array(cascades))
            self.ce_loss_trigger.emit(np.array(loss_history))
            self.cum_vmap_trigger.emit(
                np.array(sandpile_elite.area_volume_heatmap))

            if cas_count not in list(frequency_dict.keys()):
                frequency_dict[cas_count] = 1
            else:
                frequency_dict[cas_count] += 1
            self.fat_tail_trigger.emit(frequency_dict)
            self.black_red_ratio_trigger.emit(self.black_red_ratio(matrix))
            time.sleep(0.01)


class input_layout(QWidget):
    def __init__(self, parent=None):
        super(input_layout, self).__init__(parent)

        _font = QFont('Arial', 13)
        attributes = ["Grid Size", "Elite Size",
                      "Iterations", "Elite Cap", "Red ratio"]
        label = []
        self.config = []
        self.vlayout = QVBoxLayout()
        grid = QGridLayout()

        for i in range(len(attributes)):
            config1_label = QLabel(parent)
            config1_label.setText(attributes[i] + ": ")
            config1_label.setFont(_font)
            label.append(config1_label)

            config1_textbox = QLineEdit(parent)
            config1_textbox.setPlaceholderText("Enter " + attributes[i])
            config1_textbox.setFont(_font)
            self.config.append(config1_textbox)

        grid.addWidget(label[0], 0, 0)
        grid.addWidget(self.config[0], 0, 1)
        grid.addWidget(label[1], 0, 2)
        grid.addWidget(self.config[1], 0, 3)

        grid.addWidget(label[2], 1, 0)
        grid.addWidget(self.config[2], 1, 1)
        grid.addWidget(label[3], 1, 2)
        grid.addWidget(self.config[3], 1, 3)

        grid.addWidget(label[4], 2, 0)
        grid.addWidget(self.config[4], 2, 1)

        self.button_box = QHBoxLayout()

        self.simulation_start = QPushButton(parent)
        self.simulation_start.setText("Start Simulation")
        self.simulation_start.setFont(_font)

        self.simulation_stop = QPushButton(parent)
        self.simulation_stop.setText("Stop Simulation")
        self.simulation_stop.setFont(_font)

        self.preview = QPushButton(parent)
        self.preview.setText("Update configuration")
        self.preview.setFont(_font)

        self.prime_step = QPushButton(parent)
        self.prime_step.setText("Prime Model")
        self.prime_step.setFont(_font)

        self.button_box.addWidget(self.preview)
        self.button_box.addWidget(self.simulation_start)
        self.button_box.addWidget(self.simulation_stop)
        self.button_box.addWidget(self.prime_step)

        self.vlayout.addLayout(grid)
        self.vlayout.addLayout(self.button_box)

        self.setLayout(self.vlayout)

    def stop(self):
        global stop
        stop = True

    def set_prime(self):
        global isPrimed
        isPrimed = True


class simulation_layout(QWidget):
    def __init__(self, parent=None):
        super(simulation_layout, self).__init__(parent)
        height, width = 250, 250
        # Some random data for scatter plot
        self.layout_ = QVBoxLayout()
        # Create layout to hold multiple subplots
        pg_layout = pg.GraphicsLayoutWidget()
        pg_layout.setBackground('w')

        # Add subplots
        # --------------------------------------------------------------------------------------------------------------
        self.p1 = pg_layout.addPlot(pen={'color': (0, 0, 0), 'width': 2}, symbol='x', row=0,
                                    col=0, title="Elite size configuration")

        self.p1.getAxis("left").setTextPen((0, 0, 0))
        self.p1.getAxis("bottom").setTextPen((0, 0, 0))

        self.p1.setTitle(title="Elite size configuration",
                         color=(0, 0, 0), fontsize=20)

        self.elite_grid = pg.ImageItem(
            np.random.randint(255, size=(25, 25), dtype=np.uint8))  # create example image
        self.p1.setFixedHeight(height)
        self.p1.setFixedWidth(width)
        self.p1.addItem(self.elite_grid)
        # --------------------------------------------------------------------------------------------------------------
        p2 = pg_layout.addPlot(pen={'color': (0, 0, 0), 'width': 2}, symbol='x', row=0,
                               col=1)
        p2.getAxis("left").setTextPen((0, 0, 0))
        p2.getAxis("bottom").setTextPen((0, 0, 0))

        p2.setTitle(title="Sandpile State",
                    color=(0, 0, 0), fontsize=20)

        self.heatmap = np.random.randint(255, size=(25, 25), dtype=np.uint8)

        image = cv2.applyColorMap(self.heatmap, cv2.COLORMAP_JET)

        self.heatmap_item = pg.ImageItem(
            image, cmap='jet')  # create example image
        self.heatmap_item.setOpts(fixedSize=(100, 100))

        p2.addItem(self.heatmap_item)
        p2.setFixedHeight(height)
        p2.setFixedWidth(width)
        # --------------------------------------------------------------------------------------------------------------
        p3 = pg_layout.addPlot(row=1,
                               col=0, title="Number of Cascades")

        p3.setTitle(title="Number of Cascades",
                    color=(0, 0, 0), fontsize=20)
        y = np.zeros(20)
        x = np.linspace(0, 20, 20)
        self.cascades_plot = pg.PlotCurveItem(
            x, y, pen={'color': (255, 0, 0), 'width': 2})
        p3.showGrid(x=True, y=True)
        p3.addItem(self.cascades_plot)
        p3.setFixedHeight(height)
        p3.setFixedWidth(width)

        # --------------------------------------------------------------------------------------------------------------
        p4 = pg_layout.addPlot(pen={'color': (20, 20, 0), 'width': 2}, row=1,
                               col=1, title="Entropy Loss")
        p4.getAxis("left").setTextPen((0, 0, 0))
        p4.getAxis("bottom").setTextPen((0, 0, 0))
        p4.setTitle(title="Entropy Loss",
                    color=(0, 0, 0), fontsize=20)
        p4.showGrid(x=True, y=True)

        self.ce_plot = pg.PlotCurveItem(
            x, y, pen={'color': (0, 255, 255), 'width': 2})
        p4.addItem(self.ce_plot)
        p4.setFixedHeight(height)
        p4.setFixedWidth(width)
        # --------------------------------------------------------------------------------------------------------------

        p5 = pg_layout.addPlot(pen={'color': (0, 0, 0), 'width': 2},
                               row=0, col=2, title="Sandpile Cumlative heatmap")
        p5.getAxis("left").setTextPen((0, 0, 0))
        p5.getAxis("bottom").setTextPen((0, 0, 0))

        p5.setTitle(title="Sandpile Cumlative heatmap",
                    color=(0, 0, 0), fontsize=20)

        self.cum_vmap = np.random.randint(255, size=(25, 25), dtype=np.uint8)

        cum_vmap_image = cv2.applyColorMap(self.cum_vmap, cv2.COLORMAP_JET)

        self.cum_vmap_item = pg.ImageItem(
            cum_vmap_image, cmap='jet')  # create example image

        self.cum_vmap_item.setOpts(fixedSize=(100, 100))

        p5.addItem(self.cum_vmap_item)
        p5.setFixedHeight(height)
        p5.setFixedWidth(width)
        # --------------------------------------------------------------------------------------------------------------
        self.fat_tail_graph = pg_layout.addPlot(pen={'color': (0, 0, 0), 'width': 2},
                                                row=1, col=2, title="Sandpile")
        self.fat_tail_graph.getAxis("left").setTextPen((0, 0, 0))
        self.fat_tail_graph.getAxis("bottom").setTextPen((0, 0, 0))
        self.fat_tail_graph.setTitle(title="Frequency vs Magnitude",
                                     color=(0, 0, 0), fontsize=20)

        scatter_data = np.zeros(20)
        scatter_x = np.arange(scatter_data.shape[0])
        self.fat_tail_scatter = pg.ScatterPlotItem(scatter_x, scatter_data, pen={
            'color': 'g'}, brush='g')
        self.fat_tail_curve = pg.PlotCurveItem(scatter_x, scatter_data, pen={
                                               'color': (255, 140, 0), 'width': 2})
        self.fat_tail_graph.setYRange(
            np.min(scatter_data), np.max(scatter_data))

        self.fat_tail_graph.addItem(self.fat_tail_scatter)
        # self.fat_tail_graph.addItem(self.fat_tail_curve)
        self.fat_tail_graph.showGrid(x=True, y=True)
        self.fat_tail_graph.setFixedHeight(height)
        self.fat_tail_graph.setFixedWidth(width)

        self.layout_.addWidget(pg_layout)
        self.setLayout(self.layout_)

        # ---------------------------------------------------------------------------------------------------------------
        p6 = pg_layout.addPlot(row=3,
                               col=0, title="Black/Red ratio")

        p6.setTitle(title="Black/Red ratio",
                    color=(0, 0, 0), fontsize=20)
        y = np.zeros(20)
        x = np.linspace(0, 20, 20)
        self.red_ratio_plot = pg.PlotCurveItem(
            x, y, pen={'color': (255, 0, 0), 'width': 2})
        self.black_ratio_plot = pg.PlotCurveItem(
            x, y, pen={'color': (0, 0, 0), 'width': 2})
        p6.showGrid(x=True, y=True)
        p6.addItem(self.red_ratio_plot)
        p6.addItem(self.black_ratio_plot)
        p6.setFixedHeight(height)
        p6.setFixedWidth(width)

    def change_prime(self):
        global isPrimed
        isPrimed = not isPrimed

    def update_preview(self, grid_size, elite_size, iter_counts, elite_cap, red_cf_ratio):
        global matrix, iterations, black_ratio, red_ratio, isPrimed, red_config_ratio
        iterations = iter_counts
        red_config_ratio = red_cf_ratio
        matrix = sandpile_elite.initialize_pile(
            grid_size, grid_size, (elite_size, elite_size))
        sandpile_elite.elite_cap = elite_cap
        black_ratio, red_ratio = [], []
        self.elite_grid.setImage(
            sandpile_elite.masking, cmap='gray', autoRange=False)
        if not isPrimed:
            self.p1.setTitle(title="Elite size configuration",
                             color=(0, 0, 0), fontsize=20)
        else:
            self.p1.setTitle(title="Primed Elite size configuration",
                             color=(0, 0, 0), fontsize=20)
        return
