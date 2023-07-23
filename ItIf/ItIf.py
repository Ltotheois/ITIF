#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Luis Bonah
# Description: Program to create CSV files for the AWG


import os
import sys
import json
import traceback as tb
import numpy as np
import pandas as pd
import threading
import matplotlib
from matplotlib import gridspec, figure
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT

from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

QLocale.setDefault(QLocale("en_EN"))

homefolder = os.path.join(os.path.expanduser("~"), ".itif")
if not os.path.isdir(homefolder):
    os.mkdir(homefolder)

OPTIONFILE = os.path.join(homefolder, "default.json")
PYQT_MAX = 2147483647

matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['patch.facecolor'] = "blue"


### Functions for the FFT conversion
def zeropadding(xs, ys, N_zeroes=0):
    if N_zeroes == 0:
        return(xs, ys)
    
    if not len(xs):
        return(np.array([]), np.array([]))

    if N_zeroes < 0:
        num_add = int(2**np.ceil(np.log2(len(xs)))-len(xs))
    else:
        num_add = N_zeroes

    dt = xs[1] - xs[0] if len(xs) > 1 else 0
    xs = np.pad(xs, (0, num_add), "linear_ramp", end_values=(0, xs[-1] + num_add*dt))
    ys = np.pad(ys, (0, num_add), "constant", constant_values=(0, 0))
    return(xs, ys)

def calc_range(ys, margin=0.1):
    if not len(ys):
        return((0, 1))
    ymin = np.min(ys)
    ymax = np.max(ys)
    ydiff = ymax-ymin

    return((ymin-margin*ydiff, ymax+margin*ydiff))

WINDOWFUNCTIONS = ("hanning", "blackman", "hamming", "bartlett", "boxcar")
def calc_window(windowtype, N):
    functions = {
        "hanning": np.hanning,
        "blackman": np.blackman,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
    }

    if windowtype in functions:
        return(functions[windowtype](N))

    return(np.ones(N))

def calc_fft(data, config):
    if data is None:
        return(np.array([]), np.array([[]]), np.array([]), np.array([]))
    data = data.copy()

    time_xs, time_yss = data[:, 0], data[:, 1:]
    
    number_of_phases = time_yss.shape[1]
    if number_of_phases == 1:
        time_ys = time_yss[:, 0]
    elif number_of_phases == 4:
        I = (time_yss[:, 0] - time_yss[:, 2]) / 2
        Q = (time_yss[:, 3] - time_yss[:, 1]) / 2
        time_ys = I + 1j * Q
    elif number_of_phases == 8:
        I = (time_yss[:, 0] - time_yss[:, 2] + time_yss[:, 4] - time_yss[:, 6]) / 4
        Q = (time_yss[:, 3] - time_yss[:, 1] + time_yss[:, 7] - time_yss[:, 5]) / 4
        time_ys = I + 1j * Q
    else:
        self.notification("The number of different phases has to be 1, 2, or 8 but was {number_of_phases}.")
        return
    
    time_xs /= 10**config["tvaluesunit"]

    fft_min, fft_max = config["windowstart"], config["windowstop"]
    if fft_min > fft_max:
        fft_min, fft_max = fft_max, fft_min

    mask = (time_xs > fft_min) & (time_xs < fft_max)
    xs, ys = time_xs[mask], time_ys[mask]

    xs, ys = zeropadding(xs, ys, config["zeropad"])

    xs *= 10**config["tvaluesunit"]

    N = len(xs)
    window = calc_window(config["windowfunction"], N)
    if N:
        spec_xs = np.fft.fftfreq(N, (min(xs)-max(xs))/N)
        spec_xs /= 10**config["xvaluesunit"]
        spec_ys = np.abs(np.fft.fft(ys*window))


        # @Luis: Change this to have both sidebands
        spec_xs = np.fft.fftshift(spec_xs)
        spec_ys = np.fft.fftshift(spec_ys)
        
        if config["localoscillator"]:
            spec_xs += config["localoscillator"] * 10**(config["lovaluesunit"] - config["xvaluesunit"])
        else:
            mask = (spec_xs > 0)
            spec_xs = spec_xs[mask]
            spec_ys = spec_ys[mask]
        
        if config["limitfrequencies"]:
            xmin, xmax = config["frequencystart"], config["frequencystop"]
            mask = (spec_xs > xmin) & (spec_xs < xmax)
            spec_xs = spec_xs[mask]
            spec_ys = spec_ys[mask]
        
    else:
        spec_xs = np.array([])
        spec_ys = np.array([])

    return(time_xs, time_yss, spec_xs, spec_ys)


### GUI
class QSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRange(0, PYQT_MAX)
        
    def setRange(self, min, max):
        min = min if not min is None else -PYQT_MAX
        max = max if not max is None else +PYQT_MAX
        return super().setRange(min, max)

class QDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(20)
        self.setRange(0, PYQT_MAX)
        
        try:
            self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        except:
            pass

    def textFromValue(self, value):
        return(f"{value:.10f}".rstrip("0").rstrip("."))

    def valueFromText(self, text):
        return(np.float64(text))

    def setRange(self, min, max):
        min = min if not min is None else -np.inf
        max = max if not max is None else +np.inf
        return super().setRange(min, max)



class MainWindow(QMainWindow):
    updateconfig = pyqtSignal(tuple)
    redrawplot = pyqtSignal()
    
    def __init__(self, parent=None):
        global config
        
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.timer = None
        self.update_data_thread = None
        self.update_data_lock = threading.RLock()
        
        self.data = None
        self.ts_are_calculated = False
        self.spec_data = None
        self.fname = None

        config = self.config = Config(self.updateconfig, {
            "mpltoolbar": True,
            "savefigure_kwargs": {
                "dpi": 600,
            },
            "readfile_kwargs": {
                "usecols": (3, 4),
                "skip_header": 6,
                "delimiter": ",",
            },
            "asksavename": False,
            "savefile_kwargs": {
                "delimiter": "\t",
            },
            
            "windowfunction": WINDOWFUNCTIONS[0],
            "windowstart": 0,
            "windowstop": 0,
            "limitfrequencies": False,
            "frequencystart": 0,
            "frequencystop": 30,
            "zeropad": 0,
            "rescale": True,
            "localoscillator": 0,
            "samplerate": 8,
            
            "xvaluesunit": 9,
            "tvaluesunit": -6,
            "lovaluesunit": 9,
            "samplerateunit": 9,
        })

        self.gui()
        self.setWindowTitle("Time Signal to Frequency Spectrum")
        self.readoptions(OPTIONFILE, ignore=True)
        
        for widget, (key, textcreator) in self.dynamic_labels.items():
            widget.setText(textcreator())
        self.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]

        option_files = []
        data_files = []
        
        for file in files:
            if file.endswith(".json"):
                option_files.append(file)
            else:
                data_files.append(file)
        
        if len(data_files) == 1:
            self.open_file(data_files[0])
        elif len(data_files) > 1:
            self.notification("Only drop a single data file.")
        
        if len(option_files) == 1:
            self.readoptions(option_files[0])
        elif len(option_files) > 1:
            self.notification("Only drop a single option file.")

    def gui_menu(self):
        filemenu = self.menuBar().addMenu(f"File")

        filemenu.addAction(QQ(QAction, parent=self, text="&Open File", shortcut="Ctrl+O", tooltip="Open a new file", change=lambda x: self.open_file()))
        filemenu.addAction(QQ(QAction, parent=self, text="&Save File", shortcut="Ctrl+S", tooltip="Save data to file", change=lambda x: self.save_file()))
        filemenu.addSeparator()
        filemenu.addAction(QQ(QAction, parent=self, text="&Open Set of Files", tooltip="Open multiple files all holding a single phase", change=lambda x: self.open_files_set()))
        filemenu.addSeparator()
        filemenu.addAction(QQ(QAction, parent=self, text="&Load Options", tooltip="Load option file", change=lambda x: self.readoptions()))
        filemenu.addAction(QQ(QAction, parent=self, text="&Save Options", tooltip="Save options to file", change=lambda x: self.saveoptions()))
        filemenu.addSeparator()
        filemenu.addAction(QQ(QAction, parent=self, text="&Save as default", tooltip="Save options as default options", change=lambda x: self.saveoptions(OPTIONFILE)))

        actionmenu = self.menuBar().addMenu(f"Actions")
        actionmenu.addAction(QQ(QAction, parent=self, text="&Reset", shortcut="Ctrl+R", tooltip="Reset the plots and span selection", change=lambda x: self.onreset()))
        actionmenu.addSeparator()
        actionmenu.addAction(QQ(QAction, "mpltoolbar", parent=self, text="&Show MPL Toolbar", tooltip="Show or hide matplotlib toolbar", checkable=True))
        actionmenu.addSeparator()
        actionmenu.addAction(QQ(QAction, "rescale", parent=self, text="&Rescale Spectra", tooltip="Rescale spectra when data changes", checkable=True))
        actionmenu.addSeparator()
        actionmenu.addAction(QQ(QAction, parent=self, text="&Save Figure", tooltip="Save the figure", change=lambda x: self.savefigure()))

    def update_frequency_widgets(self):
        enabled = self.config["limitfrequencies"]
        for widget in self.frequency_widgets:
            widget.setEnabled(enabled)

    def gui(self):
        self.gui_menu()
    
        layout = QVBoxLayout()
        self.dynamic_labels = {}

        self.fig = figure.Figure()
        gs = gridspec.GridSpec(4, 1, height_ratios = [0.25, 1, 0.5, 1], hspace = 0, wspace=0)
        self.plotcanvas = FigureCanvas(self.fig)
        self.plotcanvas.setMinimumHeight(200)
        self.plotcanvas.setMinimumWidth(200)
        layout.addWidget(self.plotcanvas)
        
        self.config.register([
            "windowfunction", "windowstart", "windowstop", "zeropad", "localoscillator",
            "xvaluesunit", "tvaluesunit", "lovaluesunit", "limitfrequencies", "frequencystart", "frequencystop",
            ], self.update_data)
        
        self.config.register(["samplerate", "samplerateunit"], self.update_samplerate)
        
        self.redrawplot.connect(self.fig.canvas.draw_idle)
        
        self.title_ax = self.fig.add_subplot(gs[0, :])
        self.title_ax.axis("off")
        self.title_ax.set_title("Press 'Open File' to load data")

        self.ax0 = self.fig.add_subplot(gs[1, :])
        self.timesignal_lines = [self.ax0.plot([], [], color="#FF0266", label="Time series")[0]]
        self.ax0.legend(loc = "upper right")
        self.span = matplotlib.widgets.SpanSelector(self.ax0, self.onselect, interactive=True, drag_from_anywhere=True, direction="horizontal", props=dict(facecolor="#00000040"))
        self.config.register(["windowstart", "windowstop"], self.setspan)
        self.ax0.set_xlabel(f"Time [{unit_to_str(self.config['tvaluesunit'])}s]")

        self.config.register("tvaluesunit", lambda: self.ax0.set_xlabel(f"Time [{unit_to_str(self.config['tvaluesunit'])}s]"))

        tmp_ax = self.fig.add_subplot(gs[2, :])
        tmp_ax.axis("off")

        self.ax1 = self.fig.add_subplot(gs[3, :])
        self.spectrum_line = self.ax1.plot([], [], color="#0336FF", label="Frequency Spectrum")[0]
        self.ax1.legend(loc = "upper right")
        self.ax1.set_xlabel(f"Frequency [{unit_to_str(self.config['xvaluesunit'])}Hz]")

        self.config.register("xvaluesunit", lambda: self.ax1.set_xlabel(f"Frequency [{unit_to_str(self.config['xvaluesunit'])}Hz]"))
        
        self.mpltoolbar = NavigationToolbar2QT(self.plotcanvas, self)
        self.mpltoolbar.setVisible(self.config["mpltoolbar"])
        self.config.register("mpltoolbar", lambda: self.mpltoolbar.setVisible(self.config["mpltoolbar"]))
        self.addToolBar(self.mpltoolbar)

        self.notificationarea = QLabel()
        self.notificationarea.setWordWrap(True)
        self.notificationarea.setHidden(True)
        layout.addWidget(self.notificationarea)

        button_layout = QGridLayout()
        
        column_index = 0
        
        button_layout.addWidget(QLabel("Window Function: "), 0, column_index)
        button_layout.addWidget(QQ(QComboBox, "windowfunction", options=WINDOWFUNCTIONS), 0, column_index+1)

        
        tmp = QLabel()
        self.dynamic_labels[tmp] = ("tvaluesunit", lambda: f"Window Start [{unit_to_str(self.config['tvaluesunit'])}s]: ")
        button_layout.addWidget(tmp, 1, column_index)
        button_layout.addWidget(QQ(QDoubleSpinBox, "windowstart", minWidth=80, range=(None, None)), 1, column_index+1)

        tmp = QLabel()
        self.dynamic_labels[tmp] = ("tvaluesunit", lambda: f"Window Stop [{unit_to_str(self.config['tvaluesunit'])}s]: ")
        button_layout.addWidget(tmp, 2, column_index)
        button_layout.addWidget(QQ(QDoubleSpinBox, "windowstop", minWidth=80, range=(None, None)), 2, column_index+1)
        
        column_index += 2

        button_layout.addWidget(QQ(QLabel, text="Zeropadding"), 0, column_index)
        button_layout.addWidget(QQ(QSpinBox, "zeropad", range=(-1, None)), 0, column_index+1)
        
        tmp = QLabel()
        self.dynamic_labels[tmp] = ("samplerateunit", lambda: f"Samplerate [{unit_to_str(self.config['samplerateunit'])}Hz]:")
        button_layout.addWidget(tmp, 1, column_index)
        button_layout.addWidget(QQ(QDoubleSpinBox, "samplerate", minWidth=80, range=(0, None)), 1, column_index+1)
        
        tmp = QLabel()
        self.dynamic_labels[tmp] = ("lovaluesunit", lambda: f"LO [{unit_to_str(self.config['lovaluesunit'])}Hz]:")
        button_layout.addWidget(tmp, 2, column_index)
        button_layout.addWidget(QQ(QDoubleSpinBox, "localoscillator", minWidth=80, range=(0, None)), 2, column_index+1)
        
        column_index += 2
        
        
        button_layout.addWidget(QQ(QCheckBox, "limitfrequencies", text="Limit Frequencies"), 0, column_index, 1, 2)

        self.frequency_widgets = []
        
        tmp = QLabel()
        self.dynamic_labels[tmp] = ("xvaluesunit", lambda: f"Frequency Start [{unit_to_str(self.config['xvaluesunit'])}Hz]: ")
        self.frequency_widgets.append(tmp)
        button_layout.addWidget(tmp, 1, column_index)
        tmp = QQ(QDoubleSpinBox, "frequencystart", minWidth=80, range=(None, None))
        self.frequency_widgets.append(tmp)
        button_layout.addWidget(tmp, 1, column_index+1)

        tmp = QLabel()
        self.dynamic_labels[tmp] = ("xvaluesunit", lambda: f"Frequency Stop [{unit_to_str(self.config['xvaluesunit'])}Hz]: ")
        self.frequency_widgets.append(tmp)
        button_layout.addWidget(tmp, 2, column_index)
        tmp = QQ(QDoubleSpinBox, "frequencystop", minWidth=80, range=(None, None))
        self.frequency_widgets.append(tmp)
        button_layout.addWidget(tmp, 2, column_index+1)
        
        self.config.register("limitfrequencies", self.update_frequency_widgets)
        self.update_frequency_widgets()

        
        column_index += 2
        
        button_layout.setColumnStretch(column_index, 2)
        
        column_index += 1
        
        button_layout.addWidget(QQ(QPushButton, text="Reset", change=lambda: self.onreset()), 0, column_index)
        button_layout.addWidget(QQ(QPushButton, text="Open File", change=lambda: self.open_file()), 1, column_index)
        button_layout.addWidget(QQ(QPushButton, text="Save File", change=lambda: self.save_file()), 2, column_index)

        layout.addLayout(button_layout)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(layout)

        for widget, (key, textcreator) in self.dynamic_labels.items():
            self.config.register(key, lambda widget=widget, textcreator=textcreator: widget.setText(textcreator()))

    def open_file(self, fname=None):
        if fname is None:
            fname, _ = QFileDialog.getOpenFileName(None, 'Choose File to open',"")
        if not fname:
            self.notification("No file was selected. Keeping current data.")
            return
        
        _, ext = os.path.splitext(fname)
        if ext == ".npz":
            data = np.load(fname)
            yss = data["data"]
            if yss.shape[0] not in [1, 4, 8]:
                self.notification(f"Number of datasets (phases) has to be 1, 4, or 8 but was {yss.shape[0]}.")
                return
            N = yss.shape[1]
            
            xs = self.create_xs(N)
            self.data = np.vstack((xs, *yss)).T
            self.ts_are_calculated = True
        else:
            self.data = np.genfromtxt(fname, **self.config["readfile_kwargs"])
            self.ts_are_calculated = False
        
        self.fname = fname
        self.update_data()
    
    def open_files_set(self, fnames=None):
        if fnames is None:
            fnames, _ = QFileDialog.getOpenFileNames(None, 'Choose Files to open',"")
        if not fnames:
            self.notification("No files were selected. Keeping current data.")
            return
    
        fnames = sorted(fnames)
        yss = []
        for fname in fnames:
            ys = np.loadtxt(fname)
            yss.append(ys)

        N = len(yss[0])
        xs = self.create_xs(N)
        self.data = np.vstack((xs, *yss)).T
        self.ts_are_calculated = True
        
        self.fname = fnames[0]
        self.update_data()
    
    def create_xs(self, N):
        xs = np.linspace(0, N, N) / (self.config["samplerate"] * 10**self.config["samplerateunit"])
        return(xs)
        
    def save_file(self, fname=None):
        if self.spec_data is None:
            self.notification("No data to save.")
            return
        
        if self.config["asksavename"]:
            savename, _ = QFileDialog.getSaveFileName(None, 'Choose File to Save to',"")
        else:
            fname = self.fname
            basename, extension = os.path.splitext(fname)
            savename = basename + "FFT" + extension
        
        if not fname:
            self.notification("No filename specified for saving.")
            return

        data = self.spec_data
        
        header = f"Window: {self.config['windowfunction']} from {self.config['windowstart']} to {self.config['windowstop']}\nZeropadding: {self.config['zeropad']}"
        np.savetxt(savename, data, header=header, **self.config["savefile_kwargs"])
        self.notification(f"Saved data successfully to '{fname}'")

    def notification(self, text):
        self.notificationarea.setText(text)
        self.notificationarea.setHidden(False)

        if self.timer:
            self.timer.stop()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.notificationarea.setHidden(True))
        self.timer.start(5000)

    def saveoptions(self, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save options to file")
            if not filename:
                return

        with open(filename, "w+") as optionfile:
            json.dump(self.config, optionfile, indent=2)
        self.notification("Options have been saved")

    def readoptions(self, filename=None, ignore=False):
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(self, "Read options from file")
            if not filename:
                return

        if not os.path.isfile(filename):
            if not ignore:
                self.notification(f"Option file '{filename}' does not exist.")
            return

        with open(filename, "r") as optionfile:
            values = json.load(optionfile)
        for key, value in values.items():
            self.config[key] = value
        self.notification("Options have been loaded")


    def savefigure(self):
        fname, _ = QFileDialog.getSaveFileName(None, 'Choose File to Save to',"")
        if not fname:
            return
        
        self.fig.savefig(fname, **config["savefigure_kwargs"])

    def update_samplerate(self):
        if not self.ts_are_calculated:
            return
        self.data[:, 0] = self.create_xs(len(self.data[:, 0]))
        
        self.update_data()

    def update_data(self, force_rescale=False):
        thread = threading.Thread(target=self.update_data_core, args=(force_rescale, ))
        with self.update_data_lock:
            thread.start()
            self.update_data_thread = thread.ident
        return(thread)

    def update_data_core(self, force_rescale):
        with self.update_data_lock:
            ownid = threading.current_thread().ident
        
        try:
            if self.data is None:
                return
            breakpoint(ownid, self.update_data_thread)
            time_xs, time_yss, spec_xs, spec_ys = calc_fft(self.data, self.config)
            
            breakpoint(ownid, self.update_data_thread)
            self.spec_data = np.vstack((spec_xs, spec_ys)).T
            
            number_of_datasets = time_yss.shape[1]
            
            for line in self.timesignal_lines:
                line.remove()
            self.timesignal_lines = []
            
            colors = ["#fbd206", "#feaf8a", "#cc89d6", "#bfcff0", "#9ce7c9", "#4dc656", "#D1BDFF", "#fd7a8c", ]
            
            for i, (time_ys, color) in enumerate(zip(time_yss.T, colors)):
                self.timesignal_lines.append(self.ax0.plot(time_xs, time_ys, color=color, label=f"Series {i}")[0])
            self.ax0.legend(loc = "upper right")
            
            self.spectrum_line.set_data(spec_xs, spec_ys)
            breakpoint(ownid, self.update_data_thread)

            if config["rescale"] or force_rescale:
                self.ax0.set_xlim(calc_range(time_xs, margin=0))
                self.ax0.set_ylim(calc_range(time_yss))

                self.ax1.set_xlim(calc_range(spec_xs, margin=0))
                self.ax1.set_ylim(calc_range(spec_ys))
            breakpoint(ownid, self.update_data_thread)

            if self.fname:
                self.title_ax.set_title(f"{os.path.basename(self.fname)}", ha="center")
            else:
                self.title_ax.set_title("Press 'Open File' to load data", ha="center")
            breakpoint(ownid, self.update_data_thread)
            
            self.redrawplot.emit()
        except BreakpointError as E:
            pass

    def onselect(self, xmin, xmax):
        self.config["windowstart"] = xmin
        self.config["windowstop"] = xmax

    def setspan(self):
        self.span.extents = self.config["windowstart"], self.config["windowstop"]
    
    def onreset(self):
        self.span.set_visible(False)
        self.span.onselect(0, 0)
        self.update_data(force_rescale=True)

class Config(dict):
    def __init__(self, signal, init_values={}):
        super().__init__(init_values)
        self.signal = signal
        self.signal.connect(self.callback)
        self.callbacks = pd.DataFrame(columns=["id", "key", "widget", "function"], dtype="object").astype({"id": np.uint})

    def __setitem__(self, key, value, widget=None):
        super().__setitem__(key, value)
        self.signal.emit((key, value, widget))

    def callback(self, args):
        key, value, widget = args
        if widget:
            callbacks_widget = self.callbacks.query(f"key == @key and widget != @widget")
        else:
            callbacks_widget = self.callbacks.query(f"key == @key")
        for i, row in callbacks_widget.iterrows():
            row["function"]()

    def register(self, keys, function):
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for key in keys:
            id = 0
            df = self.callbacks
            df.loc[len(df), ["id", "key", "function"]] = id, key, function

    def register_widget(self, key, widget, function):
        ids = set(self.callbacks["id"])
        id = 1
        while id in ids:
            id += 1
        df = self.callbacks
        df.loc[len(df), ["id", "key", "function", "widget"]] = id, key, function, widget
        widget.destroyed.connect(lambda x, id=id: self.unregister_widget(id))

    def unregister_widget(self, id):
        self.callbacks.drop(self.callbacks[self.callbacks["id"] == id].index, inplace=True)

def QQ(widgetclass, config_key=None, **kwargs):
    widget = widgetclass()

    if "range" in kwargs:
        widget.setRange(*kwargs["range"])
    if "maxWidth" in kwargs:
        widget.setMaximumWidth(kwargs["maxWidth"])
    if "maxHeight" in kwargs:
        widget.setMaximumHeight(kwargs["maxHeight"])
    if "minWidth" in kwargs:
        widget.setMinimumWidth(kwargs["minWidth"])
    if "minHeight" in kwargs:
        widget.setMinimumHeight(kwargs["minHeight"])
    if "color" in kwargs:
        widget.setColor(kwargs["color"])
    if "text" in kwargs:
        widget.setText(kwargs["text"])
    if "options" in kwargs:
        options = kwargs["options"]
        if isinstance(options, dict):
            for key, value in options.items():
                widget.addItem(key, value)
        else:
            for option in kwargs["options"]:
                widget.addItem(option)
    if "width" in kwargs:
        widget.setFixedWidth(kwargs["width"])
    if "tooltip" in kwargs:
        widget.setToolTip(kwargs["tooltip"])
    if "placeholder" in kwargs:
        widget.setPlaceholderText(kwargs["placeholder"])
    if "singlestep" in kwargs:
        widget.setSingleStep(kwargs["singlestep"])
    if "wordwrap" in kwargs:
        widget.setWordWrap(kwargs["wordwrap"])
    if "align" in kwargs:
        widget.setAlignment(kwargs["align"])
    if "rowCount" in kwargs:
        widget.setRowCount(kwargs["rowCount"])
    if "columnCount" in kwargs:
        widget.setColumnCount(kwargs["columnCount"])
    if "move" in kwargs:
        widget.move(*kwargs["move"])
    if "default" in kwargs:
        widget.setDefault(kwargs["default"])
    if "textFormat" in kwargs:
        widget.setTextFormat(kwargs["textFormat"])
    if "checkable" in kwargs:
        widget.setCheckable(kwargs["checkable"])
    if "shortcut" in kwargs:
        widget.setShortcut(kwargs["shortcut"])
    if "parent" in kwargs:
        widget.setParent(kwargs["parent"])
    if "completer" in kwargs:
        widget.setCompleter(kwargs["completer"])
    if "hidden" in kwargs:
        widget.setHidden(kwargs["hidden"])
    if "visible" in kwargs:
        widget.setVisible(kwargs["visible"])
    if "stylesheet" in kwargs:
        widget.setStyleSheet(kwargs["stylesheet"])
    if "enabled" in kwargs:
        widget.setEnabled(kwargs["enabled"])
    if "items" in kwargs:
        for item in kwargs["items"]:
            widget.addItem(item)
    if "readonly" in kwargs:
        widget.setReadOnly(kwargs["readonly"])
    if "prefix" in kwargs:
        widget.setPrefix(kwargs["prefix"])

    if widgetclass in [QSpinBox, QDoubleSpinBox]:
        setter = widget.setValue
        changer = widget.valueChanged.connect
        getter = widget.value
    elif widgetclass == QCheckBox:
        setter = widget.setChecked
        changer = widget.stateChanged.connect
        getter = widget.isChecked
    elif widgetclass == QPlainTextEdit:
        setter = widget.setPlainText
        changer = widget.textChanged.connect
        getter = widget.toPlainText
    elif widgetclass == QLineEdit:
        setter = widget.setText
        changer = widget.textChanged.connect
        getter = widget.text
    elif widgetclass == QAction:
        setter = widget.setChecked
        changer = widget.triggered.connect
        getter = widget.isChecked
    elif widgetclass == QPushButton:
        setter = widget.setDefault
        changer = widget.clicked.connect
        getter = widget.isDefault
    elif widgetclass == QToolButton:
        setter = widget.setChecked
        changer = widget.clicked.connect
        getter = widget.isChecked
    elif widgetclass == QComboBox:
        setter = widget.setCurrentText
        changer = widget.currentTextChanged.connect
        getter = widget.currentText
    else:
        return widget

    if "value" in kwargs:
        setter(kwargs["value"])
    if config_key:
        setter(config[config_key])
        changer(lambda x=None, key=config_key: config.__setitem__(key, getter(), widget))
        config.register_widget(config_key, widget, lambda: setter(config[config_key]))
    if "change" in kwargs:
        changer(kwargs["change"])
    if "changes" in kwargs:
        for change in kwargs["changes"]:
            changer(change)

    return widget

class BreakpointError(Exception):
    pass

def breakpoint(ownid, lastid):
    if ownid != lastid:
        raise BreakpointError()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    window.notification(f"{exception}\n{''.join(tb.format_tb(traceback))}")

def unit_to_str(exponent):
    if exponent in UNITS:
        return(UNITS[exponent])
    else:
        return(f"1E{exponent:.0f}")

UNITS = {
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    -3: "m",
    -6: "µ",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
}


def start():
    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    start()
