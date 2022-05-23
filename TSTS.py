#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Luis Bonah
# Description : Convert time signal via FFT to Frequency Spectrum

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector, CheckButtons, RadioButtons
from matplotlib import gridspec


readfile_kwargs = {
	"usecols": (3, 4),
	"skip_header": 6,
	"delimiter": ",",
}

savefile_kwargs = {
	"delimiter": "\t",
}


def zeropadding(xs, ys):
	num_add = int(2**np.ceil(np.log2(len(xs)))-len(xs))
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

def calc_window(windowtype, ys):
	functions = {
		"hanning": np.hanning,
		"blackman": np.blackman,
		"hamming": np.hamming,
		"bartlett": np.bartlett,
	}

	if windowtype in functions:
		return(functions[windowtype](len(ys)))

	return(np.ones(len(ys)))

def calc_fft(data, fft_range, windowtype, zeropad):
	if data is None:
		return([], [], [], [])

	time_xs, time_ys = data[:, 0], data[:, 1]
	fft_min, fft_max = fft_range

	mask = (time_xs > fft_min) & (time_xs < fft_max)
	xs, ys = time_xs[mask], time_ys[mask]

	if zeropad:
		xs, ys = zeropadding(xs, ys)

	window = calc_window(windowtype, ys)

	N = len(xs)
	if N:
		spec_xs = np.fft.fftfreq(N, (min(xs)-max(xs))/N)
		spec_ys = np.fft.fft(ys*window)

		# Only positive frequencies
		mask = (spec_xs > 0)
		spec_xs = spec_xs[mask]
		spec_ys = abs(spec_ys[mask])
	else:
		spec_xs = []
		spec_ys = []

	return(time_xs, time_ys, spec_xs, spec_ys)

def get_file():
	try:
		import tkinter as tk
		from tkinter import filedialog
		root = tk.Tk()
		root.withdraw()
		filename = filedialog.askopenfilename()
		root.destroy()
	except Exception as E:
		filename = input("Enter filename: ")

	if not filename:
		data = None
	else:
		data = np.genfromtxt(filename, **readfile_kwargs)
	return(data, filename)

def fft_timesignal(save_data):
	global data, filename, fft_range, spec_data

	fft_range = [0, 0]
	data, filename = get_file()

	def onselect(xmin, xmax):
		global fft_range
		fft_range = (xmin, xmax)
		update_plot()

	def press(key):
		global data, filename

		if key in ["enter"]:
			save_data(spec_data, filename, fft_range, window_button.value_selected, zeropad_button.get_status()[0])
		elif key in ["esc"]:
			fig.canvas.mpl_disconnect(cid)
			plt.close()
		elif key in ["ctrl+r"]:
			span.set_visible(False)
			span.onselect(0, 0)
			update_plot(force_rescale=True)
		elif key in ["ctrl+s"]:
			try:
				import tkinter as tk
				from tkinter import filedialog
				root = tk.Tk()
				root.withdraw()
				filename = filedialog.asksaveasfilename()
				root.destroy()
			except Exception as E:
				print(str(E))
				filename = input("Enter filename: ")
			plt.savefig(filename)
		elif key in ["ctrl+o"]:
			data, filename = get_file()
			update_plot()

	def update_plot(force_rescale=False):
		global data, filename, fft_range, spec_data, window, zeropad

		window, zeropad = window_button.value_selected, zeropad_button.get_status()[0]
		xs, ys, spec_xs, spec_ys = calc_fft(data, fft_range, window, zeropad)
		spec_data = np.array((spec_xs, spec_ys)).T

		ax0.lines[0].set_data(xs, ys)
		ax1.lines[0].set_data(spec_xs, spec_ys)

		if rescale_button.get_status()[0] or force_rescale:
			ax0.set_xlim(calc_range(xs, margin=0))
			ax0.set_ylim(calc_range(ys))

			ax1.set_xlim(calc_range(spec_xs, margin=0))
			ax1.set_ylim(calc_range(spec_ys))

		title_ax.set_title(f"{os.path.basename(filename)}", ha="center")
		fig.canvas.draw_idle()

	fig = plt.figure()
	gs = gridspec.GridSpec(7, 12, height_ratios = [0.25, 1, 0.5, 1, 0.5, 0.5, 0.5], hspace = 0, wspace=0)

	title_ax = fig.add_subplot(gs[0, :])
	title_ax.axis("off")
	title_ax.set_title("Press 'Replace Files' to open files")

	ax0 = fig.add_subplot(gs[1, :])
	ax0.plot([], [], color="#FF0266", label="Time series")
	ax0.legend(loc = "upper right")
	span = SpanSelector(ax0, onselect, interactive=True, drag_from_anywhere=True, direction="horizontal")

	tmp_ax = fig.add_subplot(gs[2, :])
	tmp_ax.axis("off")

	ax1 = fig.add_subplot(gs[3, :])
	ax1.plot([], [], color="#0336FF", label="Frequency Spectrum")
	ax1.legend(loc = "upper right")

	tmp_ax = fig.add_subplot(gs[4, :])
	tmp_ax.axis("off")

	window_button = RadioButtons(fig.add_subplot(gs[5:7, 0:3]), ("hanning", "blackman", "hamming", "bartlett", "boxcar"), active=0)
	window_button.on_clicked(lambda x: update_plot())

	rescale_button = CheckButtons(fig.add_subplot(gs[5, 3:6]), ("Rescale", ), (True, ))
	rescale_button.on_clicked(lambda x: update_plot())
	zeropad_button = CheckButtons(fig.add_subplot(gs[6, 3:6]), ("Zeropad", ), (False, ))
	zeropad_button.on_clicked(lambda x: update_plot())

	file_button = Button(fig.add_subplot(gs[6, 6:9]), "Open File")
	file_button.on_clicked(lambda x: press("ctrl+o"))

	reset_button = Button(fig.add_subplot(gs[5, 6:9]), "Reset")
	reset_button.on_clicked(lambda x: press("ctrl+r"))

	savefigure_button = Button(fig.add_subplot(gs[5, 9:12]), "Save Figure")
	savefigure_button.on_clicked(lambda x: press("ctrl+s"))

	save_button = Button(fig.add_subplot(gs[6, 9:12]), "Save")
	save_button.on_clicked(lambda x: press("enter"))

	cid = fig.canvas.mpl_connect('key_press_event', lambda event: press(event.key))

	update_plot(force_rescale=True)

	fig.tight_layout()
	plt.show()


if __name__ == '__main__':
	# Set up what should happen with corrected data; in this case save to file
	def save_data(data, filename, fft_range, window, zeropad):
		data[:,0] /= 1E6
		header = f"Range of window: {fft_range}\nWindowfunction: {window}\nZeropadding: {zeropad}"
		fname, extension = os.path.splitext(filename)
		np.savetxt(fname + "FFT" + extension, data, header=header, **savefile_kwargs)

	# Start main function
	fft_timesignal(save_data)
