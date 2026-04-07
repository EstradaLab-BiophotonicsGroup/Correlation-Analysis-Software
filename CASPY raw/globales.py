from matplotlib.colors import LinearSegmentedColormap, Colormap

import tkinter as tk
import math
from tkinter import filedialog, ttk
from tkinter import Frame, Label, Toplevel, Entry, IntVar, messagebox
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import ticker #as mticker
import matplotlib.colors as colors
from czifile import imread
import threading
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from tkinter import simpledialog
from lfdfiles import SimfcsB64 as lfd
import webbrowser
import sys
import numpy as np
import os
import tqdm
import importlib.util
from tifffile import imread as tif_imread
import traceback
import matplotlib.cm as cm

# ██████  ██       ██████  ██████   █████  ██          ██    ██  █████  ██████  ██  █████  ██████  ██      ███████ ███████ 
#██       ██      ██    ██ ██   ██ ██   ██ ██          ██    ██ ██   ██ ██   ██ ██ ██   ██ ██   ██ ██      ██      ██      
#██   ███ ██      ██    ██ ██████  ███████ ██          ██    ██ ███████ ██████  ██ ███████ ██████  ██      █████   ███████ 
#██    ██ ██      ██    ██ ██   ██ ██   ██ ██           ██  ██  ██   ██ ██   ██ ██ ██   ██ ██   ██ ██      ██           ██ 
# ██████  ███████  ██████  ██████  ██   ██ ███████       ████   ██   ██ ██   ██ ██ ██   ██ ██████  ███████ ███████ ███████ 
original_kimograms = []  # Store kimograms for multiple files
fig = None
canvas = None
original_xlim = None  # To store original x-axis limits
original_ylim = None  # To store original y-axis limits
global last_fig, G_to_save, T_to_save, current_figures, images
images = []
current_figures = []
last_fig = None
ax = None
canvas = None
current_ax = None
rect_selector = None
current_index = 0  # Track current kimogram index
G_to_save = None
T_to_save = None
dwell_time = None
pixels = None
table_frame = None
image_frame = None
working_label = None
result_label = None
file_list_label = None
global mask
mask = None
global factor
factor = 1
original_limits = {}  # This will store original limits for each axis.
button_style = {
            'bg': 'lightgrey',
            'fg': 'black',
            'padx': 12,
            'pady': 6,
            'relief': 'flat',
            'borderwidth': 2,
            'highlightbackground': '#d3d3d3',
            'highlightcolor': '#a9a9a9',
            'activebackground': '#5a5a5a',
            'activeforeground': 'white'
        }

from pathlib import Path

APP_NAME = "CASPY"

def get_user_data_dir():
    base = Path.home() / "Documents" / APP_NAME
    base.mkdir(exist_ok=True)
    return base

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



def show_message(kind, message):
    '''
    this function is for creating error messages when something has gone wrong. 
    The message given will depend on which function is crushing.

    Parameters
    ----------
    message : TYPE string
        DESCRIPTION. Message to be shown

    Returns
    -------
    None.

    '''    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if kind=='Error':
        messagebox.showerror(kind, message)
    else:
        messagebox.showinfo(kind, message)
    root.destroy()  # Close the window
    

def my_colors(cmap_var):
    if isinstance(cmap_var, LinearSegmentedColormap):
        return cmap_var

    if hasattr(cmap_var, 'get'):
        cmap_name = cmap_var.get()
    else:
        cmap_name = str(cmap_var)

    if not cmap_name or cmap_name == 'choose color map':
        cmap_name = 'smooth_viridis'

    colores = {
    'smooth_viridis': ["black", "#440154", "#482677", "#3e4a89", "#2a788e", "#22a884", "#7dcd3e", "#fde725"],
    'plasma': ["#0d0887", "#46039f", "#7201a8", "#ab5b88", "#d8d83c", "#f0f921"],
    'cividis': ["#00224c", "#3b5c96", "#66a85f", "#c7c41f", "#f4f824"],
    'jet': ["#000000", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"],#["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"],
    'gray': ["#000000", "#1c1c1c", "#383838", "#555555", "#717171", "#8d8d8d", "#aaaaaa", "#c6c6c6", "#e2e2e2", "#ffffff"]
    }
    if cmap_name in colores:
        return LinearSegmentedColormap.from_list(cmap_name, colores[cmap_name])
    else:
        try:
            return plt.colormaps.get(cmap_name)
        except ValueError:
            print(f"[WARNING] Colormap '{cmap_name}' no reconocido. Usando 'smooth_viridis'.")
            return LinearSegmentedColormap.from_list('smooth_viridis', colores['smooth_viridis'])

