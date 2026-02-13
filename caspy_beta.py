#%%
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
    
def show_loading_popup(pcf_window):
    loading_win = tk.Toplevel()
    loading_win.title("Loading")
    loading_win.geometry("200x100")
    loading_win.resizable(False, False)

    label = tk.Label(loading_win, text="Loading files...", font=("Arial", 12))
    label.pack(expand=True, fill='both', padx=20, pady=20)

    # Disable closing the window manually
    loading_win.protocol("WM_DELETE_WINDOW", lambda: None)
    loading_win.transient(pcf_window)  # Keep it above main window
    loading_win.grab_set()       # Block other UI interaction

    return loading_win

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



def on_cmap_change(event):
    global cmap_var, current_figures, original_kimograms

    selected_cmap = cmap_var.get()
    if selected_cmap == 'smooth_viridis':
        selected_cmap = 'viridis'

    for fig in current_figures:
        if not hasattr(fig, 'colorbars'):
            fig.colorbars = []

        for ax in fig.axes:
            # Remove previous collections
            for collection in ax.collections:
                collection.remove()

            # Plot new kimogram with selected cmap
            kimogram = original_kimograms[0]

            try:
                new_im = ax.pcolor(kimogram.T, cmap=selected_cmap, shading='nearest')

                # Remove previous colorbars safely
                for cbar in fig.colorbars:
                    cbar.remove()
                fig.colorbars.clear()

                # Add new colorbar and track it
                cbar = fig.colorbar(new_im, ax=ax, orientation='vertical')
                fig.colorbars.append(cbar)

            except Exception as e:
                print("Error:", e)

        fig.canvas.draw_idle()

def exp_decay(t,A,b,c):
    return A*np.exp(-t*b)+c

def linear_decay(t,A,b):
    return -A*t+b


def correct_bleaching(correction='exp', downsample_factor=100):
    global original_kimograms, current_index
    try:
        original_kimogram=original_kimograms[current_index]
    except:
        print('no current axis')
        original_kimogram=original_kimograms[0]
    average = np.mean(original_kimogram, axis=1)
    n_lines = len(average)
    
    # Downsample to reduce data size
    if downsample_factor > 1:
        n_blocks = n_lines // downsample_factor
        avg_ds = average[:n_blocks * downsample_factor].reshape(n_blocks, downsample_factor).mean(axis=1)
        x_ds = np.arange(n_blocks) * downsample_factor
    else:
        avg_ds = average
        x_ds = np.arange(n_lines)
    
    if correction == 'exp':
        # Fit exponential to downsampled data
        p0 = [avg_ds[0], 1e-5, avg_ds[-1]]
        popt, _ = curve_fit(exp_decay, x_ds, avg_ds, p0=p0, maxfev=10000)
        A, b, c = popt
        
        # Evaluate the fitted decay at full resolution
        x_full = np.arange(n_lines)
        fitted_decay = exp_decay(x_full, A, b, c)
        
        # Normalize to 1 at start (optional)
        fitted_decay_norm = fitted_decay / fitted_decay[0]
    elif correction=='linear':
        popt, _ = curve_fit(linear_decay, x_ds, avg_ds, maxfev=10000)
        A, b = popt
        
        # Evaluate the fitted decay at full resolution
        x_full = np.arange(n_lines)
        fitted_decay = linear_decay(x_full, A, b)
        fitted_decay_norm = fitted_decay / fitted_decay[0]
    corrected_kimogram = original_kimogram / fitted_decay_norm[:, np.newaxis]
    corrected_kimogram = np.clip(corrected_kimogram, 0, None)
    try:
            original_kimograms[current_index] = corrected_kimogram
            show_message('Done', 'The data has been corrected')
    except:
            original_kimograms[0] = corrected_kimogram
            show_message('Done', 'The data has been corrected')
    return 

def correct_bleaching_images(correction='exp', downsample_factor=10):
    global stack

    # --- If single image, do nothing ---
    if stack.ndim < 3:
        show_message('Info', 'Single image detected — bleaching correction skipped.')
        return

    n_frames = stack.shape[0]
    avg_intensity = stack.mean(axis=(1, 2))  # mean intensity per frame

    # --- Downsample intensity curve ---
    if downsample_factor > 1 and n_frames > downsample_factor:
        n_blocks = n_frames // downsample_factor
        avg_ds = avg_intensity[:n_blocks * downsample_factor].reshape(n_blocks, downsample_factor).mean(axis=1)
        x_ds = np.arange(n_blocks) * downsample_factor
    else:
        avg_ds = avg_intensity
        x_ds = np.arange(n_frames)

    # --- Fit decay ---
    if correction == 'exp':
        # Exponential model: A * exp(-b*t) + c
        p0 = [avg_ds[0], 1e-5, avg_ds[-1]]
        popt, _ = curve_fit(exp_decay, x_ds, avg_ds, p0=p0, maxfev=10000)
        A, b, c = popt
        fitted_decay = exp_decay(np.arange(n_frames), A, b, c)
    elif correction == 'linear':
        # Linear model: A + B*t
        popt, _ = curve_fit(linear_decay, x_ds, avg_ds, maxfev=10000)
        A, B = popt
        fitted_decay = linear_decay(np.arange(n_frames), A, B)
    else:
        show_message('Error', f"Unknown correction type: {correction}")
        return

    # --- Normalize decay to 1 at t=0 and apply ---
    fitted_decay_norm = fitted_decay / fitted_decay[0]
    fitted_decay_norm = np.clip(fitted_decay_norm, 1e-6, None)

    corrected_stack = stack / fitted_decay_norm[:, np.newaxis, np.newaxis]
    corrected_stack = np.clip(corrected_stack, 0, None)
    stack = corrected_stack
    show_message('Done', 'Bleaching correction applied.')
    
    
from scipy.signal import correlate
from scipy.ndimage import shift as ndi_shift

def detrend_kimogram_linear_sections_cumulative(kimogram,
                                                segment_size=20000,
                                                ref_lines=200,
                                                max_shift=30,
                                                fill_random=True,
                                                random_seed=0):
    """
    Detrend spatial drift in sections using cumulative alignment.
    Each segment is aligned relative to the previous detrended segment,
    avoiding discontinuities at segment boundaries.
    """
    np.random.seed(random_seed)
    kimogram = kimogram.astype(float)
    n_lines, n_pixels = kimogram.shape
    detrended = np.zeros((n_lines, n_pixels), dtype=float)
    shifts = np.zeros(n_lines, dtype=float)
    
    # Background statistics for random filling
    bg_mean = np.median(kimogram)
    bg_std = np.std(kimogram) * 0.5
    
    n_segments = int(np.ceil(n_lines / segment_size))
    
    for seg in tqdm.tqdm(range(n_segments), desc='Linear detrending by cumulative sections'):
        start = seg * segment_size
        end = min((seg + 1) * segment_size, n_lines)
        segment = kimogram[start:end, :]
        
        # Determine reference
        if seg == 0:
            ref = np.mean(segment[:ref_lines, :], axis=0)
        else:
            ref = np.mean(detrended[start-ref_lines:start, :], axis=0)
        
        # Compute shift per line
        seg_shifts = []
        for line in segment:
            corr = correlate(line - np.mean(line), ref - np.mean(ref), mode='full')
            shift_idx = np.argmax(corr) - len(line) + 1
            shift_idx = np.clip(shift_idx, -max_shift, max_shift)
            seg_shifts.append(shift_idx)
        seg_shifts = np.array(seg_shifts)
        
        # Fit linear trend to this segment
        x = np.arange(len(seg_shifts))
        coef = np.polyfit(x, seg_shifts, 1)
        trend = np.polyval(coef, x)
        shifts[start:end] = trend
        
        # Apply shift correction
        for i, (line, s) in enumerate(zip(segment, trend)):
            shifted = ndi_shift(line, -s, mode='constant', cval=0, order=1)
            
            if fill_random:
                int_s = int(np.round(s))
                if int_s > 0:
                    fill_len = min(int_s, n_pixels)
                    shifted[:fill_len] = np.random.normal(bg_mean, bg_std, fill_len)
                elif int_s < 0:
                    fill_len = min(abs(int_s), n_pixels)
                    shifted[-fill_len:] = np.random.normal(bg_mean, bg_std, fill_len)
            
            detrended[start + i] = shifted
    
    return detrended, shifts


def detrend_window(parent_window):
    """
    Open a small window to request detrend parameters and apply cumulative detrending.
    Updates the original kimograms plot with detrended ones.
    """
    global original_kimograms, cmap_var, current_index

    # --- Create the parameter window ---
    param_win = tk.Toplevel(parent_window)
    param_win.title("Detrend Parameters")
    
    tk.Label(param_win, text="Segment size (lines):").grid(row=0, column=0, sticky='w')
    seg_entry = tk.Entry(param_win); seg_entry.grid(row=0, column=1)
    seg_entry.insert(0, "20000")
    
    tk.Label(param_win, text="Reference lines:").grid(row=1, column=0, sticky='w')
    ref_entry = tk.Entry(param_win); ref_entry.grid(row=1, column=1)
    ref_entry.insert(0, "200")
    
    tk.Label(param_win, text="Max shift (px):").grid(row=2, column=0, sticky='w')
    shift_entry = tk.Entry(param_win); shift_entry.grid(row=2, column=1)
    shift_entry.insert(0, "30")
    
    fill_var = tk.BooleanVar(value=True)
    tk.Checkbutton(param_win, text="Fill edges with random noise", variable=fill_var).grid(row=3, column=0, columnspan=2, sticky='w')
    
    tk.Label(param_win, text="Random seed:").grid(row=4, column=0, sticky='w')
    seed_entry = tk.Entry(param_win); seed_entry.grid(row=4, column=1)
    seed_entry.insert(0, "0")
    
    # --- Function to run detrending ---
    def run_detrend():
        try:
            segment_size = int(seg_entry.get())
            ref_lines = int(ref_entry.get())
            max_shift = int(shift_entry.get())
            fill_random = fill_var.get()
            random_seed = int(seed_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer parameters.")
            return
        
        param_win.destroy()  # Close the parameter window

        try:
            kimogram = original_kimograms[current_index]
        except:
            kimogram = original_kimograms[0]
        
        # --- Apply detrending ---
        detrended, shifts = detrend_kimogram_linear_sections_cumulative(
            kimogram,
            segment_size=segment_size,
            ref_lines=ref_lines,
            max_shift=max_shift,
            fill_random=fill_random,
            random_seed=random_seed
        )
        
        # --- Update the stored data ---
        try:
            original_kimograms[current_index] = detrended
        except:
            original_kimograms[0] = detrended
        
        # --- Update plot in parent window ---
        display_kimograms([detrended], cmap_var)  # must be a list!
        
        # --- Show drift plot in a new figure safely ---
        fig2, ax2 = plt.subplots()
        ax2.plot(shifts)
        ax2.set_title('Estimated drift (px)')
        ax2.set_xlabel('Line')
        ax2.set_ylabel('Shift (px)')
        plt.tight_layout()
        plt.show(block=False)
        
        messagebox.showinfo("Detrend", "Cumulative detrending finished.")
    
    ttk.Button(param_win, text="Run Detrend", command=run_detrend).grid(row=5, column=0, columnspan=2, pady=5)


def create_kimogram(lines):
    '''
    this is a simple function to accomodate the given data to be plotted as a kimogram
    Parameters
    ----------
    lines : matrix
        DESCRIPTION.
        data already transformed into a matrix with lines and pixels as the dimensions 

    Returns
    -------
    TYPE numpy array with the data dimensions

    '''
    return np.array(lines, dtype=np.uint16).reshape(-1, lines[0].shape[-1])

def downsample_kimogram(kimogram, target_lines=1000):
    global factor
    original_lines = kimogram.shape[0]
    factor = max(1, original_lines // target_lines)

    # Trim excess rows for even reshaping
    trimmed = kimogram[:(original_lines // factor) * factor]

    # Compute the average per block of lines
    downsampled = trimmed.reshape(-1, factor, kimogram.shape[1]).mean(axis=1)

    # Keep as float32 to preserve precision for colormap scaling
    return downsampled.astype(np.float32)


import czifile
import xml.etree.ElementTree as ET
def extract_metadata(file_path, callback):
    global dwell_time, pixels
    try:
        czi_file = czifile.CziFile(file_path)
        metadata = czi_file.metadata()  # Call the method to get metadata
        root = ET.fromstring(metadata)  # Parse the XML

        # Find the PixelDwellTime or LineTime
        dwell_time_elem = root.find('.//LineTime')
        if dwell_time_elem is not None:
            dwell_time = np.round(float(dwell_time_elem.text),6)
        else:
            dwell_time = None  # Handle case where LineTime is not found

        # Call the callback to proceed
        callback()

    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        dwell_time = None  # Reset dwell_time on error
        callback()  # Still call the callback to avoid blocking


def load_czi_file(file_path, callback=None):
    global dwell_time
    # Load the CZI file
    data = imread(file_path)
    
    # Extract channels, flatten if necessary
    channels = data[0, :, :, 0, 0, :, 0]  # Adjust this based on your data shape
    num_channels = data.shape[2]
    
    # Extract metadata (and ensure callback is called after extracting)
    extract_metadata(file_path, lambda: callback() if callback else None)

    return [channels[:, i, :] for i in range(num_channels)]

import queue
def ask_for_pixels_threadsafe(parent):
    q = queue.Queue()

    def ask():
        while True:
            user_input = simpledialog.askstring("Missing Metadata", "Enter number of pixels (default = 128):", parent=parent)
            if user_input is None or user_input.strip() == "":
                q.put(128)
                return
            try:
                val = int(user_input)
                if val > 0:
                    q.put(val)
                    return
                else:
                    messagebox.showerror("Invalid Input", "Please enter a positive integer.")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer.")

    parent.after(0, ask)  # Schedule ask() to run on main thread
    return q.get()  # Block this background thread until value is available


def read_B64(Archivo, tipo='line', parent_window=None):
    global dwell_time
    Read = lfd(Archivo)
    matrix = Read.asarray()

    metadata = None
    dwell_time = None
    pixels = None
    sampling_freq = None

    try:
        metadata = open(Archivo[:-8] + '.jrn', 'r').readlines()
    except FileNotFoundError:
        try:
            metadata = open(Archivo[:-4] + '.jrn', 'r').readlines()
        except FileNotFoundError:
            metadata = None

    if metadata:
        for line in metadata:
            if 'Box size' in line:
                pixels = int(line.split(':')[1].strip().split()[0])
                print('pixels: %s'%pixels)
            if 'Sampling freq' in line:
                sampling_freq_str = line.split('Sampling freq')[1].split(':')[1].strip().split()[0]
                sampling_freq = np.round(float(sampling_freq_str))
                print('sampling frequency: %s'%sampling_freq)
        dwell_time = pixels / sampling_freq
    else:
        # Thread-safe user prompt
        pixels = ask_for_pixels_threadsafe(parent_window)
    if tipo=='line':
        # Adjust matrix
        if len(matrix) % pixels != 0:
            last_line = (len(matrix) // pixels) * pixels
            matrix = matrix[:last_line]

        reshaped_matrix = matrix.reshape(-1, pixels)
        return reshaped_matrix
    elif tipo == 'image':
        pixels = pixels * 2
        frame_size = pixels * pixels
        faltantes = (-matrix.size) % frame_size

        if faltantes > 0:
            matrix = np.pad(matrix, (0, faltantes), mode='constant')

        reshaped_matrix = matrix.reshape((-1, pixels, pixels))
        return reshaped_matrix

        

def load_correlation(cmap_var):
    global G_to_save, T_to_save
    G_to_save= []
    file_paths = filedialog.askopenfilenames(filetypes=[("Correlation data", "*.txt;""*.csv")])
    if file_paths:
            file_time = filedialog.askopenfilenames(filetypes=[("Correlation time", "*.txt;""*.csv")])
    else:
        show_message('Error', "Can't correlate without the data :(")
        return
    T_to_save=np.loadtxt(file_time[0])
    if all(file_path.endswith('.txt') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(np.loadtxt(file_path, delimiter=','))
    elif all(file_path.endswith('.csv') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(pd.read_csv(file_path).to_numpy)
            
    else:
         show_message('Error',"I couldn't understand the shape of your files. Sorry")
         return
    
    num_kimograms = len(G_to_save)
    if num_kimograms>0:
        fig, axs = plt.subplots(1, num_kimograms, figsize=(8 * num_kimograms, 30), squeeze=False)
        axs = axs.flatten()  # Flatten the axis array for easier indexing

        for ax, G in zip(axs, G_to_save):
            vmin = 0
            vmax = np.max(G)
            y = np.arange(G.shape[0])
            im = ax.pcolor(y.transpose(),T_to_save, G.transpose(), shading="nearest",
                       cmap=my_colors(cmap_var), vmin=vmin, vmax=vmax)

            ax.set_xlabel('pixels', fontsize=16)  # replace with your xlabel variable
            ax.set_ylabel('Logarithmic Time (s)', fontsize=16)
            ax.set_title('pcf', fontsize=16)
            cbarformat = ticker.ScalarFormatter()
            cbarformat.set_scientific('%.2e')
            cbarformat.set_powerlimits((0, 0))
            cbarformat.set_useMathText(True)

            cbar = fig.colorbar(im, ax=ax, orientation='vertical', format='%.2f')
            cbar.ax.yaxis.get_offset_text().set_fontsize(16)
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(label='Amplitude', size=16)  # replace with your colorbar label

            ax.xaxis.tick_top()
            ax.tick_params(which='minor', length=2.25, width=1.25)
            ax.tick_params(which='major', length=3.5, width=1.75)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_yscale("log")
            ax.invert_yaxis()    
    
    canvas = FigureCanvasTkAgg(fig, master=image_frame)  # Use image_frame as the parent
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Pack the canvas
    return fig



def update_table_with_dwell_time():
    global table_frame, dwell_time, labels, entries, checkbutton_vars
    # Clear existing labels and entries in the table_frame
    for widget in table_frame.winfo_children():
        widget.destroy()

    # Initialize or reset labels, entries, and checkbutton_vars
    labels = ["Line Time (ms)", "First Line", "Last Line", "Distance (px)", "H Smoothing (px)", "V Smoothing (lines)", "Reverse", 'Normalize']
    entries = []  # Reset entries
    checkbutton_vars = {}  # Reset Checkbutton vars

    # Recreate the table with updated values
    for row, label in enumerate(labels):
        tk.Label(table_frame, text=label, borderwidth=2, relief="solid").grid(row=row, column=0, padx=5, pady=5, sticky="e")

        if label == 'Reverse' or label=='Normalize':
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(table_frame, variable=var)
            entries.append(checkbox)  # Append checkbox, not its variable
            checkbutton_vars[label] = var
            checkbox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        else:
            entry = tk.Entry(table_frame)
            if label == 'Line Time (ms)' and dwell_time is not None:
                entry.insert(0, f'{dwell_time * 1000}')
            elif label == 'H Smoothing (px)':
                entry.insert(0, '4')
            elif label == 'V Smoothing (lines)':
                entry.insert(0, '10')
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")


def get_table_data():
    '''
    This function collects the data the user puts into the pCF parameters table.

    Returns
    -------
    data : dict
    '''
    data = {}
    for label, widget in zip(labels, entries):
        if isinstance(widget, tk.Checkbutton):
            # Retrieve the BooleanVar associated with the Checkbutton
            var = checkbutton_vars.get(label)
            if var is not None:
                data[label] = var.get()
            else:
                data[label] = None
        elif isinstance(widget, tk.Entry):
            # Retrieve the value from Entry widgets
            value = widget.get().strip()  # Strip any whitespace
            if value:  # Only convert if the entry is not empty
                try:
                    data[label] = float(value)
                except ValueError:
                    data[label] = value  # Keep it as a string if conversion fails
            else:
                data[label] = ""  # Handle empty entry case
    
    print("Collected data:", data)  # Debug print
    return data


def display_kimograms(kimograms,cmap_var,factor=1):
    global fig, canvas, image_frame
    fig = plot_kimograms(kimograms,cmap_var, factor)
    fig.colorbars = []  # Initialize here to avoid AttributeError later
    plt.close('all')    
    if canvas is not None:
        canvas.get_tk_widget().destroy()
        plt.close('all')
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)



def update_kimogram_label():
    # Update the label to show the current kimogram
    kimogram_label.config(text=f"Current Kimogram: {current_index+1}")
    kimogram_label.grid(row=1, column=6, pady=(5, 0))  # Add some padding above the label
# Create a button to toggle between kimograms

# Toggle between kimograms
def toggle_kimogram():
    '''
    This function is only relevant if there are two kimograms.
    It is used to change over which kimogram the buttons such as zoom or h-lines will be used.
    '''
    global current_index
    if current_index == 0:
        current_index = 1
    else:
        current_index = 0
    set_current_axis(current_index)
    update_kimogram_label()  # Update the label text
    canvas.draw()


def set_current_axis(index):
    '''
    This function is used to select over which kimogram the action will be done.
    '''
    global current_ax
    if 0 <= index < len(fig.axes):
        current_ax = fig.axes[index]
        print(f"Current axis set to: Kimogram {index + 1}")
    else:
        print("Invalid axis index")


def on_axis_select(index):
    # Sample button to switch axis (replace with your own logic)
    set_current_axis(index)


def open_hlines_window(parent_window):
    '''
    this function is used to plot the horizontal profile of the kimogram, i.e. averaged lines.
    This is particularly helpful to identify regions of different intensity. 
    The plot is done in a new window where the user selects wich lines to average. 
    If there are 2 kimograms, use the tgogle function to change which one you want for this window.

    Returns
    -------
    None.

    '''
    hlines_window = Toplevel(parent_window)
    hlines_window.title("Select lines range to plot")

    def plot_lines():
        global original_kimograms, current_index
        try:
            original_kimogram=original_kimograms[current_index]
        except:
            print('no current axis')
            original_kimogram=original_kimograms[0]
        start = start_var.get()
        num = num_var.get()

        if start < 0 or start >= original_kimogram.shape[0]:
            print("Error: Start line out of range.")
            return

        if num <= 0:
            print("Error: Number of lines to average must be positive.")
            return

        end = start + num
        if end > original_kimogram.shape[0]:
            show_message("Error", "Error: End index exceeds data range.")
            print("Error: End index exceeds data range.")
            return

        # Calculate the average profile over the specified range
        lines = original_kimogram[start:end,:]
        average_lines = np.mean(lines, axis=0)

        # Create a new figure with adjusted size
        profile_fig = Figure(figsize=(8, 6), dpi=100)
        profile_ax = profile_fig.add_subplot(111)
        profile_ax.plot([i for i in range(len(original_kimogram[0]))], average_lines, color='black')
        
        # Set labels with larger font size
        profile_ax.set_title(f'Average intensity from line {start} to {end}', fontsize=16)
        profile_ax.set_xlabel('Pixel', fontsize=14)
        profile_ax.set_ylabel('Intensity', fontsize=14)
        
        # Increase padding and font size for ticks
        profile_ax.tick_params(axis='both', which='major', labelsize=10)
        profile_ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # Add grid for better readability
        profile_ax.grid(True)
        
        # Create and add the canvas
        profile_canvas = FigureCanvasTkAgg(profile_fig, master=hlines_window)
        profile_canvas.draw()
        
        # Use grid() instead of pack()
        profile_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky='nsew')
        def on_click(event):
            if event.inaxes is not None:
                    x_data = event.xdata
                    y_data = event.ydata
                    print(f"Clicked at x={x_data:.2f}, y={y_data:.4f}")
                    # Update or create a label to show these coordinates
                    coord_label.config(text=f"Coordinates: x={x_data:.2f}s, y={y_data:.4f}")
        # Connect the event handler
        profile_canvas.mpl_connect('button_press_event', on_click)
        
        # Create a label to display the coordinates
        coord_label = Label(hlines_window, text="Coordinates: x= , y=")
        coord_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    # Add entries for starting pixel and number of pixels
    start_var = IntVar(value=0)
    num_var = IntVar(value=1)
    
    start_label = Label(hlines_window, text="Start line:")
    start_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    
    start_entry = Entry(hlines_window, textvariable=start_var)
    start_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    
    num_label = Label(hlines_window, text="Number of lines to Average:")
    num_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    
    num_entry = Entry(hlines_window, textvariable=num_var)
    num_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
    
    plot_button = tk.Button(hlines_window, text="Plot H-lines", command=plot_lines)
    plot_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    hlines_window.grid_columnconfigure(1, weight=1)
    hlines_window.grid_rowconfigure(3, weight=1)  # Allow row 3 to expand if needed

def plot_on_ax(ax, G_log, t_log,title,cmap):
    '''
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    G_log : TYPE matrix
        DESCRIPTION. correlation matrix computed from pcf
    t_log : TYPE list
        DESCRIPTION. correlation delay times
    title : TYPE str
        DESCRIPTION. title asociated to the plot
    Returns
    -------
    None.

    '''
    vmin = 0
    vmax = np.max(G_log)
    y = np.arange(G_log.shape[0])
    im = ax.pcolor(y.transpose(), t_log, G_log.transpose(), shading="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel('pixels', fontsize=12)
    ax.set_ylabel('Logarithmic Time (s)', fontsize=14)
    ax.set_title(title, fontsize=16)
    cbarformat = ticker.ScalarFormatter()
    cbarformat.set_scientific('%.2e')
    cbarformat.set_powerlimits((0, 0))
    cbarformat.set_useMathText(True)
    cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', format='%.2f')
    #cbar = fig.colorbar(im, ax=ax, orientation='vertical', format='%.2f')
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label='Correlation amplitude', size=16)  # replace with your colorbar label

    ax.xaxis.tick_top()
    ax.tick_params(which='minor', length=2.25, width=1.25)
    ax.tick_params(which='major', length=3.5, width=1.75)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_yscale("log")
    ax.invert_yaxis()
    return ax


# Initialize the root window
def init_app(root):
    root.title("Correlation Analysis Software based on Python")
    root.geometry("500x750")
    root.configure(bg="white")

    # Set window icon
    image = Image.open(resource_path("logo.gif"))
    photo = ImageTk.PhotoImage(image)
    root.photo = photo
    root.iconphoto(False, photo)

    # Load animated GIF
    gif = Image.open(resource_path("logo.gif"))
    frames = [ImageTk.PhotoImage(frame.copy().convert('RGBA')) for frame in ImageSequence.Iterator(gif)]
    frame_index = 0
    logo_label = tk.Label(root, bg="white")
    logo_label.grid(row=0, column=0, pady=20)  # Changed from pack() to grid()
    animar_gif(root, logo_label, frames, frame_index)

    # Create buttons for each window
    # The CASPY window button for pCF
    btn1 = tk.Button(root, text="pCF", command=abrir_pcf_ventana_main_window, bg="white", height=2, width=20)
    btn1.grid(row=1, column=0, pady=5)  # This button now opens the pCF window
    #btn2 = tk.Button(root, text="2D-pCF", command=lambda: nueva_ventana(root, "CASPY-2DpCF", "Work In Progress"), bg="white", height=2, width=20)
    btn2 = tk.Button(root, text="2D-pCF", command=abrir_2d_pcf_ventana, bg="white", height=2, width=20)    
    btn2.grid(row=2, column=0, pady=5)  # Changed from pack() to grid()
    btn3 = tk.Button(root, text="N&B", command=abrir_NB_ventana, bg="white", height=2, width=20)
    btn3.grid(row=3, column=0, pady=5)  # Changed from pack() to grid()

# Animate the GIF
def animar_gif(root, logo_label, frames, frame_index):
    frame = frames[frame_index]
    logo_label.configure(image=frame)
    frame_index = (frame_index + 1) % len(frames)
    root.after(50, animar_gif, root, logo_label, frames, frame_index)



# Create a new window with a message
def nueva_ventana(root, titulo, mensaje):
    ventana = Toplevel(root)
    ventana.title(titulo)
    ventana.geometry("300x200")
    ventana.configure(bg="white")
    tk.Label(ventana, text=mensaje, font=("Arial", 18), bg="white").grid(pady=40)


def load_and_display(file_paths, pcf_window,cmap_var):  # Added pcf_window parameter
    loading_window = show_loading_popup(pcf_window)  # ⬅️ Show loading popup in the pCF window

    def run():
        global original_kimograms
        original_kimograms = []
        errors = []

        try:
            if len(file_paths) == 1 and file_paths[0].endswith('.czi'):
                def after_metadata():
                    try:
                        lines = load_czi_file(file_paths[0])
                        original_kimograms.extend(lines)

                        optimized_kimograms = [downsample_kimogram(k) for k in original_kimograms]
                        global factor
                        pcf_window.after(0, lambda: display_kimograms(optimized_kimograms,cmap_var,factor))
                        pcf_window.after(0, update_table_with_dwell_time)
                        pcf_window.after(0, loading_window.destroy)
                    except Exception as e:
                        pcf_window.after(0, loading_window.destroy)
                        pcf_window.after(0, lambda: show_message("Error", str(e)))

                extract_metadata(file_paths[0], after_metadata)
                return

            elif all(fp.endswith(('.tif', '.tiff')) for fp in file_paths):
                for fp in file_paths:
                    try:
                        kim = create_kimogram([tif_imread(fp)])
                        original_kimograms.append(kim)
                    except Exception as e:
                        errors.append(f"{fp}: {e}")

            else:
                for fp in file_paths:
                    try:
                        kim = read_B64(fp,parent_window=pcf_window)
                        print("B64 loaded:", type(kim))
                        original_kimograms.append(kim)
                        pcf_window.after(0, update_table_with_dwell_time)
                    except Exception as e:
                        errors.append(f"{fp}: {e}")

            optimized_kimograms = [downsample_kimogram(k) for k in original_kimograms]

            if errors:
                pcf_window.after(0, lambda: show_message("Error", "\n".join(errors)))
            global factor
            pcf_window.after(0, lambda: display_kimograms(optimized_kimograms,cmap_var,factor))
            pcf_window.after(0, loading_window.destroy)

        except Exception as e:
            import traceback
            error_msg = f"Unhandled error:\n{str(e)}\n\n{traceback.format_exc()}"
            pcf_window.after(0, loading_window.destroy)
            pcf_window.after(0, lambda: show_message("Error", error_msg))

    threading.Thread(target=run, daemon=True).start()


def load_lines(pcf_window,cmap_var):  # Added pcf_window parameter
    file_paths = filedialog.askopenfilenames(filetypes=[("line files", "*.tiff;*.tif;*.czi;*.b64;*.raw")])
    if file_paths:
        threading.Thread(target=load_and_display, args=(file_paths, pcf_window,cmap_var), daemon=True).start()  # Pass pcf_window to load_and_display
    else:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")



def get_pixels():
    # Create a dialog window
    pixels = simpledialog.askinteger("Input", "Please enter the value for pixels:")
    if pixels is None:  # Handle cancellation
        raise ValueError("Pixel input was canceled.")
    return pixels


def plot_kimograms(kimograms,cmap_var,factor=1):
    global current_figures
    current_figures = []

    def get_extent(kimogram):
        return [0, kimogram.shape[1], kimogram.shape[0], 0]  # [x_min, x_max, y_min, y_max]

    num_kimograms = len(kimograms)

    if num_kimograms > 1:
        fig, axs = plt.subplots(1, num_kimograms, figsize=(8 * num_kimograms, 20), squeeze=False)
        axs = axs.flatten()
        
        for ax, kimogram in zip(axs, kimograms):
            vmin = np.min(kimogram)
            vmax = np.mean(kimogram) + 2 * np.std(kimogram)
            extent = get_extent(kimogram)
            cax = ax.imshow(kimogram, aspect='auto', cmap=my_colors(cmap_var), norm=plt.Normalize(vmin=vmin, vmax=vmax),
        origin='upper', extent=[0, kimogram.shape[1], kimogram.shape[0], 0])
            ax.set_xlabel("Pixel")
            ticks = np.linspace(0,len(kimogram),5, dtype=int)
            ax.set_yticks(ticks,labels=[int(ticks[i]*factor/10000)*10000 for i in range(len(ticks))])
            ax.set_ylabel("Line Number")
            ax.set_title(f"Kimogram {axs.tolist().index(ax) + 1}")
            plt.colorbar(cax, ax=ax, label='Intensity')
            plt.subplots_adjust(wspace=0.5)

    else:
        kimograma = kimograms[0]
        extent = get_extent(kimograma)
        fig, ax = plt.subplots(figsize=(8, 10))
        cax = ax.imshow(
            kimograma,
            aspect='auto',
            cmap=my_colors(cmap_var),
            norm=plt.Normalize(vmin=0, vmax=np.mean(kimograma) + 2 * np.std(kimograma)),
            origin='upper',
            extent=extent
        )
        ax.set_xlabel("Pixel")
        ticks = np.linspace(0,len(kimograma),5, dtype=int)
        print(ticks)
        print(len(kimograma))
        ax.set_yticks(ticks,labels=[int(ticks[i]*factor/10000)*10000 for i in range(len(ticks))])
        ax.set_ylabel("Line Number")
        ax.set_title("Kimogram")
        plt.colorbar(cax, ax=ax, label='Intensity')

    current_figures.append(fig)
    return fig

def moving_average(a, n=3,Especial=False) :
    if Especial==False:
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    if Especial==True:
        espacios= int(np.floor(np.log10(len(a))))
        ret= np.nancumsum(a, dtype=float)
        ret0=ret
        ret0[n:] = ret0[n:] - ret0[:-n]
        ret0=ret0[n - 1:11] / n
        for i in range(espacios):
            ret1=np.nancumsum(a, dtype=float)
            n0=10**(i+1)+1
            n1=n0+10**(i+2)
            if n1>len(a):
                n1=len(a)
            ret1[n0:]= ret1[n0:] - ret1[:-n0]
            # ret1=ret1[n0 + int((10**(i+1)+10**(i))/2)-1:n1-1] / n0
            ret1=ret1[n0+int(n0/2)-1:n1-1] / n0
            ret0=np.append(ret0,ret1)
        return ret0
    else:
        print('Especifique el modo Especial como True o False')
        
def line_pCF_analysis(Kimogram, C0, C1, linetime, reverse_PCF, 
                      return_time=0, Tiempo_imagen=0, logtime=False, Movil_log=0):

    '''
    Parameters
    ----------
    Kimogram of intensity: ndarray
        Tipically 100k rows and 256 columns.
    
    C0 : int
        First column to be correlated.
    
    C1 : int
        Second column to be correlated.

    tp: float
        pixel dwell time.
        
    return_time : float, optional
        Line time return. The default is 0.
        If pixel dwell time already includes the return pixel time of the microscope,
        then the return_time parameter can be ignored. 
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : 1D-array
        Correlation between columns C0 and C1
        
    Tau : 1D-array
        Correlation time
    
    '''
    
    # First column to be correlated
    C0=Kimogram [ : ,C0]
    # Second column to be correlated
    C1=Kimogram [ : ,C1]    
    
    if reverse_PCF:
        C0, C1 = C1, C0

    ######################################
    
    ## based on https://www.cgohlke.com/ipcf/ --> linear correlation
    
    """Return linear correlation of two arrays using DFT."""
    size = C0.size
    
    # subtract mean and pad with zeros to twice the size
    C0_mean = C0.mean()
    C1_mean = C1.mean()
    C0 = np.pad(C0-C0_mean, C0.size//2, mode='constant')
    C1 = np.pad(C1-C1_mean, C1.size//2, mode='constant')

    # forward DFT
    C0 = np.fft.rfft(C0)
    C1 = np.fft.rfft(C1)
    # multiply by complex conjugate
    G = C0.conj() * C1
    # reverse DFT
    G = np.fft.irfft(G)
    # positive delays only
    G = G[:size // 2]
        
    # normalize with the averages of a and b
    G /= size * C0_mean * C1_mean
    
    ######################################
    
    Tau=[]
    
    for i in range (1,Kimogram.shape[0]+1):
                     Tau.append(i*linetime)
    


    #Apply MAV
    if Movil_log==0:

        if logtime==False:
            return np.array(G), np.array (Tau)
        if logtime==True:
            Tau=np.log10 (Tau)
            return np.array(G), np.array(Tau)
    else:

        G=moving_average(G, Movil_log , Especial=True)
    
        if logtime==False:
            Tau=moving_average(Tau, Movil_log, Especial=True )
        if logtime==True:
            Tau=np.log10(moving_average(Tau, Movil_log, Especial=True ))
    
    return np.array(G), np.array(Tau)

def pCF(Kimogram , linetime, dr=0, reverse_PCF=False, return_time=0, logtime=False, Movil_log=0):
    '''
    Parameters
    ----------
    Kimogram : ndarray tipically (100k rows, 256 columns)
        Kimogram = Kimogram or B Kimogram  
        
    dr : int
        pCF distance.
    
    tp : TYPE
        DESCRIPTION.

    return_time : float, optional
        line time return. The default is 0.
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.
    
    ACF norm: Bool, optional
        Normalizes the amplitud of correlation by the G(0)

    Returns
    -------
    G : ndarray 
        Matrix of pCF analisys between at distance.
        -) Columns are the pixel position
        -) Rows are the correlation analysis
        
    Tau : ndarray
          Correlation time
          -) Columns are the pixel position
          -) Rows are the delay time
    '''
    

    Size = len(Kimogram[0]) #cantidad de píxeles en una linea

    #calculo todas las correlaciones
    G=[]
    T=[]
    for i in tqdm.trange(Size-dr):
        result = line_pCF_analysis(Kimogram ,i ,i+dr, linetime, reverse_PCF,
                                         return_time, logtime, Movil_log=Movil_log)
        
        G.append(result[0])
        T.append(result[1])

    return np.array(G).transpose(), np.array(T).transpose()


def line_crosspCF_analysis(Kimogram, Kimogram2, C0, C1, linetime, reverse_PCF, 
                      return_time=0, Tiempo_imagen=0, logtime=False, Movil_log=0):

    '''
    Parameters
    ----------
    Kimogram of intensity: ndarray
        Tipically 100k rows and 256 columns.
    
    C0 : int
        First column to be correlated.
    
    C1 : int
        Second column to be correlated.

    linetime: float
        time taken to adquire each line.
        
    return_time : float, optional
        Line time return. The default is 0.
        If pixel dwell time already includes the return pixel time of the microscope,
        then the return_time parameter can be ignored. 
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : 1D-array
        Correlation between columns C0 and C1
        
    Tau : 1D-array
        Correlation time
    
    '''
    
    # First column to be correlated
    C0=Kimogram [ : ,C0]
    # Second column to be correlated
    C1=Kimogram2 [ : ,C1]    
    
    if reverse_PCF:
        C0, C1 = C1, C0

    ######################################
    
    ## based on https://www.cgohlke.com/ipcf/ --> linear correlation
    
    """Return linear correlation of two arrays using DFT."""
    size = C0.size
    
    # subtract mean and pad with zeros to twice the size
    C0_mean = C0.mean()
    C1_mean = C1.mean()
    C0 = np.pad(C0-C0_mean, C0.size//2, mode='constant')
    C1 = np.pad(C1-C1_mean, C1.size//2, mode='constant')

    # forward DFT
    C0 = np.fft.rfft(C0)
    C1 = np.fft.rfft(C1)
    # multiply by complex conjugate
    G = C0.conj() * C1
    # reverse DFT
    G = np.fft.irfft(G)
    # positive delays only
    G = G[:size // 2]
        
    # normalize with the averages of a and b
    G /= size * C0_mean * C1_mean
    
    ######################################
    Tau=[]
    
    for i in range (1,Kimogram.shape[0]+1):
                     Tau.append(i*linetime)
    #Tau = Tau[:len(Tau)//2]


    #Apply MAV
    if Movil_log==0:

        if logtime==False:
            return np.array(G), np.array (Tau)
        if logtime==True:
            Tau=np.log10 (Tau)
            return np.array(G), np.array(Tau)
    else:

        G=moving_average(G, Movil_log , Especial=True)
    
        if logtime==False:
            Tau=moving_average(Tau, Movil_log, Especial=True )
        if logtime==True:
            Tau=np.log10(moving_average(Tau, Movil_log, Especial=True ))
    
    return np.array(G), np.array(Tau)


def crosspCF(Kimogram, Kimogram2, linetime, dr=0, reverse_PCF=False, return_time=0, logtime=False, Movil_log=0):
    '''
    Parameters
    ----------
    Kimogram : ndarray tipically (100k rows, 256 columns)
        Kimogram = Kimogram or B Kimogram  
        
    dr : int
        pCF distance.
    
    tp : TYPE
        DESCRIPTION.

    return_time : float, optional
        line time return. The default is 0.
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : ndarray 
        Matrix of pCF analisys between at distance.
        -) Columns are the pixel position
        -) Rows are the correlation analysis
        
    Tau : ndarray
          Correlation time
          -) Columns are the pixel position
          -) Rows are the delay time
    '''
    
        
    # if Movil_log==0:
        # print('MAV = 0')
    # else:
        # print('MAV = %s' % (Movil_log))
    Size = len(Kimogram[0]) #cantidad de píxeles en una linea
    print(f"Starting crosspCF with Size: {Size}, dr: {dr}, line time: {linetime}")  # Debug print

    #calculo todas las correlaciones
    G=[]
    T=[]
    for i in tqdm.trange(Size-dr):
        #print(f"Processing index: {i}")  # Debug print
        pCF_analysis = line_crosspCF_analysis(Kimogram ,Kimogram2, i ,i+dr, linetime, reverse_PCF,
                                         return_time, logtime, Movil_log=Movil_log)

        G.append(pCF_analysis[0])
        T.append(pCF_analysis[1])

    return np.array(G).transpose(), np.array(T).transpose()

def apply_ccpCF():
    '''
    this function applys the cross-pair correlation to the uploaded files. There should be two kimograms for this function to work.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    global G_to_save, T_to_save, table_frame, original_kimograms
    if original_kimograms is None:
        show_message('Error',"No kimograms data to process.")
        pass
    
    # Get the parameters from the table and apply the pCF function
    data = get_table_data()
    #data = get_values()
    try:
        first_line = int(data.get("First Line", ""))
    except:
        show_message('Error', "You need to especify the parameters in the table below")
    last_line = int(data.get("Last Line", ""))
    line_time = data.get("Line Time (ms)", "")
    print(data)
    dr = int(data.get("Distance (px)", ""))
    reverse = data.get('Reverse')
    sigma = [
    int(data.get("H Smoothing (px)", 4)),  # Default to 0 if empty
    int(data.get("V Smoothing (lines)", 10))  # Default to 0 if empty
]
    acfnorm = data.get('Normalize')
    G, T = crosspCF(original_kimograms[0][first_line:last_line], original_kimograms[1][first_line:last_line], linetime=line_time/1000, dr=dr, reverse_PCF=reverse)
    G2, T2 = crosspCF(original_kimograms[1][first_line:last_line], original_kimograms[0][first_line:last_line], linetime=line_time/1000, dr=dr, reverse_PCF=reverse)
    x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint = False)    
    t_lineal = T[:,0]
    t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)
    G_basura = []
    for i in x1:
        G_basura.append(list(G[i]))
    G = np.asarray(G_basura).transpose()
    t = []
    for i in x1:
        t.append(t_lineal[i])
    t_lineal = np.asarray(t)
    G_log = np.empty_like(G)
    for i, gi in enumerate(G):
        G_log[i] = np.interp(t_log, t_lineal, gi)
    G_log = gaussian_filter(G_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    G_to_save = G_log  # Store G_log for saving
    T_to_save = t_log
    if acfnorm:
        print('Normalizing correlation by ACF...')
        A, _ = pCF(original_kimograms[0][first_line:last_line], line_time/1000, dr=0,reverse_PCF=reverse)
        x1 = np.geomspace(1, len(A), 256, dtype=int, endpoint = False)    
        A_basura = []
        for i in x1:
            A_basura.append(list(A[i]))
        A = np.asarray(A_basura).transpose()
        A_log = np.empty_like(A)
        for i, gi in enumerate(A):
            A_log[i] = np.interp(t_log, t_lineal, gi)
        A_log = gaussian_filter(A_log, sigma = sigma) 
        #maxim = A_log[:,0]
        maxim = np.max(A_log, axis=1)
        G_to_save = [G_to_save[i,j]/maxim[i] for i in range(len(G_to_save)) for j in range(len(G_to_save[0]))]
        for i in range(len(G_to_save)):
            if G_to_save[i]>2:
                G_to_save[i]=2
                
        G_to_save = np.reshape(G_to_save, np.shape(G_log))
    x1 = np.geomspace(1, len(G2), 256, dtype=int, endpoint = False)    
    G2_basura = [list(G2[i]) for i in x1]
    G2 = np.asarray(G2_basura).transpose()
    G2_log = np.empty_like(G2)
    for i, gi in enumerate(G2):
        G2_log[i] = np.interp(t_log, t_lineal, gi)
    G2_log = gaussian_filter(G2_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    
    G2_to_save = G2_log  # Store G_log for saving
    if acfnorm:
        print('Normalizing correlation by ACF...')
        A, _ = pCF(original_kimograms[1][first_line:last_line], line_time/1000, dr=0,reverse_PCF=reverse)
        x1 = np.geomspace(1, len(A), 256, dtype=int, endpoint = False)    
        A_basura = []
        for i in x1:
            A_basura.append(list(A[i]))
        A = np.asarray(A_basura).transpose()
        A_log = np.empty_like(A)
        for i, gi in enumerate(A):
            A_log[i] = np.interp(t_log, t_lineal, gi)
        A_log = gaussian_filter(A_log, sigma = sigma) 
        maxim = np.max(A_log, axis=1)
        G2_to_save = [G2_to_save[i,j]/maxim[i] for i in range(len(G2_to_save)) for j in range(len(G2_to_save[0]))]
        for i in range(len(G2_to_save)):
            if G2_to_save[i]>2:
                G2_to_save[i]=2
                
        G2_to_save = np.reshape(G2_to_save, np.shape(G2_log))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)
 # Plot Channel 1 on the first axis
    cmap_value = cmap_var.get() if hasattr(cmap_var, "get") else cmap_var
    cmap = resolve_cmap(cmap_value)
    plot_on_ax(ax1, G_log,t_log, 'Ch1 to Ch2', cmap=cmap)
 
    try: 
        # Plot Channel 2 on the second axis
        cmap_value = cmap_var.get() if hasattr(cmap_var, "get") else cmap_var
        cmap = resolve_cmap(cmap_value)
        plot_on_ax(ax2, G2_log,t_log, 'Ch2 to Ch1', cmap=cmap)
        G_to_save = [G_log,G2_log]  # Store G_log for saving
    except:
        pass
    fig.tight_layout(pad=2.0)
    return fig
    
from matplotlib.colors import LinearSegmentedColormap, Colormap

def resolve_cmap(cmap_input):
    if isinstance(cmap_input, Colormap):
        return cmap_input
    elif isinstance(cmap_input, str):
        return plt.get_cmap(cmap_input)
    elif isinstance(cmap_input, dict):
        return LinearSegmentedColormap('custom', cmap_input)
    else:
        raise ValueError("Unsupported cmap type")
    
def apply_pCF():
    '''
    This function computes the correlation (using the pCF function) for the given kimograms with the provided
    parameters. 

    Returns
    -------
    fig : matplotlib.figure.Figure
    '''
    global G_to_save, T_to_save, table_frame

    if original_kimograms is None:
        show_message('Error', "No kimograms data to process.")
        return

    # Get parameters from the table
    data = get_table_data()
    try:
        first_line = int(data.get("First Line", ""))
        last_line = int(data.get("Last Line", ""))
        line_time = float(data.get("Line Time (ms)", "")) / 1000  # convert to seconds
        dr = int(data.get("Distance (px)", ""))
    except Exception as e:
        show_message('Error', f"You need to specify all required parameters. {str(e)}")
        return

    reverse = data.get('Reverse')
    acfnorm = data.get('Normalize')
    sigma = [
        int(data.get("H Smoothing (px)", 4)),
        int(data.get("V Smoothing (lines)", 10))
    ]
    print(f"Valor H Smoothing: {data.get('H Smoothing (px)', 4)}")
    print(f"Valor V Smoothing: {data.get('V Smoothing (lines)', 10)}")
    # ---- CHANNEL 1 ----
    G1, T1 = pCF(original_kimograms[0][first_line:last_line], line_time, dr=dr, reverse_PCF=reverse)
    x1_ch1 = np.geomspace(1, len(G1), 256, dtype=int, endpoint=False)
    t_lineal_ch1 = T1[:, 0]
    t_log_ch1 = np.geomspace(t_lineal_ch1[0], t_lineal_ch1[-1], 256, endpoint=True)

    G1_downsampled = np.asarray([G1[i] for i in x1_ch1]).T
    t_lineal_ch1_resampled = np.asarray([t_lineal_ch1[i] for i in x1_ch1])

    G1_log = np.empty_like(G1_downsampled)
    for i, gi in enumerate(G1_downsampled):
        G1_log[i] = np.interp(t_log_ch1, t_lineal_ch1_resampled, gi)

    G1_log = gaussian_filter(G1_log, sigma=sigma)
    G_to_save = G1_log
    T_to_save = t_log_ch1

    if acfnorm:
        print('Normalizing Channel 1 by ACF...')
        A1, _ = pCF(original_kimograms[0][first_line:last_line], line_time, dr=0, reverse_PCF=reverse)
        A1_downsampled = np.asarray([A1[i] for i in x1_ch1]).T
        A1_log = np.empty_like(A1_downsampled)
        for i, gi in enumerate(A1_downsampled):
            A1_log[i] = np.interp(t_log_ch1, t_lineal_ch1_resampled, gi)
        A1_log = gaussian_filter(A1_log, sigma=sigma)
        A1_max = np.max(A1_log, axis=1)
        G1_log = np.clip(np.reshape([
            G1_log[i, j] / A1_max[i] for i in range(G1_log.shape[0]) for j in range(G1_log.shape[1])
        ], G1_log.shape), None, 2)
        G_to_save = G1_log

    # ---- CHANNEL 2 ----
    if len(original_kimograms) > 1:
        G2, T2 = pCF(original_kimograms[1][first_line:last_line], line_time, dr=dr, reverse_PCF=reverse)
        x1_ch2 = np.geomspace(1, len(G2), 256, dtype=int, endpoint=False)
        t_lineal_ch2 = T2[:, 0]
        t_log_ch2 = np.geomspace(t_lineal_ch2[0], t_lineal_ch2[-1], 256, endpoint=True)

        G2_downsampled = np.asarray([G2[i] for i in x1_ch2]).T
        t_lineal_ch2_resampled = np.asarray([t_lineal_ch2[i] for i in x1_ch2])

        G2_log = np.empty_like(G2_downsampled)
        for i, gi in enumerate(G2_downsampled):
            G2_log[i] = np.interp(t_log_ch2, t_lineal_ch2_resampled, gi)

        G2_log = gaussian_filter(G2_log, sigma=sigma)
        G2_to_save = G2_log

        if acfnorm:
            print('Normalizing Channel 2 by ACF...')
            A2, _ = pCF(original_kimograms[1][first_line:last_line], line_time, dr=0, reverse_PCF=reverse)
            A2_downsampled = np.asarray([A2[i] for i in x1_ch2]).T
            A2_log = np.empty_like(A2_downsampled)
            for i, gi in enumerate(A2_downsampled):
                A2_log[i] = np.interp(t_log_ch2, t_lineal_ch2_resampled, gi)
            A2_log = gaussian_filter(A2_log, sigma=sigma)
            A2_max = np.max(A2_log, axis=1)
            G2_log = np.clip(np.reshape([
                G2_log[i, j] / A2_max[i] for i in range(G2_log.shape[0]) for j in range(G2_log.shape[1])
            ], G2_log.shape), None, 2)
            G2_to_save = G2_log

        # --- PLOT BOTH CHANNELS ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)
        cmap_value = cmap_var.get() if hasattr(cmap_var, "get") else cmap_var
        cmap = resolve_cmap(cmap_value)
        plot_on_ax(ax1, G1_log, t_log_ch1, 'Channel 1', cmap=cmap)
        plot_on_ax(ax2, G2_log, t_log_ch2, 'Channel 2', cmap=cmap)
        G_to_save = [G1_log, G2_log]
        T_to_save = [t_log_ch1, t_log_ch2]

    else:
        # Only one channel
        fig, ax = plt.subplots(figsize=(6, 8))
        cmap_value = cmap_var.get() if hasattr(cmap_var, "get") else cmap_var
        cmap = resolve_cmap(cmap_value)
        plot_on_ax(ax, G1_log, t_log_ch1, f'pCF({dr})', cmap=cmap)

    fig.tight_layout(pad=2.0)
    return fig


from scipy.optimize import curve_fit
def one_component_acf(t, w0, wz, td, N):
    t = np.asarray(t, dtype=float) 
    V = 1.0
    G = (1.0 / (V * N)) / (1.0 + t/td) / np.sqrt(1.0 + (w0/wz)**2 * t/td)
    return G

# Fit function
def fit_acf(time, G, w0, wz, dim=3,components=1):
    if components == 1:
        # Initial guess for td and N
        initial_guess = [1e-2, 5.0]  # td, N

        # Model with w0 and wz fixed
        def model(t, td, N):
            t = np.asarray(t, dtype=float).ravel()
            result = one_component_acf(t, w0, wz, td, N)
            return np.asarray(result, dtype=float).ravel() 

        try:
            popt, pcov = curve_fit(model, time, G, p0=initial_guess, maxfev=10000)
        except Exception as e:
            print("curve_fit failed:", e)
            raise

        td_fit, N_fit = popt
        D = w0**2 / (dim*2 * td_fit)
        G_fit = model(time, *popt)

        # Plotting
        plt.figure()
        plt.plot(time, G, 'bo', label='Data')
        plt.plot(time, G_fit, 'r-', label='Fitted Curve')
        plt.xscale('log')
        plt.xlabel('Time')
        plt.ylabel('G')
        plt.legend()
        plt.title('ACF Fit (1-Component Model)')
        plt.grid(True)
        plt.show()

        print(f"Fitted td = {td_fit:.3e} s")
        print(f"Fitted N = {N_fit:.3f}")
        print(f"Estimated D = {D:.3e} μm²/s")
        return G_fit, td_fit, N_fit, D

    else:
        raise NotImplementedError("Only one-component model is implemented.")

def save_data():
    '''
    Function to save the correlation data as a csv file in a choosen file.
    If there are multiple plots it will create multiple files.
    Returns
    -------
    None.

    '''
    global G_to_save
    if G_to_save is not None:
        if isinstance(G_to_save, list):
            for i, matrix in enumerate(G_to_save):
                file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                           filetypes=[("CSV files", "*.csv"),("txt files", '*.txt') ,("All files", "*.*")],
                                                           title=f"Save G{i+1} Matrix")
                if file_path:
                    np.savetxt(file_path, matrix,  delimiter=',', fmt='%.6f', comments='')
                    show_message('Good', f"Data saved to {file_path}")
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                       filetypes=[("CSV files", "*.csv"),("txt files", '*.txt') , ("All files", "*.*")])
            if file_path:
                np.savetxt(file_path, G_to_save, delimiter=',', comments='')
                show_message('Good', f"Data saved to {file_path}")
    else:
        show_message('Error',"No data to save")


def save_time():
    '''
    function to save the correlation time

    Returns
    ------- saved file
    None.
    '''
    global T_to_save
    if T_to_save is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                   filetypes=[("txt files", "*.txt"), ("All files", "*.*")])
        if file_path:
            np.savetxt(file_path, T_to_save)
            show_message('Done!', f"Correlation time saved to {file_path}")
    else:
         show_message('Error',"No data to save")
            


def save_plot(fig,figsize=None):
    '''
    function to save as a png the plot.
    Parameters
    ----------
    fig : TYPE figure
        DESCRIPTION. figure to be saved

    Returns
    -------
    None.
    '''
    # Open a file dialog to choose the save location and filename
    file_path = filedialog.asksaveasfilename(
    defaultextension=".png",
    filetypes=[("PNG files", "*.png"), ("SVG files", "*.svg")],
    title="Save Plot As"
    )

    if file_path:
        # Save the plot
        if figsize:
            fig.savefig(file_path)
        else:
            fig.savefig(file_path)
        print(f"Plot saved as {file_path}")


def open_profiles_window(parent_window):
    """
    Plot vertical profiles in kimograms (temporal evolution for one pixel
    or the average over N pixels).
    """

    global G_to_save, T_to_save

    # --- helpers -------------------------------------------------------------
    def _get_channel_count():
        return len(G_to_save) if isinstance(G_to_save, (list, tuple)) else 1

    def _get_G(ch_idx):
        """Return the (pixels x time) array for the chosen channel."""
        if isinstance(G_to_save, (list, tuple)):
            return np.asarray(G_to_save[ch_idx])
        return np.asarray(G_to_save)

    def _get_T(ch_idx):
        """Return a 1D time vector for the chosen channel."""
        T = T_to_save
        if isinstance(T, (list, tuple)):
            T = T[ch_idx]
        T = np.asarray(T).squeeze()
        if T.ndim == 2:
            if T.shape[0] in (1, 2):
                T = T[min(ch_idx, T.shape[0]-1), :]
            else:
                T = T.ravel()
        return T

    # --- window & figure -----------------------------------------------------
    profile_window = Toplevel(parent_window)
    profile_window.title("Select Profile Pixel Range")

    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)

    click_marker = None   # <<< REQUIRED for nonlocal marker logic

    canvas = FigureCanvasTkAgg(fig, master=profile_window)
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=3, sticky='nsew')

    coord_label = Label(profile_window, text="Coordinates: x= , y=")
    coord_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

    # --- controls ------------------------------------------------------------
    start_pixel_var = IntVar(value=0)
    num_pixels_var = IntVar(value=1)

    ch_names = ['Channel 1'] if _get_channel_count() == 1 else ['Channel 1', 'Channel 2']
    channel_var = tk.StringVar(value=ch_names[0])

    Label(profile_window, text="Start Pixel:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
    Entry(profile_window, textvariable=start_pixel_var).grid(row=0, column=1, padx=10, pady=10, sticky='ew')

    Label(profile_window, text="Number of Pixels to Average:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
    Entry(profile_window, textvariable=num_pixels_var).grid(row=1, column=1, padx=10, pady=10, sticky='ew')

    Label(profile_window, text="Select Channel:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
    tk.OptionMenu(profile_window, channel_var, *ch_names).grid(row=2, column=1, padx=10, pady=10, sticky='ew')

    # --- plotting logic ------------------------------------------------------
    def plot_profile():
        nonlocal click_marker

        start = int(start_pixel_var.get())
        npx = max(1, int(num_pixels_var.get()))
        end = start + npx
        ch_idx = 0 if channel_var.get() == 'Channel 1' else 1

        G = _get_G(ch_idx)
        T = _get_T(ch_idx).astype(float)

        if start < 0:
            start = 0
        if end > G.shape[0]:
            show_message(
                "Error",
                f"End pixel index {end-1} exceeds data range (0..{G.shape[0]-1})."
            )
            return

        avg = np.mean(G[start:end, :], axis=0)

        T = T.squeeze()
        if T.ndim != 1:
            T = T.ravel()

        if T.size != avg.size:
            show_message(
                "Error",
                f"Time length ({T.size}) != profile length ({avg.size})."
            )
            return

        ax.clear()
        click_marker = None   # reset marker on re-plot

        ax.plot(T, avg, color='black')
        ax.set_title(
            f'{channel_var.get()} Profile: pixels {start}–{end-1}',
            fontsize=16
        )
        ax.set_xlabel(r'$\tau$', fontsize=16)
        ax.set_ylabel(r'G ($\tau$,dr)', fontsize=16)
        ax.grid(True)

        if np.all(T > 0):
            ax.set_xscale('log')
            ax.set_xlim(T.min(), T.max())

        y0, y1 = float(np.min(avg)), float(np.max(avg))
        if y0 == y1:
            y1 = y0 + 1e-12
        ax.set_ylim(y0, y1)

        canvas.draw_idle()

    def cut_to_1s():
        if not ax.lines:
            show_message("Info", "Please plot a profile first.")
            return

        line = ax.lines[0]
        T = line.get_xdata()
        avg = line.get_ydata()

        mask = T <= 1.0
        if not np.any(mask):
            show_message("Error", "No time points ≤ 1 s found.")
            return

        line.set_data(T[mask], avg[mask])
        ax.set_xlim(0, 1.0)
        y0, y1 = float(np.min(avg[mask])), float(np.max(avg[mask]))
        if y0 == y1:
            y1 = y0 + 1e-12
        ax.set_ylim(y0, y1)

        canvas.draw()

    # --- marker click logic --------------------------------------------------
    def on_click(event):
        nonlocal click_marker

        if event.inaxes is not ax:
            return

        x, y = event.xdata, event.ydata
        coord_label.config(text=f"Coordinates: x={x:.4f}, y={y:.6f}")

        if click_marker is None:
            click_marker, = ax.plot(
                [x], [y],
                marker='x',
                color='red',
                markersize=10,
                markeredgewidth=2,
                linestyle='None'
            )
        else:
            click_marker.set_data([x], [y])

        canvas.draw_idle()

    canvas.mpl_connect('button_press_event', on_click)

    tk.Button(profile_window, text="Plot Profile", command=plot_profile)\
        .grid(row=2, column=2, padx=10, pady=10, sticky='ew')

    tk.Button(profile_window, text="Cut at 1 s", command=cut_to_1s)\
        .grid(row=3, column=2, padx=10, pady=10, sticky='ew')

    profile_window.grid_columnconfigure(1, weight=1)
    profile_window.grid_rowconfigure(4, weight=1)
    profile_window.grid_rowconfigure(5, weight=0)

    profile_window.update_idletasks()



def open_fitacf_window():
    global G_to_save, T_to_save

    fit_window = tk.Toplevel()
    fit_window.title("Fit ACF")
    fit_window.geometry("900x700")

    # Input frame
    input_frame = Frame(fit_window)
    input_frame.pack(pady=10, padx=20, anchor="n")

    # Row 0: Pixel start index
    Label(input_frame, text="Pixel Start Index:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    pixel_entry = Entry(input_frame)
    pixel_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

    # Row 1: Number of pixels to average
    Label(input_frame, text="Number of Pixels to Average:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    average_entry = Entry(input_frame)
    average_entry.grid(row=1, column=1, sticky='w', padx=5, pady=5)

    # Row 2: w0 and wz
    Label(input_frame, text="w0:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    w0_entry = Entry(input_frame)
    w0_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)

    Label(input_frame, text="wz:").grid(row=2, column=2, sticky='e', padx=5, pady=5)
    wz_entry = Entry(input_frame)
    wz_entry.grid(row=2, column=3, sticky='w', padx=5, pady=5)

    # Row 3: Dim
    Label(input_frame, text="Dim:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    dim_entry = Entry(input_frame)
    dim_entry.grid(row=3, column=1, sticky='w', padx=5, pady=5)

    # Row 4: Channel selector
    Label(input_frame, text="Channel:").grid(row=4, column=0, sticky='e', padx=5, pady=5)
    channel_var = tk.StringVar(value="Channel 1")
    channel_menu = tk.OptionMenu(input_frame, channel_var, "Channel 1", "Channel 2")
    channel_menu.grid(row=4, column=1, sticky='w', padx=5, pady=5)

    # Row 5: Fit button
    def run_fit():
        try:
            pixel = int(pixel_entry.get())
            average = int(average_entry.get())
            w0 = float(w0_entry.get())
            wz = float(wz_entry.get())
            dim = int(dim_entry.get())
            channel_choice = channel_var.get()

            # Select channel
            if len(G_to_save) != 2:
                G_matrix = G_to_save
                time = T_to_save
            elif len(G_to_save) == 2:
                if channel_choice == "Channel 1":
                    G_matrix = np.asarray(G_to_save[0], dtype=float)
                    time = T_to_save[0]
                elif channel_choice == "Channel 2":
                    G_matrix = np.asarray(G_to_save[1], dtype=float)
                    time = T_to_save[1]
                else:
                    raise ValueError("Invalid channel selected.")
            else:
                raise ValueError("Invalid G_to_save dimensions.")

            if pixel + average > G_matrix.shape[0]:
                raise IndexError("Pixel + average exceeds available data.")

            G_avg = np.mean(G_matrix[pixel:pixel + average, :], axis=0)
            
            G_fit, td, N, D = fit_acf(time, G_avg, w0, wz, dim=dim)

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(time, G_avg, 'bo', label='Data')
            ax.plot(time, G_fit, 'r-', label='Fitted Curve')
            ax.set_xscale('log')
            ax.set_xlabel('Time')
            ax.set_ylabel('G')
            ax.set_title(f'{channel_choice} ACF Fit: td={td:.2e}s, N={N:.2f}, D={D:.2e} μm²/s')
            ax.legend()
            ax.grid(True)

            for widget in plot_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            result_label.config(text=f"Estimated N = {N:.2f}\nEstimated D = {D:.2e} μm²/s")

        except Exception as e:
            print(f"Error: {e}")

    fit_button = tk.Button(input_frame, text="Fit ACF", command=run_fit)
    fit_button.grid(row=5, column=0, columnspan=2, pady=(10, 0))

    # Row 6: Result label
    result_label = Label(input_frame, text="", fg="blue", justify='left')
    result_label.grid(row=6, column=0, columnspan=4, pady=(5, 10))

    # Plot frame
    plot_frame = Frame(fit_window)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

def display_plot(pcf_window):
    '''
    This function creates an interface for the computed pCF to be plotted on. It also includes the buttons useful for manipulating the
    resulted kimogram.
    
    Returns
    -------
    None.
    '''
    print("[DEBUG] Entered display_plot()")  # <-- Add this first
    try:
        # Generate the plot (ensure this function returns a Matplotlib figure)
        fig = apply_pCF()
        
        # Create a Tkinter window
        plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
        plot_window.title("pCF Plot")
        plot_window.geometry("1200x800")  # Larger default size

# Use grid layout manager
        plot_window.rowconfigure(0, weight=1)  # Plot frame row
        plot_window.rowconfigure(1, weight=0)  # Button frame row
        plot_window.columnconfigure(0, weight=1)

# Frame for the plot
        frame = Frame(plot_window)
        frame.grid(row=0, column=0, sticky="nsew")  # Fill entire top part
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

# Canvas for the figure
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

# Frame for the buttons
        button_frame = Frame(plot_window)
        button_frame.grid(row=1, column=0, pady=10)

# Buttons
        save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
        save_plot_button.pack(side=tk.LEFT, padx=5)

        save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
        save_data_button.pack(side=tk.LEFT, padx=5)

        save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
        save_time_button.pack(side=tk.LEFT, padx=5)

        profiles_button = tk.Button(button_frame, text="Profiles", command=lambda : open_profiles_window(plot_window), **button_style)
        profiles_button.pack(side=tk.LEFT, padx=5)
        
        fitacf_button = tk.Button(button_frame, text="Fit ACF", command=open_fitacf_window, **button_style)
        fitacf_button.pack(side=tk.LEFT, padx=5)
        # Schedule updates
        pcf_window.after(0, hide_working_and_update_result, "Plot Applied!")


        # Ensure the window is shown
        plot_window.mainloop()  # Ensure to call this for the window to stay open
        print("[DEBUG] Mid-function checkpoint")  # <-- Add this in middle
    except Exception as e:
        print(f"[CRITICAL] display_plot failed: {str(e)}")
        raise
    


def display_ccplot():
    '''
    this function creates an interface for the computed cross-pcf to be plotted on. 
    It also includes the buttons more useful for manipulating the resulted kimogram.

    Returns
    -------
    None.

    '''
    # Generate the plot
    fig = apply_ccpCF()
    plt.close('all')
    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    plot_window.title("ccpCF Plot")
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame

    # Create buttons and pack them side by side
    print("Creating buttons...")  # Debug print
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=lambda : open_profiles_window(plot_window), **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)
    #root.after(0, hide_working_and_update_result, "Plot Applied!")  # Schedule update from the main thread
    hide_working_and_update_result("Plot Applied!")  # Schedule update from the main thread
    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open

pot_entries = []
def ask_pot_exp():
    """
    Ask POT correction for experimental dataset.
    Returns [pot_ch1, pot_ch2]
    """
    pot_win = tk.Toplevel()
    pot_win.title("POT correction (experiment)")
    pot_win.grab_set()
    pot_win.resizable(False, False)

    tk.Label(pot_win, text="POT correction – experiment").grid(
        row=0, column=0, columnspan=2, pady=8
    )

    tk.Label(pot_win, text="CH1").grid(row=1, column=0, padx=5, pady=5)
    e1 = tk.Entry(pot_win, width=8)
    e1.insert(0, "1")
    e1.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(pot_win, text="CH2").grid(row=2, column=0, padx=5, pady=5)
    e2 = tk.Entry(pot_win, width=8)
    e2.insert(0, "1")
    e2.grid(row=2, column=1, padx=5, pady=5)

    result = {"val": [1.0, 1.0]}

    def on_ok():
        try:
            result["val"] = [float(e1.get()), float(e2.get())]
        except ValueError:
            pass
        pot_win.destroy()

    tk.Button(pot_win, text="OK", command=on_ok)\
        .grid(row=3, column=0, columnspan=2, pady=10)

    pot_win.wait_window()
    return result["val"]

def ask_pot_ctrl():
    """
    Ask POT correction for control datasets.
    Returns:
        [[pot_EGFP_ch1, pot_EGFP_ch2],
         [pot_mCherry_ch1, pot_mCherry_ch2]]
    """
    pot_win = tk.Toplevel()
    pot_win.title("POT correction (controls)")
    pot_win.grab_set()
    pot_win.resizable(False, False)

    tk.Label(pot_win, text="POT correction – controls")\
        .grid(row=0, column=0, columnspan=3, pady=8)

    labels = ["CH1", "CH2"]
    for j, lab in enumerate(labels):
        tk.Label(pot_win, text=lab).grid(row=1, column=j+1)

    tk.Label(pot_win, text="EGFP").grid(row=2, column=0, padx=5)
    tk.Label(pot_win, text="mCherry").grid(row=3, column=0, padx=5)

    entries = [[None, None], [None, None]]

    for i in range(2):
        for j in range(2):
            e = tk.Entry(pot_win, width=8)
            e.insert(0, "1")
            e.grid(row=i+2, column=j+1, padx=5, pady=5)
            entries[i][j] = e

    result = {"val": [[1.0, 1.0], [1.0, 1.0]]}

    def on_ok():
        try:
            result["val"] = [
                [float(entries[0][0].get()), float(entries[0][1].get())],
                [float(entries[1][0].get()), float(entries[1][1].get())]
            ]
        except ValueError:
            pass
        pot_win.destroy()

    tk.Button(pot_win, text="OK", command=on_ok)\
        .grid(row=4, column=0, columnspan=3, pady=10)

    pot_win.wait_window()
    return result["val"]


def upload_filter_files():
    global file_list_label
    '''
    Function for uploading files from controls to filter the collected data.
    
    Returns
    -------
    None.
    '''
    global f1, f2, f1_extra, f2_extra 
    global pot_correction_control, pot_correction_exp
    
    pot_correction_control = ask_pot_ctrl()
    pot_correction_exp = ask_pot_exp()
    f1 = filedialog.askopenfilenames(filetypes=[("Control files for protein 1", "*.tiff"), 
                                             ("Control files for protein 1", "*.tif"),
                                             ("Control files for protein 1", "*.czi"),
                                             ("Control files for protein 1", "*.b64"),
                                             ("Control files for protein 1", "*.txt")])
    
    if f1:
        # Check if any of the selected files are .b64
        if any(file.endswith('.b64') for file in f1) or any(file.endswith('.txt') for file in f1):
            # Upload additional files for .B64 case
            f1_extra = filedialog.askopenfilenames(filetypes=[("Additional files for protein 1", "*.tiff;*.tif;*.czi;*.b64;*.txt")])
            f2 = filedialog.askopenfilenames(filetypes=[("Control files for protein 2", "*.tiff;*.tif;*.czi,;*.b64;*.txt")])
            f2_extra = filedialog.askopenfilenames(filetypes=[("Additional files for protein 2", "*.tiff;*.tif;*.czi;*.b64;*.txt")])
            
            if f1_extra and f2_extra:
                file_list_label.config(text=(
                    "Files for protein 1: uploaded"
                    "Files for protein 2: uploaded"
                ))
                pass
            else:
                file_list_label.config(text="Not all additional files were selected.")
        else:
            f2 = filedialog.askopenfilenames(filetypes=[("Control files for protein 2", "*.tiff"), 
                                             ("Control files for protein 2", "*.tif"),
                                             ("Control files for protein 2", "*.czi"),
                                             ("Control files for protein 2", "*.b64"),
                                             ("Control files for protein 2", "*.txt")])
            if f2:
                file_list_label.config(text="Files uploaded")
                pass
            else:
                file_list_label.config(text="No files selected for protein 2")
    else:
        file_list_label.config(text="No files selected for protein 1")


def open_hv_window(parent_window):
    hv_window = Toplevel(parent_window)
    hv_window.title("Select average factor")

    def plot_lines():
        global stack
        start = start_var.get()
        num = num_var.get()

        if start < 0 or start >= stack.shape[0]:
            print("Error: Start image out of range.")
            return

        if num <= 0:
            print("Error: Number of images to average must be positive.")
            return
        # Calculate the average profile over the specified range
        subset = stack[start:,:,:]
        average = np.mean(subset, axis=(1,2))
        def chunked_average_1d(array, num):
            length = len(array) - (len(array) % num)  # trim end if needed
            return array[:length].reshape(-1, num).mean(axis=1)

        # Example usage
        average = chunked_average_1d(average, num)
        # Create a new figure with adjusted size
        profile_fig = Figure(figsize=(8, 6), dpi=100)
        profile_ax = profile_fig.add_subplot(111)
        profile_ax.plot([x for x in range(len(average))], average, color='black')
        
        # Set labels with larger font size
        profile_ax.set_title('Average intensity', fontsize=16)
        profile_ax.set_xlabel('Images', fontsize=14)
        profile_ax.set_ylabel('Intensity', fontsize=14)
        
        # Increase padding and font size for ticks
        profile_ax.tick_params(axis='both', which='major', labelsize=10)
        profile_ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # Add grid for better readability
        profile_ax.grid(True)
        
        # Create and add the canvas
        profile_canvas = FigureCanvasTkAgg(profile_fig, master=hv_window)
        profile_canvas.draw()
        
        # Use grid() instead of pack()
        profile_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky='nsew')

    # Add entries for starting pixel and number of pixels
    start_var = IntVar(value=0)
    num_var = IntVar(value=1)
    
    start_label = Label(hv_window, text="Start image:")
    start_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    
    start_entry = Entry(hv_window, textvariable=start_var)
    start_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    
    num_label = Label(hv_window, text="Number of images to average:")
    num_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    
    num_entry = Entry(hv_window, textvariable=num_var)
    num_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
    
    plot_button = tk.Button(hv_window, text="Plot Intensity", command=plot_lines)
    plot_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    hv_window.grid_columnconfigure(1, weight=1)
    hv_window.grid_rowconfigure(3, weight=1)  # Allow row 3 to expand if needed


        

def open_hugevector_window(parent_window):
    '''
    this function is used to plot the horizontal profile of the kimogram, i.e. averaged lines.
    This is particularly helpful to identify regions of different intensity. 
    The plot is done in a new window where the user selects wich lines to average. 
    If there are 2 kimograms, use the tgogle function to change which one you want for this window.

    Returns
    -------
    None.

    '''
    hv_window = Toplevel(parent_window)
    hv_window.title("Select average factor")

    def plot_lines():
        global original_kimograms, current_index
        try:
            original_kimogram=original_kimograms[current_index]
        except:
            print('no current axis')
            original_kimogram=original_kimograms[0]
        start = start_var.get()
        num = num_var.get()

        if start < 0 or start >= original_kimogram.shape[0]:
            print("Error: Start line out of range.")
            return

        if num <= 0:
            print("Error: Number of lines to average must be positive.")
            return


        # Calculate the average profile over the specified range
        subset = original_kimogram[start:,:]
        average = np.mean(subset, axis=1)
        def chunked_average_1d(array, num):
            length = len(array) - (len(array) % num)  # trim end if needed
            return array[:length].reshape(-1, num).mean(axis=1)

        # Example usage
        average = chunked_average_1d(average, num)
        # Create a new figure with adjusted size
        profile_fig = Figure(figsize=(8, 6), dpi=100)
        profile_ax = profile_fig.add_subplot(111)
        profile_ax.plot([x for x in range(len(average))], average, color='black')
        
        # Set labels with larger font size
        profile_ax.set_title(f'Average intensity', fontsize=16)
        profile_ax.set_xlabel('lines', fontsize=14)
        profile_ax.set_ylabel('Intensity', fontsize=14)
        
        # Increase padding and font size for ticks
        profile_ax.tick_params(axis='both', which='major', labelsize=10)
        profile_ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # Add grid for better readability
        profile_ax.grid(True)
        
        # Create and add the canvas
        profile_canvas = FigureCanvasTkAgg(profile_fig, master=hv_window)
        profile_canvas.draw()
        
        # Use grid() instead of pack()
        profile_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky='nsew')

    # Add entries for starting pixel and number of pixels
    start_var = IntVar(value=0)
    num_var = IntVar(value=1)
    
    start_label = Label(hv_window, text="Start line:")
    start_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    
    start_entry = Entry(hv_window, textvariable=start_var)
    start_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    
    num_label = Label(hv_window, text="Number of lines to Average:")
    num_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    
    num_entry = Entry(hv_window, textvariable=num_var)
    num_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
    
    plot_button = tk.Button(hv_window, text="Plot Intensity", command=plot_lines)
    plot_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    hv_window.grid_columnconfigure(1, weight=1)
    hv_window.grid_rowconfigure(3, weight=1)  # Allow row 3 to expand if needed



def zoom_in():
    '''
    This function applies zoom in over the kimogram.
    '''
    global rect_selector, current_index, original_limits
    set_current_axis(current_index)
    if current_ax is None:
        show_message('Error', 'No kimogram has been selected. Try with the Toggle kimogram button')
        return

    # Save the original limits for the current axis before zooming
    original_limits[current_index] = {
        'xlim': current_ax.get_xlim(),
        'ylim': current_ax.get_ylim()
    }

    if rect_selector:
        rect_selector.set_active(False)
        rect_selector.disconnect_events()
        rect_selector = None

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        current_ax.set_xlim(min(x1, x2), max(x1, x2))
        current_ax.set_ylim(min(y1, y2), max(y1, y2))
        canvas.draw()
        if rect_selector:
            rect_selector.set_active(False)

    # Create a new selector
    rect_selector = RectangleSelector(
        current_ax, onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )

def zoom_out():
    '''
    Zoom out over the selected kimogram.
    '''
    global current_ax, original_limits, rect_selector
    if current_ax is None:
        show_message('Error', 'No kimogram has been selected. Try with the Toggle kimogram button')
        return

    if current_index in original_limits:
        current_ax.set_xlim(original_limits[current_index]['xlim'])
        current_ax.set_ylim(original_limits[current_index]['ylim'])

        # Fully disconnect and destroy the selector
        if rect_selector is not None:
            try:
                rect_selector.set_active(False)
                rect_selector.disconnect_events()
                rect_selector = None
            except Exception as e:
                print("Error clearing RectangleSelector:", e)

        canvas.draw()
    else:
        show_message('Error', 'Original limits are not available for this kimogram.')


def open_help():
    url = "https://github.com/natiph/pcf_interface/wiki/Tutorial" 
    webbrowser.open(url)

def hide_working_and_update_result(result_text):
        working_label.grid_forget()
        result_label.config(text=result_text)
        result_label.grid(row=3, column=0, pady=10)

def hide_result_label():
        result_label.grid_forget()

#███████ ██ ██      ████████ ███████ ██████  
#██      ██ ██         ██    ██      ██   ██ 
#█████   ██ ██         ██    █████   ██████  
#██      ██ ██         ██    ██      ██   ██ 
#██      ██ ███████    ██    ███████ ██   ██ 
                                            
                                           
def find_p(f1, f2, pot_correction=[[1, 1], [1, 1]]):
    """
    Compute filtering matrix p using the same convention as filtrado_norm.

    Rows: detectors (CH1, CH2)
    Columns: proteins (protein 1, protein 2)
    pot_correction[i] = [corr_CH1, corr_CH2] for each control
    """

    try:
        # Case: CZI-like arrays
        if f1.shape[0] == 1 and f2.shape[0] == 1:

            # Apply POT correction FIRST
            f1_ch1 = f1[0, :, 0, 0, :, :, 0] * pot_correction[0][0]
            f1_ch2 = f1[0, :, 1, 0, :, :, 0] * pot_correction[0][1]

            f2_ch1 = f2[0, :, 0, 0, :, :, 0] * pot_correction[1][0]
            f2_ch2 = f2[0, :, 1, 0, :, :, 0] * pot_correction[1][1]

            p = np.array([
                [np.mean(f1_ch1), np.mean(f2_ch1)],  # CH1 detector
                [np.mean(f1_ch2), np.mean(f2_ch2)]   # CH2 detector
            ])

            print("I created p correctly (GUI, fixed)")
            return p

    except Exception:
        pass

    try:
        # Case: simple matrices / lines
        if len(f1) == 2 and len(f2) == 2:

            f1_ch1 = f1[0] * pot_correction[0][0]
            f1_ch2 = f1[1] * pot_correction[0][1]

            f2_ch1 = f2[0] * pot_correction[1][0]
            f2_ch2 = f2[1] * pot_correction[1][1]

            p = np.array([
                [np.mean(f1_ch1), np.mean(f2_ch1)],
                [np.mean(f1_ch2), np.mean(f2_ch2)]
            ])

            print("I created p correctly (GUI, fixed)")
            return p

    except Exception:
        pass

    raise ValueError("Input arrays do not have the expected shape.")




def bleeding_filter(kimograms,first_line, last_line,p):
    '''
    this function takes the files provided with the chosen parameters and the p calculated to actually filter the data.

    Parameters
    ----------
    kimograms : TYPE list of matrices each one corresponding to the data uploaded
        
    first_line : TYPE int
        DESCRIPTION. first line to analyse
    last_line : TYPE int
        DESCRIPTION.last line to analyse
    p : TYPE (2x2) matrix
        DESCRIPTION.filtering matrix

    Returns
    -------
    ch1_filtrado : TYPE matrix
        DESCRIPTION. the data provided by the first kimogram but filtered
    ch2_filtrado : TYPE matrix
        DESCRIPTION.  the data provided by the second kimogram but filtered

    '''
    global pot_correction_exp
    ch1 = kimograms[0][first_line:last_line]*pot_correction_exp[0]
    ch2 = kimograms[1][first_line:last_line]*pot_correction_exp[1]
    ch1_filtrado, ch2_filtrado = ch1.astype(float, copy=True), ch1.astype(float, copy=True)
    for j in range(ch1.shape[0]):
        for k in range(ch1.shape[1]):
            try:
                a = np.linalg.solve(p, [ch1[j, k], ch2[j, k]])
                ch1_filtrado[j, k] = max(a[0], 0)
                ch2_filtrado[j, k] = max(a[1], 0)
            except np.linalg.LinAlgError as e:
                print(f"Error solving for indices {j}, {k}: {e}")
                ch1_filtrado[j, k] = 0
                ch2_filtrado[j, k] = 0
    print(f'ch1 filtrado shape = {np.shape(ch1_filtrado)}')
    
    return ch1_filtrado, ch2_filtrado



def plot_two_kimograms(t_log, G_log, G2_log, cruzado=True):
    '''
    This function plots the correlation calculated somewhere else.
    '''
    global G_to_save
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)

    if cruzado:
        l1 = 'Ch1 to Ch2'
        l2 = 'Ch2 to Ch1'
    else:
        l1 = 'Channel 1'
        l2 = 'Channel 2'

    try:
        if G_log is not None and G2_log is not None:
            cmap_value = cmap_var.get() if hasattr(cmap_var, "get") else cmap_var
            cmap = resolve_cmap(cmap_value)
            plot_on_ax(ax1, G_log, t_log, l1, cmap=cmap)
            plot_on_ax(ax2, G2_log, t_log, l2, cmap=cmap)
            G_to_save = [G_log, G2_log]  # Store G_log for saving
        else:
            print("G_log or G2_log is None")
            return None
    except Exception as e:
        print(f"Error in plot_two_kimograms: {e}")
        return None

    fig.tight_layout(pad=2.0)
    return fig



def plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=True):
    '''
    Computes the corresponding pair correlation for the filtered data.
    '''
    global G_to_save, T_to_save
    try:
        if cruzado:
            G, T = crosspCF(ch1_filtrado, ch2_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            G2, T2 = crosspCF(ch2_filtrado, ch1_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            print('LA CORRELACION ES CRUZADA')
        else:
            print(f'ch1 shape: {ch1_filtrado.shape}, ch2 shape: {ch2_filtrado.shape}')
            print(f'dr: {dr}, tp: {line_time / 100 / ch1_filtrado.shape[1]}')
            print(f'reverse: {reverse}')
            G, T = crosspCF(ch1_filtrado, ch1_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            print('LA CORRELACION ES EN CADA CANAL POR SEPARADO')

            G2, T2 = crosspCF(ch2_filtrado, ch2_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
        print(f"G shape: {G.shape}, G2 shape: {G2.shape}")  # Debug print

        if len(G) == 0 or len(G2) == 0:
            show_message('Error', "G or G2 returned from crosspCF is empty.")
            return None

        x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint=False)
        t_lineal = T[:, 0]
        t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)

        G_basura = []
        for i in x1:
            G_basura.append(list(G[i]))
        G = np.asarray(G_basura).transpose()

        t = []
        for i in x1:
            t.append(t_lineal[i])
        t_lineal = np.asarray(t)

        G_log = np.empty_like(G)
        for i, gi in enumerate(G):
            G_log[i] = np.interp(t_log, t_lineal, gi)

        G_log = gaussian_filter(G_log, sigma=sigma)
        x1 = np.geomspace(1, len(G2), 256, dtype=int, endpoint=False)

        G2_basura = []
        for i in x1:
            G2_basura.append(list(G2[i]))
        G2 = np.asarray(G2_basura).transpose()

        G2_log = np.empty_like(G2)
        for i, gi in enumerate(G2):
            G2_log[i] = np.interp(t_log, t_lineal, gi)
        G2_log = gaussian_filter(G2_log, sigma=sigma)

        G_to_save = [G_log, G2_log]
        T_to_save = t_log

        return plot_two_kimograms(t_log, G_log, G2_log, cruzado)

    except Exception as e:
        print(f"Error in plot_filtered_data: {e}")
        show_message('Error', f"An error occurred while plotting filtered data: {e}")
        return None



global pot_correction_control, pot_correction_exp
pot_correction_exp = [1,1]
pot_correction_control = [[1,1],[1,1]]

def apply_filter_ccpcf(cross):
    '''
    this function takes the uploaded filtering files, transform them into matrixs, performes the filtering (using the bleeding_filter function)
    then plots the filter correlation. It provides two plots. Either ther pCF for both channels or the ccpCF in both directions.

    Parameters
    ----------
    cross : TYPE str
        DESCRIPTION. either 'ccpcf' or 'pcf'. 'pcf' will be interpreted otherwise

    Returns
    -------
    the plots

    '''
    global original_kimograms, f1, f2, f1_extra, f2_extra,pot_correction_control, pot_correction_exp
    if f1 and f2:
        try:
            f1_data = []
            f2_data = []
            
            for file in f1:
                if file.endswith(('.tiff', '.tif')):
                    f1_data.append(np.array(tif_imread(file).astype(np.uint16)))
                elif file.endswith('.czi'):
                    f1_data = imread(file)  # Load CZI file
                elif file.endswith('.b64'):
                    f1_data.append(read_B64(file))
                    print('I appended the files in f1')
                    f1_data.append(read_B64(f1_extra[0]))
                    print('I appended the file for f1_extra')
                elif file.endswith('.txt'):
                    f1_data.append(np.reshape(np.loadtxt(r'%s'%file),(128,128)))
                    f1_data.append(np.reshape(np.loadtxt(r'%s'%f1_extra),(128,128)))


            for file in f2:
                if file.endswith(('.tiff', '.tif')):
                    f2_data.append(np.array(tif_imread(file).astype(np.uint16)))
                elif file.endswith('.czi'):
                    f2_data = imread(file)  # Load CZI file
                elif file.endswith('.b64'):
                    f2_data.append(read_B64(file))
                    print('I appended f2')
                    f2_data.append(read_B64(f2_extra[0]))
                    print('I appended the file for f2_extra')
                elif file.endswith('.txt'):
                    f2_data.append(np.reshape(np.loadtxt(r'%s'%file),(128,128)))
                    f2_data.append(np.reshape(np.loadtxt(r'%s'%f2_extra),(128,128)))
                    
            try:
                p = find_p(f1_data, f2_data,pot_correction_control)
                print('find_p output:', p)  # Debug: check the output of find_p
            except:
                print('error finding p')
            if p is None or not isinstance(p, np.ndarray):
                raise ValueError("Invalid output from find_p function.")
            data = get_table_data()
            try:
                first_line = int(data.get("First Line", ""))
                last_line = int(data.get("Last Line", ""))
            except:
                show_message('Error', "You need to especify the parameters in the table below")
                pass
            line_time = float(data.get("Line Time (ms)", 0.0))  # Default to 0.0
            dr = int(data.get("Distance (px)", 0))      # Default to 0 if not found
            sigma = [int(data.get("H Smoothing (px)", 0)), int(data.get("V Smoothing (lines)", 0))]  # Default to 0 if not found
            reverse = data.get('Reverse', False)  # Default to False if not found
            print('the table data is ok')
           
            
            ch1_filtrado, ch2_filtrado = bleeding_filter(original_kimograms,first_line, last_line,p)
            print(p)
            if cross=='ccpcf':
                print('Entré al if cross de apply_filter_ccpcf')
                fig = plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=True)
            else:
                try:
                    fig = plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=False)
                except:
                    print('error in plot_filtered_data')
                    pass
            hide_working_and_update_result("Fliter Applied")  # Schedule update from the main thread
            return fig
        except:
            print("Error in apply_filter_ccpcf function")
        
            
def compute_and_display_plot(cross, parent_window):
    fig = apply_filter_ccpcf(cross)  # Run the filtering computation only once
    parent_window.after(0, display_filter_ccpcf, fig,cross)  # Schedule the plot in the main thread

def display_filter_ccpcf(fig, cross):
    '''
    Displays the filtered correlation plots in a new window.

    Parameters
    ----------
    fig : matplotlib figure
        The computed figure to display.

    Returns
    -------
    None
    '''
    plt.close('all')

    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    if cross=='ccpcf':
        plot_window.title("Filtered ccpCF Plot")
    else:
        plot_window.title("Filtered pCF Plot")
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame
    
    # Create buttons and pack them side by side
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=lambda: open_profiles_window(plot_window), **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)
    hide_working_and_update_result("Plot Applied!")  # Schedule update from the main thread
    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open

def run_task(task_function, parent_window):
        working_label.grid(row=3, column=0, pady=10)
        task_thread = threading.Thread(target=task_function)
        task_thread.start()
        parent_window.update()
def on_apply_filter(tipo, parent_window):
    run_task(compute_and_display_plot(tipo, parent_window), parent_window)

def update_kimogram_label():
    # Update the label to show the current kimogram
    kimogram_label.config(text=f"Current Kimogram: {current_index+1}")
    kimogram_label.grid(row=1, column=6, pady=(5, 0))  # Add some padding above the label

def toggle_kimogram():
    '''
    This function is only relevant if there are two kimograms.
    It is used to change over which kimogram the buttons such as zoom or h-lines will be used.
    '''
    global current_index
    if current_index == 0:
        current_index = 1
    else:
        current_index = 0
    set_current_axis(current_index)
    update_kimogram_label()  # Update the label text
    canvas.draw()



def load_and_display_images(file_paths,parent_window,control_frame):
    global images
    loading_window = show_loading_popup(parent_window)
    def run():
        try:
            images = []
            for fp in file_paths:
                if fp.endswith('.czi'):
                    im = imread(fp)[0,:,0,0,:,:,0]  # should return stack (T, Y, X)
                    images.append(im)
                elif fp.endswith(('.tif', '.tiff')):
                    im = tif_imread(fp)  # should return stack (T, Y, X)
                    images.append(im)
                elif fp.endswith('.b64'):
                    im = read_B64(fp,tipo='image')  # assume returns stack
                    images.append(im)

            all_stacks = np.concatenate(images, axis=0)  # combine into one big stack
            parent_window.after(0, lambda: open_image_viewer_with_nb_controls(all_stacks, parent_window, control_frame))
            parent_window.after(0, loading_window.destroy)
        except Exception as e:
            import traceback
            error_msg = f"Unhandled error:\n{str(e)}\n\n{traceback.format_exc()}"
            parent_window.after(0, loading_window.destroy)
            parent_window.after(0, lambda: show_message("Error", error_msg))

    threading.Thread(target=run, daemon=True).start()

def load_images(parent_window, control_frame):
    file_paths = filedialog.askopenfilenames(filetypes=[("image files", "*.tiff;*.tif;*.czi;*.b64")])
    if file_paths:
        threading.Thread(target=load_and_display_images, args=(file_paths,parent_window,control_frame,), daemon=True).start()
    else:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")

def abrir_NB_ventana():
    button_style = {
            'bg': 'lightgrey',
            'fg': 'black',
            'padx': 12,
            'pady': 6,
            'relief': 'flat',
            'borderwidth': 2,
            'highlightbackground': '#d3d3d3',
            'highlightcolor': '#d3d3d3',
            'activebackground': '#5a5a5a',
            'activeforeground': 'white'
        }
    fondo = 'white'
    # Create a new Toplevel window for the pCF window
    nb_window = tk.Toplevel()
    nb_window.title("N&B Window")  # Title for the new pCF window
    nb_window.geometry("800x800")  # Adjust size as needed
    
    # Set up the specific layout
    nb_window.grid_rowconfigure(1, weight=1)
    nb_window.grid_columnconfigure(0, weight=1)
    
    # Controls frame for navigation and zoom buttons
    control_frame = tk.Frame(nb_window, bg=fondo)
    control_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=10)

    
    load_button = tk.Button(control_frame, text="Load file", command=lambda: load_images(image_frame,control_frame), **button_style)
    load_button.pack(side=tk.LEFT, padx=5, pady=10)

    image_frame = tk.Frame(nb_window, bg=fondo)
    image_frame.grid(row=1, column=0, sticky="nsew")
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_columnconfigure(0, weight=1)

def open_image_viewer_with_nb_controls(stack, parent_frame, control_frame):
    global mask

    for widget in parent_frame.winfo_children():
        widget.destroy()

    # =======================
    # Image viewer
    # =======================
    plot_frame = tk.Frame(parent_frame, height=450)
    plot_frame.pack(side="top", fill="x", padx=10, pady=(10, 5))
    plot_frame.pack_propagate(False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im_plot = ax.imshow(stack[0], cmap="viridis")
    ax.axis("off")

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    current_frame = tk.IntVar(value=0)

    def update_plot(val=None):
        idx = current_frame.get()
        if 0 <= idx < len(stack):
            im_plot.set_data(stack[idx])
            ax.set_title(f"Frame {idx+1} / {len(stack)}")
            canvas.draw_idle()

    slider = tk.Scale(
        parent_frame,
        from_=0,
        to=len(stack) - 1,
        orient="horizontal",
        variable=current_frame,
        command=update_plot,
    )
    slider.pack(fill="x", padx=10)

    plt.close(fig)

    # =======================
    # Controls
    # =======================
    bottom_controls = tk.Frame(parent_frame)
    bottom_controls.pack(side="top", fill="x", padx=10, pady=10)

    # ---- Parameters ----
    param_frame = tk.LabelFrame(bottom_controls, text="N&B Parameters", padx=10, pady=10)
    param_frame.pack(side="left", fill="y", padx=(0, 10))

    entry_start = tk.Entry(param_frame, width=8)
    entry_end = tk.Entry(param_frame, width=8)
    entry_s = tk.Entry(param_frame, width=8)
    entry_offset = tk.Entry(param_frame, width=8)
    entry_sigma = tk.Entry(param_frame, width=8)

    entry_s.insert(0, "1")
    entry_offset.insert(0, "0")
    entry_sigma.insert(0, "0")

    labels = ["Start", "End", "S", "Offset", "Sigma"]
    entries = [entry_start, entry_end, entry_s, entry_offset, entry_sigma]

    for i, (lab, ent) in enumerate(zip(labels, entries)):
        tk.Label(param_frame, text=lab).grid(row=i, column=0, sticky="e")
        ent.grid(row=i, column=1)

    # ---- Filters ----
    filter_frame = tk.LabelFrame(bottom_controls, text="Filters (Min / Max)", padx=10, pady=10)
    filter_frame.pack(side="left", fill="y")

    def make_filter_row(row, label):
        tk.Label(filter_frame, text=label).grid(row=row, column=0, sticky="w")
        e_min = tk.Entry(filter_frame, width=8)
        e_max = tk.Entry(filter_frame, width=8)
        e_min.grid(row=row, column=1)
        e_max.grid(row=row, column=2)
        return e_min, e_max

    entry_I_min, entry_I_max = make_filter_row(1, "Intensity (I)")
    entry_N_min, entry_N_max = make_filter_row(2, "Number (N)")
    entry_B_min, entry_B_max = make_filter_row(3, "Brightness (B)")

    mask_status = tk.Label(control_frame, text="No mask set", fg="red")
    mask_status.pack(side="right", padx=5)

    # =======================
    # Mask
    # =======================
    def upload_mask():
        global mask
        files = filedialog.askopenfilenames(filetypes=[("Matrix", "*.txt")])
        if files:
            mask = np.loadtxt(files[0]).astype(bool).ravel()
            mask_status.config(text="Mask set", fg="green")

    # =======================
    # Save
    # =======================
    def save_nb_results(N, B):
        pixels = int(np.sqrt(len(N)))
        N = N.reshape(pixels, pixels)
        B = B.reshape(pixels, pixels)

        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            np.savetxt(path.replace(".csv", "_N.csv"), N, delimiter=",")
            np.savetxt(path.replace(".csv", "_B.csv"), B, delimiter=",")

    # =======================
    # Apply N&B
    # =======================
    def apply_nb():
        global mask

        start = int(entry_start.get() or 0)
        end = int(entry_end.get() or len(stack))
        S = float(entry_s.get())
        offset = float(entry_offset.get())
        sigma = float(entry_sigma.get())

        data = stack[start:end]
        I = np.mean(data, axis=0).ravel()
        VAR = np.var(data, axis=0).ravel()

        B = (VAR - sigma**2) / (S * (I - offset))
        N = (I - offset)**2 / (VAR - sigma**2)

        # ---- Filters ----
        I_min = float(entry_I_min.get() or -np.inf)
        I_max = float(entry_I_max.get() or np.inf)
        N_min = float(entry_N_min.get() or -np.inf)
        N_max = float(entry_N_max.get() or np.inf)
        B_min = float(entry_B_min.get() or -np.inf)
        B_max = float(entry_B_max.get() or np.inf)

        base_filter = (
            (I >= I_min) & (I <= I_max) &
            (N >= N_min) & (N <= N_max) &
            (B >= B_min) & (B <= B_max)
        )

        if mask is not None and mask.size == base_filter.size:
            base_filter &= mask

        # =======================
        # Results window
        # =======================
        shape = stack.shape[1:]
        win = tk.Toplevel(parent_frame)
        win.title("N&B Results")

        plots = tk.Frame(win,background='white')
        plots.pack(side="top", fill="both", expand=True)

        hists = tk.Frame(win)
        hists.pack(side="bottom", fill="both", expand=True)

        # --- estado filtro gaussiano ---
        gauss_state = {"type": None, "mu": None, "sigma": None}

        def apply_gaussian_filter():
            filt = base_filter.copy()

            if gauss_state["type"] == "I":
                lo = gauss_state["mu"] - gauss_state["sigma"]
                hi = gauss_state["mu"] + gauss_state["sigma"]
                filt &= (I >= lo) & (I <= hi)
            elif gauss_state["type"] == "B":
                lo = gauss_state["mu"] - gauss_state["sigma"]
                hi = gauss_state["mu"] + gauss_state["sigma"]
                filt &= (B >= lo) & (B <= hi)
            elif gauss_state["type"] == "N":
                lo = gauss_state["mu"] - gauss_state["sigma"]
                hi = gauss_state["mu"] + gauss_state["sigma"]
                filt &= (N >= lo) & (N <= hi)

            B_plot = B.copy()
            N_plot = N.copy()

            B_plot[~filt] = np.nan
            N_plot[~filt] = np.nan

            ax_B.clear()
            ax_N.clear()

            imB = ax_B.imshow(B_plot.reshape(shape), cmap="viridis")
            imN = ax_N.imshow(N_plot.reshape(shape), cmap="viridis")

            ax_B.set_title("Brightness (B)")
            ax_N.set_title("Number (N)")

            fig_B.canvas.draw_idle()
            fig_N.canvas.draw_idle()

        # ---- Maps ----
        fig_B, ax_B = plt.subplots(figsize=(4, 4))
        fig_N, ax_N = plt.subplots(figsize=(4, 4))

        ax_B.imshow(B.reshape(shape), cmap="viridis")
        ax_N.imshow(N.reshape(shape), cmap="viridis")
        ax_N.set_title('Number')
        ax_B.set_title('Brightness')
        FigureCanvasTkAgg(fig_B, master=plots).get_tk_widget().pack(side="left", expand=True)
        FigureCanvasTkAgg(fig_N, master=plots).get_tk_widget().pack(side="left", expand=True)

        # ---- Histograms ----
        for data, title, xlabel in [
            (I, "Intensity", "I"),
            (B, f"B (mean={np.nanmean(B):.2f})", "B"),
            (N, f"N (mean={np.nanmean(N):.2f})", "N"),
        ]:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(data[~np.isnan(data)], bins=30, color="slateblue")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Counts")
            FigureCanvasTkAgg(fig, master=hists).get_tk_widget().pack(side="left", expand=True)

        # ---- Gaussian buttons ----
        def open_gauss_popup(var, data):
            pop = tk.Toplevel(win)
            pop.title(f"Gaussian {var}")

            mu = tk.DoubleVar(value=np.nanmean(data))
            sig = tk.DoubleVar(value=np.nanstd(data))

            tk.Label(pop, text="μ").pack()
            tk.Scale(pop, from_=mu.get()-sig.get()*3, to=mu.get()+sig.get()*3,
                     resolution=0.01, orient="horizontal", variable=mu).pack(fill="x")

            tk.Label(pop, text="σ").pack()
            tk.Scale(pop, from_=0.01, to=3*sig.get(),
                     resolution=0.01, orient="horizontal", variable=sig).pack(fill="x")

            def apply():
                gauss_state["type"] = var
                gauss_state["mu"] = mu.get()
                gauss_state["sigma"] = sig.get()
                apply_gaussian_filter()

            tk.Button(pop, text="Apply", command=apply).pack(pady=5)

        gauss_frame = tk.Frame(win)
        gauss_frame.pack(pady=5)

        tk.Button(gauss_frame, text="Gauss I",
                  command=lambda: open_gauss_popup("I", I)).pack(side="left", padx=3)
        tk.Button(gauss_frame, text="Gauss B",
                  command=lambda: open_gauss_popup("B", B)).pack(side="left", padx=3)
        tk.Button(gauss_frame, text="Gauss N",
                  command=lambda: open_gauss_popup("N", N)).pack(side="left", padx=3)

        tk.Button(win, text="Save N&B", command=lambda: save_nb_results(N, B)).pack(pady=5)

    # =======================
    # Buttons
    # =======================
    tk.Button(control_frame, text="Apply N&B", command=apply_nb).pack(side="left", padx=5)
    tk.Button(control_frame, text="Set mask", command=upload_mask).pack(side="left", padx=5)


stack = None  # global to hold the loaded image stack


def load_images_and_update(parent_window, control_frame, im_plot, ax, slider, canvas,tipo):
    global stack
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image files", "*.tiff;*.tif;*.czi;*.b64")]
    )
    if not file_paths:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")
        return
    loading_window = show_loading_popup(parent_window)

    def run():
        global stack
        try:
            images = []
            for fp in file_paths:
                fpl = fp.lower()
                if fpl.endswith('.czi'):
                        im = load_czi_file(fp)[0]        # expected shape (T, Y, X)
                    
                elif fpl.endswith(('.tif', '.tiff')):
                    im = tif_imread(fp)            # expected shape (T, Y, X)
                    print(im.shape)
                    if im.shape[1]==1:
                        if im.shape[-1]==1:
                            im = im[:,0,:,:,0]
                    elif im.shape[0]==1:
                        im = im[0]
                        
                elif fpl.endswith('.b64'):
                    im = read_B64(fp,tipo='image')              # expected shape (T, Y, X)
                    print(im.shape)
                else:
                    continue

                # Make sure each is 3D
                if im.ndim == 2:
                    im = im[np.newaxis, ...]
                images.append(im)

            if not images:
                raise ValueError("No images read.")

            stack = np.concatenate(images, axis=0)

            # Update plot & slider on main thread
            parent_window.after(0, lambda: update_display(im_plot, ax, slider, canvas))
        except Exception as e:
            parent_window.after(
                0,
                lambda err=e: show_message("Error", f"Could not load images: {err}")
                )            
        finally:
            parent_window.after(0, loading_window.destroy)

    threading.Thread(target=run, daemon=True).start()


def _autoscale_image(im_plot, frame):
    # Robustly set color limits for the current frame
    vmin = float(np.nanmin(frame))
    vmax = float(np.nanmax(frame))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallback to a tiny range to avoid identical vmin/vmax
        vmax = vmin + 1e-12
    im_plot.set_clim(vmin=vmin, vmax=vmax)

def update_display(im_plot, ax, slider, canvas):
    if stack is not None and len(stack) > 0:
        frame0 = stack[0]
        im_plot.set_data(frame0)
        _autoscale_image(im_plot, frame0)
        ax.set_title(f"Frame 1 / {len(stack)}")
        slider.config(to=len(stack) - 1)
        slider.set(0)
        canvas.draw_idle()

from numba import njit, prange

# ---------------------------
# Helper functions (Numba)
# ---------------------------
@njit
def average_numba(c, bins):
    out = np.empty(len(bins), dtype=np.float32)
    out[0] = np.mean(c[:bins[0]])
    for i in range(1, len(bins)):
        out[i] = np.mean(c[bins[i-1]:bins[i]])
    return out

@njit
def smooth_numba(c):
    out = c.copy()
    if len(out) > 1:
        out[0] = out[1]
    for i in range(1, len(out)):
        out[i] = 0.3*out[i] + 0.7*out[i-1]
    for i in range(len(out)-2, -1, -1):
        out[i] = 0.3*out[i] + 0.7*out[i+1]
    return out

@njit
def decaimiento_numba(c):
    idx = 0
    vmax = c[0]
    for i in range(1, c.shape[0]):
        if c[i] > vmax:
            vmax = c[i]
            idx = i
    return idx, vmax


@njit
def sprite_calculation_numba(vals, coords):
    weights = np.clip(vals, 0, np.inf)
    dxs = coords[:, 0]
    dys = coords[:, 1]

    sum_w = np.sum(weights)
    if sum_w < 1e-10:
        return 0.0, 0.0

    x_mean = np.sum(dxs * weights) / sum_w
    y_mean = np.sum(dys * weights) / sum_w

    dx = dxs - x_mean
    dy = dys - y_mean

    u11 = np.sum(dx * dy * weights)
    u20 = np.sum(dx**2 * weights)
    u02 = np.sum(dy**2 * weights)

    denominator = u20 - u02
    theta = 0.5 * np.arctan2(2*u11, denominator)

    sqrt_term = np.sqrt(max(0.0, 4*u11**2 + (u20-u02)**2))
    lambda_short = (u20+u02)/2 - sqrt_term/2
    lambda_long  = (u20+u02)/2 + sqrt_term/2

    denom = lambda_long + lambda_short
    if denom <= 1e-10:
        A = 0.0
    else:
        A = (lambda_long - lambda_short) / denom
        A = max(0.0, min(A, 1.0))
    return theta, A

# ---------------------------
# Main 2D-pCF function
# ---------------------------
def perform_2Dpcf(stack, frame_time, radius=6, nbins=32):
    global anisotropy_map, direction_map, tau_map

    stack = np.squeeze(stack)
    if stack.ndim != 3:
        raise ValueError(f"Stack debe ser 3D (T, H, W), pero llegó con shape {stack.shape}")

    T, H, W = stack.shape
    T = 2**int(np.log2(T))
    stack = stack[:T]

    # --- bins logarítmicos ---
    bins = np.unique(
        np.logspace(0, np.log2(T // 2), nbins, base=2, endpoint=True).astype(np.int32)
    )
    taus = frame_time * bins

    # --- coordenadas del anillo ---
    coords_list = [(dx, dy) for dx in range(-radius, radius + 1)
                             for dy in range(-radius, radius + 1)
                             if round(np.hypot(dx, dy)) == radius]
    coords = np.array(coords_list, dtype=np.int32)
    n_coords = len(coords)

    # --- FFT precomputada ---
    fft_stack = np.fft.rfft(stack, axis=0)

    # --- mapas resultado ---
    anisotropy_map = np.zeros((H - 2 * radius, W - 2 * radius), dtype=np.float32)
    direction_map  = np.zeros_like(anisotropy_map)
    tau_map        = np.zeros_like(anisotropy_map)

    # --- loop espacial ---
    for y in tqdm.tqdm(range(radius, H - radius), desc="Rows"):
        for x in range(radius, W - radius):

            a_fft = fft_stack[:, y, x]

            correl_amp = np.zeros(n_coords, dtype=np.float32)
            correl_tau = np.zeros(n_coords, dtype=np.float32)

            for k, (dx, dy) in enumerate(coords):
                b_fft = fft_stack[:, y + dy, x + dx]

                # correlación cruzada
                c = np.fft.irfft(a_fft * np.conj(b_fft))
                c = c[:T // 2]
                c /= (a_fft[0].real * b_fft[0].real / T)
                c -= 1.0

                # binning + smoothing
                c_bin    = average_numba(c, bins)
                c_smooth = smooth_numba(c_bin)

                # pico de correlación
                idx, vmax = decaimiento_numba(c_smooth)
                correl_amp[k] = vmax
                correl_tau[k] = taus[idx]

            # --- sprite ---
            #tau0 = np.min(correl_tau)
            #weights = correl_tau / tau0
            theta, A = sprite_calculation_numba(correl_amp, coords)
            anisotropy_map[y - radius, x - radius] = A
            direction_map[y - radius, x - radius]  = theta

            if correl_amp.sum() > 0:
                tau_map[y - radius, x - radius] = np.sum(correl_tau * correl_amp) / np.sum(correl_amp)

    return anisotropy_map, direction_map

def degree_to_rad(direction):
    angles_rad = direction
    # Calculate arrow components
    u = np.cos(angles_rad)  # x-component
    v = np.sin(angles_rad)  # y-component
    return u, v
import tifffile as tiff

def graph_direction(a,tita,th=None,intensity=None):
    u, v = degree_to_rad(tita)
    
    if th is None:
        th = np.average(a)
        mask = a > th
    else:
        mask = a > th
    
        
    plt.figure(figsize=(8, 8))

    # Use the mask to filter the arrow locations
    y, x = np.indices(a.shape)
    if intensity:        
         i = np.average(tiff.imread(intensity), axis=0)
         img2 = plt.imshow(i, cmap='Greens', origin='upper')
         #img2 = plt.imshow(yfp, cmap='Greens')
         cbar2 = plt.colorbar(img2)
         cbar2.set_label('Intensity Scale')  # Label for the color bar
    plt.quiver(x[mask], y[mask], u[mask], v[mask], angles='xy', scale_units='xy', scale=0.3, color='grey')
    
    
    plt.xlim(-1, a.shape[1])
    plt.xticks([0,32,64,96,128])
    plt.yticks([0,32,64,96,128])
    plt.ylim(-1, a.shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

global anisotropy_map, direction_map
anisotropy_map = None
direction_map = None

import threading
  
from skimage.measure import profile_line
def apply_2d_pcf(entries, parent_window):
    global stack, anisotropy_map, direction_map

    computing_label = tk.Label(
        parent_window,
        text="Computing...",
        fg="red",
        font=("Arial", 14)
    )
    computing_label.pack(side="bottom", pady=10)

    def run_pcf():
        data = get_table_data()

        first = int(data.get("First frame", 0))
        last = int(data.get("Last frame", len(stack)))
        frame_time = float(data.get("Frame time (s)", 1.0))
        radius = int(data.get("dr (px)", 3))
        binning = int(data.get("binning", 16))

        A, D = perform_2Dpcf(
            stack[first:last],
            frame_time,
            radius,
            binning
        )

        def on_done():
            global anisotropy_map, direction_map
            anisotropy_map = A
            direction_map = D

            computing_label.destroy()

            show_2dpcf_results(
                stack[first:last],
                anisotropy_map,
                direction_map
            )

        parent_window.after(0, on_done)

    parent_window.after(
        50,
        lambda: threading.Thread(target=run_pcf, daemon=True).start()
    )





from matplotlib.ticker import FuncFormatter

def open_A_profile_window(win, original_anisotropy):
    prof_win = tk.Toplevel(win)
    prof_win.title("Anisotropy Profiles")
    prof_win.geometry("600x600")

    fig_p, ax_p = plt.subplots(figsize=(5, 5))
    im = ax_p.imshow(original_anisotropy, cmap='viridis', origin='upper')
    ax_p.set_title("Click two points to extract A profile")
    fig_p.colorbar(im, ax=ax_p, label="Anisotropy (A)")

    canvas_p = FigureCanvasTkAgg(fig_p, master=prof_win)
    canvas_p.get_tk_widget().pack(fill="both", expand=True)

    # Store clicks and artists
    clicks = []
    point_artists = []
    line_artist = None

    def clear_selection():
        nonlocal line_artist
        for p in point_artists:
            p.remove()
        point_artists.clear()

        if line_artist is not None:
            line_artist.remove()
            line_artist = None

        clicks.clear()
        canvas_p.draw_idle()
    prof_plot_win = None
    fig_prof = None
    ax_prof = None
    canvas_prof = None

    def on_click(event):
        nonlocal line_artist
        nonlocal prof_plot_win, fig_prof, ax_prof, canvas_prof
        if event.button != 1:
            return
        if event.inaxes != ax_p:
            return
    
        if len(clicks) == 2:
            clear_selection()

        r, c = event.ydata, event.xdata
        clicks.append((r, c))

        p, = ax_p.plot(c, r, 'ro')
        point_artists.append(p)
        canvas_p.draw_idle()


        if len(clicks) == 2:
            (r0, c0), (r1, c1) = clicks
            line_artist, = ax_p.plot([c0, c1], [r0, r1], 'r-')
            canvas_p.draw_idle()

            # Extract profile
            profile = profile_line(
                original_anisotropy,
                (r0, c0),
                (r1, c1),
                mode='constant',
                cval=np.nan
                )

            if prof_plot_win is None or not prof_plot_win.winfo_exists():
                prof_plot_win = tk.Toplevel(prof_win)
                prof_plot_win.title("A Profile")

                fig_prof, ax_prof = plt.subplots(figsize=(6, 4))
                canvas_prof = FigureCanvasTkAgg(fig_prof, master=prof_plot_win)
                canvas_prof.get_tk_widget().pack(fill="both", expand=True)
            else:
                ax_prof.clear()

            # Plot profile
            dist = np.arange(len(profile))
            ax_prof.plot(dist, profile, '-k')
            ax_prof.set_xlabel("Distance (pixels)")
            ax_prof.set_ylabel("Anisotropy (A)")
            ax_prof.set_title("Anisotropy Profile")
            ax_prof.grid(True)

            canvas_prof.draw_idle()


    fig_p.canvas.mpl_connect('button_press_event', on_click)
    fig_p.tight_layout()
    canvas_p.draw_idle()
    plt.close()

def show_2dpcf_results(stack_cut, anisotropy_map, direction_map):
    """
    Display 2D-pCF results in a Tk window with an interactive threshold slider.
    """
    win = tk.Toplevel()
    win.title("2D pCF Results")
    win.geometry("1200x950")
    win.configure(bg="#f4f4f4")

    # Keep originals
    original_anisotropy = anisotropy_map.copy()
    original_direction = direction_map.copy()

    # =======================
    # Controls frame (top)
    # =======================
    control_frame = tk.Frame(win, bg="#f4f4f4")
    control_frame.pack(fill="x", padx=10, pady=5)

    tk.Label(control_frame, text="Anisotropy Threshold:", bg="#f4f4f4").pack(side="left", padx=5)

    # Slider range
    min_val = float(np.nanmin(original_anisotropy))
    max_val = float(np.nanmax(original_anisotropy))
    threshold_var = tk.DoubleVar(value=min_val)

    # =======================
    # Figure and axes
    # =======================
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # 1) Mean frame
    try:
        mean_frame = np.average(stack_cut, axis=0)
    except Exception:
        mean_frame = np.zeros_like(original_anisotropy)

    im1 = ax1.imshow(mean_frame, cmap='gray', origin='upper')
    ax1.set_title("Average Intensity")
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # 2) Histogram of anisotropy
    flat_vals = original_anisotropy.flatten()
    flat_vals = flat_vals[np.isfinite(flat_vals)]
    counts, bin_edges, bars = ax2.hist(flat_vals, bins=50, color="steelblue", edgecolor='none')
    ax2.set_title("Anisotropy Histogram")
    ax2.set_yscale('log')

    thresh_line = ax2.axvline(threshold_var.get(), color='red', linestyle='--', linewidth=2)

    hist_bars = bars
    hist_bins_left = bin_edges[:-1]
    hist_bins_right = bin_edges[1:]

    # 3) Anisotropy map
    ma_anis = np.ma.array(original_anisotropy, mask=np.zeros_like(original_anisotropy, dtype=bool))
    cmap_a = plt.cm.get_cmap('viridis').copy()
    cmap_a.set_bad('black')
    im3 = ax3.imshow(ma_anis, cmap=cmap_a, origin='upper')
    ax3.set_title("Anisotropy Map (A)")
    cb3 = fig.colorbar(im3, ax=ax3)
    cb3.ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # 4) Direction map
    ma_dir = np.ma.array(original_direction, mask=np.zeros_like(original_direction, dtype=bool))
    cmap_d = plt.cm.get_cmap('hsv').copy()
    cmap_d.set_bad('black')
    im4 = ax4.imshow(ma_dir, cmap=cmap_d, origin='upper')
    ax4.set_title("Anisotropy Direction")

    def rad_to_deg(x, pos):
        return f"{np.degrees(x):.0f}°"

    cb4 = fig.colorbar(im4, ax=ax4, format=FuncFormatter(rad_to_deg))
    cb4.ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    cb4.set_label("Direction (degrees)")

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw_idle()

    # =======================
    # Info label
    # =======================
    frac_label = tk.Label(control_frame, text="", bg="#f4f4f4")
    frac_label.pack(side="left", padx=12)

    # =======================
    # Update function
    # =======================
    def update_maps(val=None):
        thresh = threshold_var.get()
        keep_mask = original_anisotropy >= thresh

        im3.set_data(np.ma.array(original_anisotropy, mask=~keep_mask))
        cb3.update_normal(im3)

        im4.set_data(np.ma.array(original_direction, mask=~keep_mask))
        cb4.update_normal(im4)

        for bar, left, right in zip(hist_bars, hist_bins_left, hist_bins_right):
            center = (left + right) / 2
            if center >= thresh:
                bar.set_facecolor("orange")
            else:
                bar.set_facecolor("lightgray")

        thresh_line.set_xdata([thresh, thresh])

        total = original_anisotropy.size
        above = np.count_nonzero(keep_mask)
        frac_label.config(text=f"Above threshold: {above}/{total} ({100*above/total:.1f}%)")

        canvas.draw_idle()

    # Slider
    threshold_slider = tk.Scale(
        control_frame,
        from_=min_val, to=max_val,
        orient="horizontal",
        resolution=(max_val - min_val) / 1000 if max_val > min_val else 1e-6,
        variable=threshold_var,
        length=420,
        command=update_maps
    )
    threshold_slider.pack(side="left", padx=5)

    update_maps()
    plt.close()

    # =======================
    # NEW: Save maps button
    # =======================
    def save_maps():
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save 2D-pCF maps"
        )
        if not path:
            return

        base = path.replace(".csv", "")
        np.savetxt(base + "_anisotropy.csv", original_anisotropy, delimiter=",")
        np.savetxt(base + "_direction.csv", original_direction, delimiter=",")

    tk.Button(control_frame, text="Save maps", command=save_maps).pack(side="right", padx=5)

    # A Profiles button (ya existente)
    tk.Button(
        control_frame,
        text="A Profiles",
        command=lambda: open_A_profile_window(win, original_anisotropy)
    ).pack(side="right", padx=5)

    return win


def apply_2d_pcf_core(table_data):
    """
    Core de cálculo 2D-pCF.
    NO toca Tkinter.
    NO crea threads.
    """

    global anisotropy_map, direction_map, stack

    first = int(table_data.get("First frame", 0))
    last  = int(table_data.get("Last frame", len(stack)))

    frame_time = float(table_data.get("Frame time (s)", 1.0))
    radius     = int(table_data.get("dr (px)", 3))
    binning    = int(table_data.get("binning", 16))

    A, D = perform_2Dpcf(
        stack[first:last],
        frame_time,
        radius,
        binning
    )

    anisotropy_map = A
    direction_map  = D
    return first, last


def abrir_2d_pcf_ventana():
    first_last = None

    def update_plot(val=None):
        if 'stack' not in globals() or stack is None or len(stack) == 0:
            return
        idx = slider_var.get()
        frame = stack[idx]
        im_plot.set_data(frame)
        _autoscale_image(im_plot, frame)
        ax.set_title(f"Frame {idx+1} / {len(stack)}")
        canvas.draw_idle()

    # -------------------------------
    # Main window
    # -------------------------------
    pcf2d_window = tk.Toplevel()
    pcf2d_window.title("2D pCF Window")
    pcf2d_window.geometry("1200x850")
    pcf2d_window.configure(bg="#f4f4f4")
    
    # -------------------------------
    # TOP BAR: Buttons + Options
    # -------------------------------
    top_frame = tk.Frame(pcf2d_window, bg="#f4f4f4")
    top_frame.pack(side="top", fill="x", padx=10, pady=(10, 5))

    button_style = {
        'bg': '#e0e0e0', 'fg': 'black', 'padx': 10, 'pady': 5,
        'relief': 'raised', 'borderwidth': 1,
        'activebackground': '#c8c8c8', 'activeforeground': 'black'
    }

    load_button = tk.Button(
        top_frame,
        text="Load File",
        command=lambda: load_images_and_update(
            pcf2d_window, top_frame, im_plot, ax, slider, canvas,tipo='image'
        ),
        **button_style
    )
    load_button.grid(row=0, column=0, padx=6, pady=5)

    hv_button = tk.Button(top_frame, text="Intensity vector",
                          command=lambda: open_hv_window(pcf2d_window), **button_style)
    hv_button.grid(row=0, column=1, padx=6, pady=5)

    bleaching_mode = tk.StringVar(value="exp")
    bleach_frame = tk.Frame(top_frame, bg="#f4f4f4")
    bleach_frame.grid(row=0, column=2, padx=20)
    tk.Label(bleach_frame, text="Bleaching type:", bg="#f4f4f4").pack(side="left", padx=(0, 6))
    bleaching_menu = tk.OptionMenu(bleach_frame, bleaching_mode, "exp", "linear")
    bleaching_menu.configure(bg="#e0e0e0")
    bleaching_menu.pack(side="left")
    bleach_button = tk.Button(top_frame, text="Correct Bleaching",
                              command=lambda: correct_bleaching_images(correction=bleaching_mode.get()),
                              **button_style)
    bleach_button.grid(row=0, column=3, padx=6, pady=5)

    # -------------------------------
    # PARAMETERS PANEL
    # -------------------------------
    params_frame = tk.LabelFrame(pcf2d_window, text=" 2D pCF Parameters ",
                                 bg="#ffffff", fg="#333", padx=10, pady=10)
    params_frame.pack(fill="x", padx=15, pady=(5, 10))

    global labels, entries
    labels = ["Frame time (s)", "dr (px)", "binning", "First frame", "Last frame"]
    entries = []
    
    for i, label in enumerate(labels):
        tk.Label(params_frame, text=label, bg="#ffffff").grid(row=0, column=i, padx=5, pady=2)
        e = tk.Entry(params_frame, width=10)
        e.grid(row=1, column=i, padx=5, pady=2)
        entries.append(e)
    status_var = tk.StringVar(value="")
    status_label = tk.Label(
        params_frame,
        textvariable=status_var,
        fg="#555",
        bg="#ffffff"
        )
    status_label.grid(row=2, column=0, columnspan=6, pady=(5, 0))
    
    def run_2d_pcf_threaded():
        nonlocal first_last
        table_data = get_table_data()
        apply_button.config(
            text="Running...",
            state="disabled",
            bg="#b0b0b0"
            )
        status_var.set("Computing 2D pCF… please wait")
        pcf2d_window.config(cursor="watch")
        pcf2d_window.update_idletasks()

        def task():
            nonlocal first_last
            try:
                first_last = apply_2d_pcf_core(table_data)

            finally:
                pcf2d_window.after(0, finish)


        threading.Thread(target=task, daemon=True).start()
        

    def finish():
        apply_button.config(
            text="Run 2D pCF",
            state="normal",
            bg="#5e8fd1"
            )
        status_var.set("Done")
        pcf2d_window.config(cursor="")
        
        if first_last is not None:
            first, last = first_last
            show_2dpcf_results(
            stack[first:last],
            anisotropy_map,
            direction_map
            )
    apply_button = tk.Button(
    params_frame,
    text="Run 2D pCF",
    command=run_2d_pcf_threaded,
    bg="#5e8fd1", fg="white", padx=12, pady=6
)

    apply_button.grid(row=1, column=len(labels), padx=12)

    # -------------------------------
    # MAIN PLOT + SLIDER AREA
    # -------------------------------
    plot_frame = tk.Frame(pcf2d_window, bg="#f4f4f4")
    plot_frame.pack(fill="both", expand=True, padx=15, pady=(0, 5))

    # --- Slider container (top) ---
    slider_var = tk.IntVar(value=0)
    slider_container = tk.Frame(plot_frame, bg="#f4f4f4")
    slider_container.pack(side="top", fill="x", pady=5)
    slider = tk.Scale(
        slider_container,
        from_=0,
        to=0,  # initial dummy range
        orient="horizontal",
        variable=slider_var,
        command=update_plot,
        length=980,
        bg="#ffffff",
        troughcolor="#dcdcdc",
        sliderlength=40,
        highlightthickness=0
    )
    slider.pack(fill="x", padx=5)

    # --- Canvas container (below slider) ---
    canvas_container = tk.Frame(plot_frame, bg="#f4f4f4")
    canvas_container.pack(side="top", fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im_plot = ax.imshow(np.zeros((10, 10)), cmap="viridis")
    ax.axis("off")
    plt.close()

    canvas = FigureCanvasTkAgg(fig, master=canvas_container)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # -------------------------------
    # Function to update slider after loading images
    # -------------------------------
    def update_slider(n_frames):
        slider.config(to=max(n_frames-1, 0))
        slider_var.set(0)

    # Attach function to window so load_images_and_update can call it
    pcf2d_window.update_slider = update_slider



# Open pCF window when button is clicked in CASPY window
def abrir_pcf_ventana():
    global image_frame, cmap_var, working_label, result_label
    cmap_var = tk.StringVar(value='viridis') #default
    global table_frame, dwell_time, labels, entries, checkbutton_vars
    button_style = {
        'bg': 'lightgrey',
        'fg': 'black',
        'padx': 12,
        'pady': 6,
        'relief': 'flat',
        'borderwidth': 2,
        'highlightbackground': '#d3d3d3',
        'highlightcolor': '#d3d3d3',
        'activebackground': '#5a5a5a',
        'activeforeground': 'white'
    }
    fondo = 'white'
    # Create a new Toplevel window for the pCF window
    pcf_window = tk.Toplevel()
    pcf_window.title("pCF Window")  # Title for the new pCF window
    pcf_window.geometry("1000x1000")  # Adjust size as needed

    # Set up the pCF-specific layout
    pcf_window.grid_rowconfigure(0, weight=0)
    pcf_window.grid_rowconfigure(1, weight=1)   # Image frame takes more space
    pcf_window.grid_rowconfigure(2, weight=0)   # Bottom controls take less space
    pcf_window.grid_rowconfigure(3, weight=0)   # Table area smaller
    pcf_window.grid_columnconfigure(0, weight=1)

    # Controls frame for navigation and zoom buttons
    control_frame = tk.Frame(pcf_window, bg=fondo)
    control_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=8)

    # Nav frame inside control_frame
    nav_frame = tk.Frame(control_frame, bg=fondo)
    nav_frame.grid(row=0, column=0, sticky='w')

    h_lines_button = tk.Button(nav_frame, text="H-lines", 
                               command=lambda: open_hlines_window(pcf_window), **button_style)
    h_lines_button.pack(side=tk.LEFT, padx=5, pady=6)

    hv_button = tk.Button(nav_frame, text="V-lines", command=lambda: open_hugevector_window(pcf_window), **button_style)
    hv_button.pack(side=tk.LEFT, padx=6, pady=6)

    bleaching_mode = tk.StringVar(value='exp')  # default: exponential

    # Dropdown to select correction type
    bleaching_menu = tk.OptionMenu(nav_frame, bleaching_mode, 'exp', 'linear')
    bleaching_menu.config(**button_style)  # optional for consistent styling
    bleaching_button = tk.Button(nav_frame, text="Bleaching correction",
                             command=lambda: correct_bleaching(correction=bleaching_mode.get()), **button_style)
    bleaching_button.pack(side=tk.LEFT, padx=7, pady=6)
    bleaching_menu.pack(side=tk.LEFT, padx=8, pady=6)
    
    detrend_button = tk.Button(nav_frame, text="Detrend",
                             command=lambda: detrend_window(pcf_window), **button_style)
    detrend_button.pack(side=tk.LEFT, padx=9, pady=6)
    

    zoom_in_img_raw = Image.open(resource_path("zoomin.png"))
    zoom_out_img_raw = Image.open(resource_path("zoomout.png"))

    zoom_in_img_resized = zoom_in_img_raw.resize((30, 30), Image.Resampling.LANCZOS)
    zoom_out_img_resized = zoom_out_img_raw.resize((30, 30), Image.Resampling.LANCZOS)

    # Convert to PhotoImage
    zoom_in_img = ImageTk.PhotoImage(zoom_in_img_resized)
    zoom_out_img = ImageTk.PhotoImage(zoom_out_img_resized)

    # Zoom frame inside control_frame
    zoom_frame = tk.Frame(control_frame, bg=fondo)
    zoom_frame.grid(row=0, column=1, sticky='w')

    zoom_in_button = tk.Button(zoom_frame, image=zoom_in_img, command=zoom_in, **button_style)
    zoom_in_button.image = zoom_in_img  # prevent garbage collection
    zoom_in_button.pack(side=tk.LEFT, padx=5, pady=6)

    zoom_out_button = tk.Button(zoom_frame, image=zoom_out_img, command=zoom_out, **button_style)
    zoom_out_button.image = zoom_out_img  # prevent garbage collection
    zoom_out_button.pack(side=tk.LEFT, padx=5, pady=6)

    # Color map dropdown inside control_frame
    cmap_var = tk.StringVar(value='viridis')
    cmap_dropdown = ttk.Combobox(control_frame, textvariable=cmap_var, state='readonly')
    cmap_dropdown['values'] = ['viridis', 'plasma', 'cividis', 'jet', 'grey']
    cmap_dropdown.bind("<<ComboboxSelected>>", on_cmap_change)
    cmap_dropdown.grid(row=0, column=2, padx=5, pady=6)

    # Kimogram label inside control_frame
    toggle_button = tk.Button(control_frame, text="Toggle Kimogram", command=toggle_kimogram, **button_style)
    toggle_button.grid(row=0, column=3, padx=5, pady=6)

    kimogram_label = tk.Label(control_frame, text=f"Current Kimogram: {current_index+1}")

    save_plot_button = tk.Button(control_frame, text="Save Kimogram Plot",
                             command=lambda: save_plot(fig, figsize=(5, 8)), **button_style)
    save_plot_button.grid(row=0, column=7, padx=5, pady=6)

    help_button = tk.Button(control_frame, text="Help", command=open_help, **button_style)
    help_button.grid(row=0, column=9, padx=5, pady=6)

    # Load buttons inside control_frame, in a new row
    load_buttons_frame = tk.Frame(control_frame, bg=fondo)
    load_buttons_frame.grid(row=1, column=0, columnspan=5, pady=5, sticky="ew")

    load_lines_button = tk.Button(load_buttons_frame, text="Load Lines",
                              command=lambda: load_lines(pcf_window, cmap_var), **button_style)
    load_lines_button.pack(side=tk.LEFT, padx=4)

    load_pcf_files_button = tk.Button(load_buttons_frame, text="Load pCF files",
                                  command=lambda: load_correlation(cmap_var), **button_style)
    load_pcf_files_button.pack(side=tk.LEFT, padx=4)

    # Image frame (main visualization area)
    image_frame = tk.Frame(pcf_window, bg=fondo)
    image_frame.grid(row=1, column=0, sticky="nsew")
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_columnconfigure(0, weight=1)

    # Bottom button frame (more compact)
    table_button_frame = tk.Frame(pcf_window, bg=fondo)
    table_button_frame.grid(row=2, column=0, pady=5, sticky="ew")

    # Apply pCF button
    apply_pCF_button = tk.Button(table_button_frame, text="Apply pCF",
                             command=lambda: pcf_window.after(0, display_plot(pcf_window)), **button_style)
    apply_pCF_button.grid(row=0, column=0, padx=3, pady=3)

    # Apply ccpCF button
    apply_ccpCF_button = tk.Button(table_button_frame, text="Apply ccpCF",
                               command=lambda: pcf_window.after(0, display_ccplot), **button_style)
    apply_ccpCF_button.grid(row=0, column=1, padx=3, pady=3)

    # Upload filter files
    global file_list_label
    file_list_label = tk.Label(table_button_frame, text="No files selected")
    file_list_label.grid(row=1, column=2, pady=5)
    filterfiles_button = tk.Button(table_button_frame, text="Spectral filter", command=upload_filter_files, **button_style)
    filterfiles_button.grid(row=0, column=2, padx=3, pady=3)

    # Filter buttons
    apply_fccpCF_button = tk.Button(table_button_frame, text="Apply filtered ccpCF",
                                command=lambda: on_apply_filter('ccpcf', pcf_window), **button_style)
    apply_fccpCF_button.grid(row=0, column=4, padx=3, pady=3)

    apply_fpCF_button = tk.Button(table_button_frame, text="Apply filtered pCF",
                              command=lambda: on_apply_filter('pCF', pcf_window), **button_style)
    apply_fpCF_button.grid(row=0, column=3, padx=3, pady=3)

    table_container = tk.Frame(pcf_window, bg=fondo)
    table_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
    table_container.configure(height=300)

    # Canvas to enable scrolling
    table_canvas = tk.Canvas(table_container, height=300, bg=fondo)
    table_scrollbar = tk.Scrollbar(
    table_container, orient="vertical", command=table_canvas.yview
        )

    table_canvas.configure(yscrollcommand=table_scrollbar.set)

    table_canvas.pack(side="left", fill="both", expand=True)
    table_scrollbar.pack(side="right", fill="y")
    table_frame = tk.Frame(table_canvas, bg=fondo)

    table_canvas.create_window((0, 0), window=table_frame, anchor="nw")

    # Make canvas update scroll region automatically
    table_frame.bind(
        "<Configure>",
        lambda e: table_canvas.configure(scrollregion=table_canvas.bbox("all"))
        )


    labels = ["Line Time (ms)", "First Line", "Last Line", 'Distance (px)', 'H Smoothing (px)', 'V Smoothing (lines)', 'Reverse', 'Normalize']
    entries = []
    checkbutton_vars = {}  # For storing the Checkbutton states

    for row, label in enumerate(labels):
        tk.Label(table_frame, text=label, borderwidth=2, relief="solid").grid(row=row, column=0, padx=5, pady=4, sticky="e")

        if label == 'Reverse' or label == 'Normalize':
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(table_frame, variable=var)
            entries.append(checkbox)
            checkbutton_vars[label] = var
            checkbox.grid(row=row, column=1, padx=5, pady=4, sticky="ew")

        elif label == 'Line time (ms)':
            entry = tk.Entry(table_frame)
            entry.insert(0, '100')  # Example value
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=4, sticky="ew")

        elif label == 'H Smoothing (px)':
            entry = tk.Entry(table_frame)
            entry.insert(0, '4')  # Default value
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=4, sticky="ew")
        elif label == 'V Smoothing (lines)':
            entry = tk.Entry(table_frame)
            entry.insert(0, '10')  # Default value
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=4, sticky="ew")

        else:
            entry = tk.Entry(table_frame)
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=4, sticky="ew")

    # Working and result labels
    working_label = tk.Label(pcf_window, text="Working...", font=("Arial", 20), fg="black")
    result_label = tk.Label(pcf_window, text="Done!", font=("Arial", 20), fg="black")
    result_label.grid(row=4, column=0, pady=6)
    result_label.grid_forget()

# Function to be called when user clicks the pCF button in the main window
def abrir_pcf_ventana_main_window():
    abrir_pcf_ventana()  # Simply call abrir_pcf_ventana to open the window



################################################################################
# Function to start the application
def start_application():
    root = tk.Tk()
    init_app(root)
    root.mainloop()

if __name__ == "__main__":
    start_application()

