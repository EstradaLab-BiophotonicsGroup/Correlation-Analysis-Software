#%%
from globales import *

from MATE_BA import run_MATE_BA

# import tkinter as tk
# import math
# from tkinter import filedialog, ttk
# from tkinter import Frame, Label, Toplevel, Entry, IntVar, messagebox
# from PIL import Image, ImageTk, ImageSequence
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.widgets import RectangleSelector
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from scipy.stats import gaussian_kde
# from scipy.ndimage import gaussian_filter
# from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
# from matplotlib import ticker #as mticker
# import matplotlib.colors as colors
# from czifile import imread
# import threading
# import pandas as pd
# from matplotlib.colors import LinearSegmentedColormap
# from tkinter import simpledialog
# from lfdfiles import SimfcsB64 as lfd
# import webbrowser
# import sys
# import numpy as np
# import os
# import tqdm
# import importlib.util
# from tifffile import imread as tif_imread
# import traceback
# import matplotlib.cm as cm


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
    # animar_gif(root, logo_label, frames, frame_index)

    # Create buttons for each window
    # The CASPY window button for pCF

    btn1 = tk.Button(root, text="N&B", command=abrir_NB_ventana, bg="white", height=2, width=20)
    btn1.grid(row=3, column=0, pady=5)  # Changed from pack() to grid()


    btn2 = tk.Button(root, text="MATE-BA", command=lambda: run_MATE_BA(root), bg="white", height=2, width=20)
    btn2.grid(row=4, column=0, pady=5)  # Changed from pack() to grid()



# Animate the GIF
def animar_gif(root, logo_label, frames, frame_index):
    frame = frames[frame_index]
    logo_label.configure(image=frame)
    frame_index = (frame_index + 1) % len(frames)
    root.after(50, animar_gif, root, logo_label, frames, frame_index)

    print(3)


# Create a new window with a message
def nueva_ventana(root, titulo, mensaje):
    ventana = Toplevel(root)
    ventana.title(titulo)
    ventana.geometry("300x200")
    ventana.configure(bg="white")
    tk.Label(ventana, text=mensaje, font=("Arial", 18), bg="white").grid(pady=40)

    print(4)

def get_user_data_dir():
    base = Path.home() / "Documents" / APP_NAME
    base.mkdir(exist_ok=True)
    print(5)
    return base

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    print(6)
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
    print(7)
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
    print(8)

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
    print(9)


def on_cmap_change(event):
    global cmap_var, current_figures, original_kimograms
    print(10)
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

import czifile
import xml.etree.ElementTree as ET
def extract_metadata(file_path, callback):
    global dwell_time, pixels
    print(11)
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
    print(12)
    return [channels[:, i, :] for i in range(num_channels)]

import queue
def ask_for_pixels_threadsafe(parent):
    q = queue.Queue()
    print(13)
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

    print(14)

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



def load_and_display_images(file_paths, parent_window, control_frame, apply_button, mask_button, mask_status):
    global images
    loading_window = show_loading_popup(parent_window)

    print(15)

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
            parent_window.after(0, lambda: open_image_viewer_with_nb_controls(
                all_stacks,parent_window, control_frame,apply_button, mask_button, mask_status))
            parent_window.after(0, loading_window.destroy)
        except Exception as e:
            import traceback
            error_msg = f"Unhandled error:\n{str(e)}\n\n{traceback.format_exc()}"
            parent_window.after(0, loading_window.destroy)
            parent_window.after(0, lambda: show_message("Error", error_msg))

    threading.Thread(target=run, daemon=True).start()
    
def load_images(parent_window, control_frame, apply_button, mask_button, mask_status):
    file_paths = filedialog.askopenfilenames(
        filetypes=[("image files", "*.tiff;*.tif;*.czi;*.b64")]
    )

    if file_paths:
        load_and_display_images(file_paths, parent_window,control_frame, apply_button, mask_button,mask_status)
    else:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")
    
    print(16)


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
    pcf_window = tk.Toplevel()
    pcf_window.title("pCF Window")  # Title for the new pCF window
    pcf_window.geometry("1000x1000")  # Adjust size as needed
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
    
    
    load_button = tk.Button(
    control_frame,
    text="Load file",
    command=lambda: load_images(
        image_frame,
        control_frame,
        apply_button,
        mask_button,
        mask_status
    ),
    **button_style
)
    load_button.pack(side=tk.LEFT, padx=5, pady=10)
    mask_status = tk.Label(control_frame, text="No mask set", fg="red")
    mask_status.pack(side="right", padx=5)

    apply_button = tk.Button(control_frame, text="Apply N&B")
    apply_button.pack(side="left", padx=5)

    mask_button = tk.Button(control_frame, text="Set mask")
    mask_button.pack(side="left", padx=5)
    image_frame = tk.Frame(nb_window, bg=fondo)
    image_frame.grid(row=1, column=0, sticky="nsew")
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_columnconfigure(0, weight=1)


def open_image_viewer_with_nb_controls(stack, parent_frame, control_frame, apply_button, mask_button, mask_status):
    print(18)
    global mask
    mask = None

    # Limpiar el frame excepto los controles
    for widget in parent_frame.winfo_children():
        if widget not in (control_frame,):
            widget.destroy()

    # --- NUEVA ETIQUETA PARA COORDENADAS ---
    # Se coloca arriba del gráfico para que sea visible
    coord_label = tk.Label(parent_frame, text="Mueve el mouse sobre la imagen", 
                          font=("Consolas", 10, "bold"), bg="white", fg="blue")
    coord_label.pack(side="top", anchor="e", padx=20, pady=2)

    # =======================
    # Image viewer
    # =======================
    plot_frame = tk.Frame(parent_frame, height=450)
    plot_frame.pack(side="top", fill="x", padx=10, pady=(10, 5))
    plot_frame.pack_propagate(False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im_plot = ax.imshow(stack[0], cmap="viridis")
    ax.axis("on") # Cambiado a "on" para ver los ejes (opcional)
    ax.set_title(f"Frame 0 / {len(stack)-1}")
    
    
    # CRITICO: Guardar el objeto canvas en una variable separada
    canvas_obj = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas_obj.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    current_frame = tk.IntVar(value=0)

    # --- FUNCIÓN DE EVENTO PARA EL MOUSE ---
    def on_mouse_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            # Matplotlib usa (x,y), pero numpy usa [y,x]
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            
            # Validar límites de la imagen
            if 0 <= iy < stack.shape[1] and 0 <= ix < stack.shape[2]:
                idx = current_frame.get()

                val = stack[idx, iy, ix]
                coord_label.config(text=f"Frame {idx} | x={ix}, y={iy} | Intensidad={val:.2f}")
        else:
            coord_label.config(text="Fuera de imagen")

    # CONECTAR EL EVENTO
    canvas_obj.mpl_connect('motion_notify_event', on_mouse_move)

    def update_plot(*args):
        try:
            idx = current_frame.get()
            print(idx)
            if idx >= len(stack):
                idx = len(stack) - 1
            if 0 <= idx < len(stack):
                im_plot.set_data(stack[idx])
                # Autoscale para mejorar visualización
                # vmin, vmax = np.percentile(stack[idx], [1, 99])
                vmin, vmax = min(stack[idx].ravel()), max(stack[idx].ravel())
                
                im_plot.set_clim(vmin, vmax)
                ax.set_title(f"Frame {idx} / {len(stack)-1}")
                canvas_obj.draw_idle()
            print(19)
        except tk.TclError:
            pass # Maneja casos donde el input del spinbox sea temporalmente inválido
    
    
    # slider = tk.Scale(
    #     parent_frame,
    #     from_=1,
    #     to=len(stack)-1,
    #     orient="horizontal",
    #     variable=current_frame,
    #     command=update_plot,
    # )
    # slider.pack(fill="x", padx=100)

 # --- FRAME PARA SPINBOX Y TOTAL ---
    navigation_frame = tk.Frame(parent_frame, bg="white")
    navigation_frame.pack(side="top", fill="x", padx=100, pady=5)

    tk.Label(navigation_frame, text="Frame: ", bg="white").pack(side="left")

    # Spinbox configurado de 0 a Total-1
    frame_spinbox = tk.Spinbox(
        navigation_frame,
        from_=0,
        to=len(stack) - 1,
        textvariable=current_frame,
        command=update_plot, # Se ejecuta al usar las flechas
        width=10
    )
    frame_spinbox.pack(side="left", padx=5)

    # Etiqueta de total
    tk.Label(navigation_frame, text=f"of {len(stack) - 1}", 
             bg="white", font=("Arial", 9, "italic")).pack(side="left")

    current_frame.trace_add("write", update_plot)

    # =======================
    # Controls (se mantiene igual el resto...)
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
        print(20)
        tk.Label(filter_frame, text=label).grid(row=row, column=0, sticky="w")
        e_min = tk.Entry(filter_frame, width=8)
        e_max = tk.Entry(filter_frame, width=8)
        e_min.grid(row=row, column=1)
        e_max.grid(row=row, column=2)
        return e_min, e_max

    entry_I_min, entry_I_max = make_filter_row(1, "Intensity (I)")
    entry_N_min, entry_N_max = make_filter_row(2, "Number (N)")
    entry_B_min, entry_B_max = make_filter_row(3, "Brightness (B)")

    def upload_mask():
        global mask
        files = filedialog.askopenfilenames(filetypes=[("Matrix", "*.txt")])
        if files:
            mask_data = np.loadtxt(files[0])
            mask = mask_data.astype(bool)
            mask_status.config(text="Mask set", fg="green")
        print(21)

    def save_nb_results(N, B):
        pixels = int(np.sqrt(len(N)))
        N_mat = N.reshape(pixels, pixels)
        B_mat = B.reshape(pixels, pixels)
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            np.savetxt(path.replace(".csv", "_N.csv"), N_mat, delimiter=",")
            np.savetxt(path.replace(".csv", "_B.csv"), B_mat, delimiter=",")
        print(22)
  
    # Aquí definiremos la función apply_nb dentro para que tenga acceso a los entry
    def apply_nb_internal():
        # Ver la implementación de apply_nb abajo e integrarla aquí o llamarla
        apply_nb(stack, entry_start, entry_end, entry_s, entry_offset, entry_sigma, 
                 entry_I_min, entry_I_max, entry_N_min, entry_N_max, entry_B_min, entry_B_max, 
                 parent_frame, save_nb_results, upload_mask)

    apply_button.config(command=apply_nb_internal)
    mask_button.config(command=upload_mask)


def apply_nb(stack, entry_start, entry_end, entry_s, entry_offset, entry_sigma, 
             entry_I_min, entry_I_max, entry_N_min, entry_N_max, entry_B_min, entry_B_max, 
             parent_frame, save_nb_results, upload_mask):

    print(23)
    global mask

    start = int(entry_start.get() or 0)
    end = int(entry_end.get() or len(stack))
    S = float(entry_s.get())
    offset = float(entry_offset.get())
    sigma = float(entry_sigma.get())

    data = stack[start:end]

    if mask is not None:
        mask_reshaped = mask.reshape(data.shape[1], data.shape[2])
        data = data * mask_reshaped
    
    I = np.mean(data, axis=0).ravel()
    VAR = np.nanvar(data, axis=0, ddof=0).ravel()

    B = (VAR - sigma**2) / (S * (I - offset))
    N = (I - offset)**2 / (VAR - sigma**2)

    # for k in range(0,len(I)):
    #     if math.isnan(I[k]) or math.isnan(VAR[k]):
    #         B[k]= np.nan  
    #         N[k]= np.nan



    #     if I[k] == 0 or I[k] - offset==0 or VAR[k] == sigma**2 or VAR[k] - sigma**2-S*(I[k]-offset)==0:
    #         B[k]= 1  
    #         N[k]= 0


    #     else:
    #         B[k] = ((VAR[k] - sigma**2))/(S*(I[k] - offset))  ## B is define as Eq. 6 divide by S factor in DOI: 10.1002/jemt.20526
    #         N[k] = ((I[k]-offset)**2)/(VAR[k] - sigma**2)

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

    # =======================
    # Results window
    # =======================
    shape = stack.shape[1:]
    win = tk.Toplevel(parent_frame)
    win.title("N&B Results")
    win.configure(bg='white')

    # --- ETIQUETA DE COORDENADAS PARA RESULTADOS ---
    res_info_label = tk.Label(win, text="Mueve el mouse sobre los mapas para ver valores", 
                             font=("Arial", 11, "bold"), bg="white", fg="darkblue")
    res_info_label.pack(side="top", pady=5)

    plots = tk.Frame(win, background='white')
    plots.pack(side="top", fill="both", expand=True)

    hists = tk.Frame(win)
    hists.pack(side="bottom", fill="both", expand=True)

    # Convertir a 2D para indexar fácilmente
    B_2D = B.reshape(shape)
    N_2D = N.reshape(shape)

    # ---- Maps ----
    fig_B, ax_B = plt.subplots(figsize=(4, 4))
    fig_N, ax_N = plt.subplots(figsize=(4, 4))

    ax_B.imshow(B_2D, cmap="viridis")
    ax_B.set_title('Brightness (B)')
    ax_N.imshow(N_2D, cmap="viridis")
    ax_N.set_title('Number (N)')

    # GUARDAR CANVAS COMO OBJETOS
    canvas_B = FigureCanvasTkAgg(fig_B, master=plots)
    canvas_B.get_tk_widget().pack(side="left", expand=True, fill="both")
    
    canvas_N = FigureCanvasTkAgg(fig_N, master=plots)
    canvas_N.get_tk_widget().pack(side="left", expand=True, fill="both")

    # --- FUNCIÓN DE RASTREO PARA RESULTADOS ---
    def on_move_results(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        # Validar límites
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            if event.inaxes == ax_B:
                val = B_2D[iy, ix]
                res_info_label.config(text=f"Mapa B | x:{ix}, y:{iy} | Valor:{val:.4f}")
            elif event.inaxes == ax_N:
                val = N_2D[iy, ix]
                res_info_label.config(text=f"Mapa N | x:{ix}, y:{iy} | Valor:{val:.4f}")

    # CONECTAR AMBOS CANVAS
    canvas_B.mpl_connect('motion_notify_event', on_move_results)
    canvas_N.mpl_connect('motion_notify_event', on_move_results)
 
 # 2. Función de actualización corregida
    def on_move_results(event):
        if event.inaxes is None:
            return
        
        # Detectar en qué gráfico estamos
        if event.inaxes == ax_B:
            name = "B"
            val = B_2D[int(round(event.ydata)), int(round(event.xdata))]
        elif event.inaxes == ax_N:
            name = "N"
            val = N_2D[int(round(event.ydata)), int(round(event.xdata))]
        else:
            return

        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        res_info_label.config(text=f"{name} Map  | (x={ix}, y={iy}) | Value={val:.4f}")

    # 3. Conectar los eventos a los objetos canvas guardados
    canvas_B.mpl_connect('motion_notify_event', on_move_results)
    canvas_N.mpl_connect('motion_notify_event', on_move_results)

    
    print(25)

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

        print(26)

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

            print(27)

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

print(28)

# apply_button.config(command=apply_nb)
# mask_button.config(command=upload_mask)

stack = None  # global to hold the loaded image stack


def load_images_and_update(parent_window, control_frame, im_plot, ax, slider, canvas,tipo):

    print(29)

    global stack
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image files", "*.tiff;*.tif;*.czi;*.b64")]
    )
    if not file_paths:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")
        return
    loading_window = show_loading_popup(parent_window)

    def run():
        
        print(30)

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

    print(31)

    # Robustly set color limits for the current frame
    vmin = float(np.nanmin(frame))
    vmax = float(np.nanmax(frame))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallback to a tiny range to avoid identical vmin/vmax
        vmax = vmin + 1e-12
    im_plot.set_clim(vmin=vmin, vmax=vmax)

def update_display(im_plot, ax, slider, canvas):

    print(32)

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

    print(33)

    out = np.empty(len(bins), dtype=np.float32)
    out[0] = np.mean(c[:bins[0]])
    for i in range(1, len(bins)):
        out[i] = np.mean(c[bins[i-1]:bins[i]])
    return out

@njit
def smooth_numba(c):

    print(34)

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

    print(35)

    idx = 0
    vmax = c[0]
    for i in range(1, c.shape[0]):
         if c[i] > vmax:
             vmax = c[i]
             idx = i
    return idx, vmax


@njit
def sprite_calculation_numba(vals, coords):

    print(36)

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


################################################################################
# Function to start the application
def start_application():

    print(37)

    root = tk.Tk()
    init_app(root)
    root.mainloop()

if __name__ == "__main__":
    print(38)
    start_application()
