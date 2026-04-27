#%%


from globales import *


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

# def exp_decay(t,A,b,c):
#     return A*np.exp(-t*b)+c

# def linear_decay(t,A,b):
#     return -A*t+b


# def correct_bleaching(correction='exp', downsample_factor=100):
#     global original_kimograms, current_index
#     try:
#         original_kimogram=original_kimograms[current_index]
#     except:
#         print('no current axis')
#         original_kimogram=original_kimograms[0]
#     average = np.mean(original_kimogram, axis=1)
#     n_lines = len(average)
    
#     # Downsample to reduce data size
#     if downsample_factor > 1:
#         n_blocks = n_lines // downsample_factor
#         avg_ds = average[:n_blocks * downsample_factor].reshape(n_blocks, downsample_factor).mean(axis=1)
#         x_ds = np.arange(n_blocks) * downsample_factor
#     else:
#         avg_ds = average
#         x_ds = np.arange(n_lines)
    
#     if correction == 'exp':
#         # Fit exponential to downsampled data
#         p0 = [avg_ds[0], 1e-5, avg_ds[-1]]
#         popt, _ = curve_fit(exp_decay, x_ds, avg_ds, p0=p0, maxfev=10000)
#         A, b, c = popt
        
#         # Evaluate the fitted decay at full resolution
#         x_full = np.arange(n_lines)
#         fitted_decay = exp_decay(x_full, A, b, c)
        
#         # Normalize to 1 at start (optional)
#         fitted_decay_norm = fitted_decay / fitted_decay[0]
#     elif correction=='linear':
#         popt, _ = curve_fit(linear_decay, x_ds, avg_ds, maxfev=10000)
#         A, b = popt
        
#         # Evaluate the fitted decay at full resolution
#         x_full = np.arange(n_lines)
#         fitted_decay = linear_decay(x_full, A, b)
#         fitted_decay_norm = fitted_decay / fitted_decay[0]
#     corrected_kimogram = original_kimogram / fitted_decay_norm[:, np.newaxis]
#     corrected_kimogram = np.clip(corrected_kimogram, 0, None)
#     try:
#             original_kimograms[current_index] = corrected_kimogram
#             show_message('Done', 'The data has been corrected')
#     except:
#             original_kimograms[0] = corrected_kimogram
#             show_message('Done', 'The data has been corrected')
#     return 

# def correct_bleaching_images(correction='exp', downsample_factor=10):
#     global stack

#     # --- If single image, do nothing ---
#     if stack.ndim < 3:
#         show_message('Info', 'Single image detected — bleaching correction skipped.')
#         return

#     n_frames = stack.shape[0]
#     avg_intensity = stack.mean(axis=(1, 2))  # mean intensity per frame

#     # --- Downsample intensity curve ---
#     if downsample_factor > 1 and n_frames > downsample_factor:
#         n_blocks = n_frames // downsample_factor
#         avg_ds = avg_intensity[:n_blocks * downsample_factor].reshape(n_blocks, downsample_factor).mean(axis=1)
#         x_ds = np.arange(n_blocks) * downsample_factor
#     else:
#         avg_ds = avg_intensity
#         x_ds = np.arange(n_frames)

#     # --- Fit decay ---
#     if correction == 'exp':
#         # Exponential model: A * exp(-b*t) + c
#         p0 = [avg_ds[0], 1e-5, avg_ds[-1]]
#         popt, _ = curve_fit(exp_decay, x_ds, avg_ds, p0=p0, maxfev=10000)
#         A, b, c = popt
#         fitted_decay = exp_decay(np.arange(n_frames), A, b, c)
#     elif correction == 'linear':
#         # Linear model: A + B*t
#         popt, _ = curve_fit(linear_decay, x_ds, avg_ds, maxfev=10000)
#         A, B = popt
#         fitted_decay = linear_decay(np.arange(n_frames), A, B)
#     else:
#         show_message('Error', f"Unknown correction type: {correction}")
#         return

#     # --- Normalize decay to 1 at t=0 and apply ---
#     fitted_decay_norm = fitted_decay / fitted_decay[0]
#     fitted_decay_norm = np.clip(fitted_decay_norm, 1e-6, None)

#     corrected_stack = stack / fitted_decay_norm[:, np.newaxis, np.newaxis]
#     corrected_stack = np.clip(corrected_stack, 0, None)
#     stack = corrected_stack
#     show_message('Done', 'Bleaching correction applied.')
    


def abrir_MATE_ventana():
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
    mate_window = tk.Toplevel()
    mate_window.title("MATE Window")  # Title for the new pCF window
    mate_window.geometry("1500x1000")  # Adjust size as needed

    # Set up the pCF-specific layout
    mate_window.grid_rowconfigure(0, weight=0)
    mate_window.grid_rowconfigure(1, weight=1)   # Image frame takes more space
    mate_window.grid_rowconfigure(2, weight=0)   # Bottom controls take less space
    mate_window.grid_rowconfigure(3, weight=0)   # Table area smaller
    mate_window.grid_columnconfigure(0, weight=1)

    # Controls frame for navigation and zoom buttons
    control_frame = tk.Frame(mate_window, bg=fondo)
    control_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=8)

    # Nav frame inside control_frame
    nav_frame = tk.Frame(control_frame, bg=fondo)
    nav_frame.grid(row=0, column=0, sticky='w')

    h_lines_button = tk.Button(nav_frame, text="H-lines", 
                               command=lambda: open_hlines_window(mate_window), **button_style)
    h_lines_button.pack(side=tk.LEFT, padx=5, pady=6)

    hv_button = tk.Button(nav_frame, text="V-lines", command=lambda: open_hugevector_window(mate_window), **button_style)
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
                             command=lambda: detrend_window(mate_window), **button_style)
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
                              command=lambda: load_lines(mate_window, cmap_var), **button_style)
    load_lines_button.pack(side=tk.LEFT, padx=4)

    load_pcf_files_button = tk.Button(
        load_buttons_frame,
        text="Load pCF files",
        command=lambda: load_correlation(cmap_var, image_frame),  # ✅ pass image_frame
        **button_style)

    load_pcf_files_button.pack(side=tk.LEFT, padx=4)

    # Image frame (main visualization area)
    image_frame = tk.Frame(mate_window, bg=fondo)
    image_frame.grid(row=1, column=0, sticky="nsew")
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_columnconfigure(0, weight=1)

    # Bottom button frame (more compact)
    table_button_frame = tk.Frame(mate_window, bg=fondo)
    table_button_frame.grid(row=2, column=0, pady=5, sticky="ew")

    # Apply MATE-BA button
    apply_mate_ba_button = tk.Button(table_button_frame, text="Apply MATE-BA",
                             command=lambda: mate_window.after(0, display_plot(mate_window)), **button_style)
    apply_mate_ba_button.grid(row=0, column=0, padx=3, pady=3)

    # Apply pCF to MATE-BA button
    apply_pcf_of_mate_ba_button = tk.Button(table_button_frame, text="Apply pCF to MATE-BA",
                             command=lambda: mate_window.after(0, display_plot(mate_window)), **button_style)
    apply_pcf_of_mate_ba_button.grid(row=0, column=2, padx=3, pady=3)

    # # Apply ccpCF button
    # apply_ccpCF_button = tk.Button(table_button_frame, text="Apply ccpCF",
    #                            command=lambda: mate_window.after(0, display_ccplot), **button_style)
    # apply_ccpCF_button.grid(row=0, column=1, padx=3, pady=3)

    # Upload filter files
    # global file_list_label
    # file_list_label = tk.Label(table_button_frame, text="No files selected")
    # file_list_label.grid(row=1, column=2, pady=5)
    # filterfiles_button = tk.Button(table_button_frame, text="Spectral filter", command=upload_filter_files, **button_style)
    # filterfiles_button.grid(row=0, column=2, padx=3, pady=3)

    # # Filter buttons
    # apply_fccpCF_button = tk.Button(table_button_frame, text="Apply filtered ccpCF",
    #                             command=lambda: on_apply_filter('ccpcf', mate_window), **button_style)
    # apply_fccpCF_button.grid(row=0, column=4, padx=3, pady=3)

    # apply_fpCF_button = tk.Button(table_button_frame, text="Apply filtered pCF",
    #                           command=lambda: on_apply_filter('pCF', mate_window), **button_style)
    # apply_fpCF_button.grid(row=0, column=3, padx=3, pady=3)

    table_container = tk.Frame(mate_window, bg=fondo)
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
    working_label = tk.Label(mate_window, text="Working...", font=("Arial", 20), fg="green")
    result_label = tk.Label(mate_window, text="Done!", font=("Arial", 20), fg="black")
    result_label.grid(row=4, column=0, pady=6)
    result_label.grid_forget()


##############
def open_profiles_window(mate_window):
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
    profile_window = Toplevel(mate_window)
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

        

def load_correlation(cmap_var, parent_frame):
    global G_to_save, T_to_save
    G_to_save = []
    file_paths = filedialog.askopenfilenames(filetypes=[("Correlation data", "*.txt;*.csv")])
    if not file_paths:
        show_message('Error', "Can't correlate without the data :(")
        return

    file_time = filedialog.askopenfilenames(filetypes=[("Correlation time", "*.txt;*.csv")])
    T_to_save = np.loadtxt(file_time[0])
    if all(file_path.endswith('.txt') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(np.loadtxt(file_path, delimiter=','))
    elif all(file_path.endswith('.csv') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(pd.read_csv(file_path).to_numpy())
            
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
    
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)  # ✅ attach to the frame inside mate_window
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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


def plot_kimograms(kimograms,cmap_var,factor=1, title="Kimogram of intensity"):
    global current_figures
    current_figures = []

    def get_extent(kimogram):
        return [0, kimogram.shape[1], kimogram.shape[0], 0]  # [x_min, x_max, y_min, y_max]

    num_kimograms = len(kimograms)
    font = 16
    if num_kimograms > 1:
        fig, axs = plt.subplots(1, num_kimograms, figsize=(8 * num_kimograms, 20), squeeze=False)
        axs = axs.flatten()
        
        for ax, kimogram in zip(axs, kimograms):
            vmin = np.min(kimogram)
            vmax = np.mean(kimogram) + 2 * np.std(kimogram)
            extent = get_extent(kimogram)
            
            cax = ax.imshow(kimogram, aspect='auto', cmap=my_colors(cmap_var), norm=plt.Normalize(vmin=vmin, vmax=vmax),
        origin='upper', extent=[0, kimogram.shape[1], kimogram.shape[0], 0])
            ax.set_xlabel("Pixel", fontsize=font-2)
            ticks = np.linspace(0,len(kimogram),5, dtype=int)
            ax.set_yticks(ticks,labels=[int(ticks[i]*factor/10000)*10000 for i in range(len(ticks))], fontsize=font-2)
            ax.set_ylabel("Line Number", fontsize=font)
            ax.set_title(f"Kimogram {axs.tolist().index(ax) + 1}", fontsize=font-2)
            cbar = plt.colorbar(cax, ax=ax)
            cbar.set_label('Intensity', fontsize=font)
            cbar.ax.tick_params(labelsize=font-2)
            plt.subplots_adjust(wspace=0.6)

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
        ax.set_title(title)
        plt.colorbar(cax, ax=ax, label='Intensity')

    current_figures.append(fig)
    return fig

# def display_kimograms(kimograms, cmap_var, factor, parent_frame):
#     fig = plot_kimograms(kimograms, cmap_var, factor)
#     fig.colorbars = []

#     canvas = FigureCanvasTkAgg(fig, master=parent_frame)  # ✅ use parent_frame
#     canvas.draw()
#     canvas.get_tk_widget().grid(row=0, column=0)

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


def open_hlines_window(mate_window):
    '''
    this function is used to plot the horizontal profile of the kimogram, i.e. averaged lines.
    This is particularly helpful to identify regions of different intensity. 
    The plot is done in a new window where the user selects wich lines to average. 
    If there are 2 kimograms, use the tgogle function to change which one you want for this window.

    Returns
    -------
    None.

    '''
    hlines_window = Toplevel(mate_window)
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
    vmax = np.percentile(G_log, 90)
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


def open_hlines_window(mate_window):
    '''
    this function is used to plot the horizontal profile of the kimogram, i.e. averaged lines.
    This is particularly helpful to identify regions of different intensity. 
    The plot is done in a new window where the user selects wich lines to average. 
    If there are 2 kimograms, use the tgogle function to change which one you want for this window.

    Returns
    -------
    None.

    '''
    hlines_window = Toplevel(mate_window)
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

def show_loading_popup(mate_window):
    loading_win = tk.Toplevel()
    loading_win.title("Loading")
    loading_win.geometry("200x100")
    loading_win.resizable(False, False)

    label = tk.Label(loading_win, text="Loading files...", font=("Arial", 12))
    label.pack(expand=True, fill='both', padx=20, pady=20)

    # Disable closing the window manually
    loading_win.protocol("WM_DELETE_WINDOW", lambda: None)
    loading_win.transient(mate_window)  # Keep it above main window
    loading_win.grab_set()       # Block other UI interaction

    return loading_win

def load_and_display(file_paths, mate_window,cmap_var):  # Added mate_window parameter
    loading_window = show_loading_popup(mate_window)  # ⬅️ Show loading popup in the pCF window

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
                        mate_window.after(0, lambda: display_kimograms(optimized_kimograms,cmap_var,factor, mate_window))  # Pass mate_window to display_kimograms
                        mate_window.after(0, update_table_with_dwell_time)
                        mate_window.after(0, loading_window.destroy)
                    except Exception as e:
                        mate_window.after(0, loading_window.destroy)
                        mate_window.after(0, lambda: show_message("Error", str(e)))

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
                        kim = read_B64(fp,parent_window=mate_window)
                        print("B64 loaded:", type(kim))
                        original_kimograms.append(kim)
                        mate_window.after(0, update_table_with_dwell_time)
                    except Exception as e:
                        errors.append(f"{fp}: {e}")

            optimized_kimograms = [downsample_kimogram(k) for k in original_kimograms]

            if errors:
                mate_window.after(0, lambda: show_message("Error", "\n".join(errors)))
            global factor
            mate_window.after(0, lambda: display_kimograms(optimized_kimograms,cmap_var,factor))  # Pass mate_window to display_kimograms
            mate_window.after(0, loading_window.destroy)

        except Exception as e:
            import traceback
            error_msg = f"Unhandled error:\n{str(e)}\n\n{traceback.format_exc()}"
            mate_window.after(0, loading_window.destroy)
            mate_window.after(0, lambda: show_message("Error", error_msg))

    threading.Thread(target=run, daemon=True).start()

def resolve_cmap(cmap_input):
    if isinstance(cmap_input, Colormap):
        return cmap_input
    elif isinstance(cmap_input, str):
        return plt.get_cmap(cmap_input)
    elif isinstance(cmap_input, dict):
        return LinearSegmentedColormap('custom', cmap_input)
    else:
        raise ValueError("Unsupported cmap type")
def load_lines(mate_window,cmap_var):  # Added mate_window parameter
    file_paths = filedialog.askopenfilenames(filetypes=[("line files", "*.tiff;*.tif;*.czi;*.b64;*.raw")])
    if file_paths:
        threading.Thread(target=load_and_display, args=(file_paths, mate_window,cmap_var), daemon=True).start()  # Pass mate_window to load_and_display
    else:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")

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

def MATE_BA(Lines, tp='', tr=0, window_points = 100, delta_line=100, window_shift=1, correlated_lines=True):

    '''
    Parameters
    ----------
    Lines : ndarray
        2D array. Each row is a Line
        
    tp : float
        pixel dwell time. Value must be in seconds.
        
    window_points : int
        Number of lines for epsilon calculation.
        By default 100 lines are consider.
    
    delta_line : int, optional
        Distance between lines in a package. The default is 100.

    window_shift : TYPE, optional
        Displacement of the Package of line for epsilon calculation. The default is 1.
    
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    B Apparent brightness kimogram.
    
    '''
    
    if window_shift=='':
        raise ValueError ('The parameter window shift can not be empty. \n ¿How many lines must be jumped between intensity values?')

    if tp=='':
        raise ValueError ('The parameter pixel dwell time (tp) can not be empty')

    Time = Line_time(Lines[:,0].size, Lines[0,:].size, tp, tr=tr)
    
    
    time = []

    B=[]

    p=0

    l_f = window_points*delta_line  ## l_f = final line parameter use for index. 
    #==============================================================================
    #             Initilization of line's index values for B calculation
    #==============================================================================  
    index=(np.arange(0, l_f, delta_line))+window_shift*p

    # print(index)
    for p in tqdm.trange(0,delta_line,window_shift):

        pack_of_lines=[]    

        #==========================================================================
        #             Update of line's index values for B calculation
        #==========================================================================
        index=(np.arange(0, l_f, delta_line))+window_shift*p
            
        for i in index:
            pack_of_lines.append(list(Lines[i]))

        pack_of_lines=np.asarray(pack_of_lines)        
                
        k_medio = (np.mean(pack_of_lines, axis=0)).ravel()
        
        # Variance is calculated as intensity distribution 2nd moment --->  Var = <(I^2)> - (<I>)^2 but with the denominator as N-1 (ie: ddof=1)
        VAR = np.var(pack_of_lines, axis=0, ddof=1).ravel() 

        B_line = np.array([1.0]*len(k_medio)).ravel()
        
        for k in range(0,len(k_medio)):
            if k_medio[k] == 0 or VAR[k]==k_medio[k]:
                B_line[k]= 1  

            else:
                B_line[k] = VAR[k]/k_medio[k]

        B.append(list(B_line))

        time.append(Time[0]*(p+1))


    B = np.asarray(B)
    
    return B, time


def pCF_of_MATE_BA(Kimogram , linetime, dr=0, reverse_PCF=False, return_time=0, logtime=False, Movil_log=0):
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

def apply_pCF_of_MATE():
    '''
    This function computes the correlation (using the MATE function) for the given kimograms with the provided
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
    G1, T1 = pCF_of_MATE_BA(original_kimograms[0][first_line:last_line], line_time, dr=dr, reverse_PCF=reverse)
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
        A1, _ = pCF_of_MATE_BA(original_kimograms[0][first_line:last_line], line_time, dr=0, reverse_PCF=reverse)
        A1_downsampled = np.asarray([A1[i] for i in x1_ch1]).T
        A1_log = np.empty_like(A1_downsampled)
        for i, gi in enumerate(A1_downsampled):
            A1_log[i] = np.interp(t_log_ch1, t_lineal_ch1_resampled, gi)
        A1_log = gaussian_filter(A1_log, sigma=sigma)
        A1_max = np.maximum(np.max(A1_log, axis=1), 1e-6)
        G1_log = G1_log / A1_max[:len(G1_log), None]
        G1_log = np.clip(G1_log, None, 2)

        G_to_save = G1_log

    # ---- CHANNEL 2 ----
    if len(original_kimograms) > 1:
        G2, T2 = pCF_of_MATE_BA(original_kimograms[1][first_line:last_line], line_time, dr=dr, reverse_PCF=reverse)
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
            A2, _ = pCF_of_MATE_BA(original_kimograms[1][first_line:last_line], line_time, dr=0, reverse_PCF=reverse)
            A2_downsampled = np.asarray([A2[i] for i in x1_ch2]).T
            A2_log = np.empty_like(A2_downsampled)
            for i, gi in enumerate(A2_downsampled):
                A2_log[i] = np.interp(t_log_ch2, t_lineal_ch2_resampled, gi)
            A2_log = gaussian_filter(A2_log, sigma=sigma)
            A2_max = np.maximum(np.max(A2_log, axis=1), 1e-6)
            G2_log = G2_log / A2_max[:len(G2_log), None]
            G2_log = np.clip(G2_log, None, 2)
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

def display_plot(mate_window):
    try:
        fig = apply_pCF_of_MATE()

        # Create a child window attached to mate_window
        plot_window = tk.Toplevel(mate_window)
        plot_window.title("MATE-BA Plot")
        plot_window.geometry("1200x800")

        frame = tk.Frame(plot_window)
        frame.grid(row=0, column=0, sticky="nsew")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        button_frame = tk.Frame(plot_window)
        button_frame.grid(row=1, column=0, pady=10)

        # tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style).pack(side=tk.LEFT, padx=5)
        # tk.Button(button_frame, text="Save Data", command=save_data, **button_style).pack(side=tk.LEFT, padx=5)
        # tk.Button(button_frame, text="Save Time", command=save_time, **button_style).pack(side=tk.LEFT, padx=5)
        # tk.Button(button_frame, text="Profiles", command=lambda: open_profiles_window(plot_window), **button_style).pack(side=tk.LEFT, padx=5)
        # tk.Button(button_frame, text="Fit ACF", command=open_fitacf_window, **button_style).pack(side=tk.LEFT, padx=5)

        mate_window.after(0, hide_working_and_update_result, "Plot Applied!")

    except Exception as e:
        print(f"[CRITICAL] display_plot failed: {str(e)}")
        raise
    


###############################################################################

def run_MATE_BA(mate_window):
    """
    Entry point for MATE-BA analysis.
    Called from the main GUI.
    """
    # Example: open file dialog and run correlation
    cmap_var = tk.StringVar(value="viridis")
    abrir_MATE_ventana()
