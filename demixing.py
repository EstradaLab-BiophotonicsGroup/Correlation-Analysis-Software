# -*- coding: utf-8 -*-
"""
Send to Communications Biology 
regarding the manuscript "Spectral demixin-enhanced dual-color pair correlation function: 
                          Application to the study of host-pathogen protein interaction"
                          
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from PIL import Image

def filtrado(data, calculop=True, p=None):
    """'
    This function, provided of the control data, will compute the p matrix and demixed controls 
    and if wanted, a test experiment (cotransfected). 
    
    data:   matrix of the form [data_protein1, data_protein2, test]
            and each one of the matrix corresponding to an experiment should contain
            both channels of adquisition with the structure:
            [[time series, pixels, pixels]_ch1, [time series, pixels, pixels]_ch2]
            
    calculop:   the default is True. It will use the data provided to compute the p matrix.
                If False, a p matrix should be provided and the code will only demixed the data
                using said matrix. This is useful for testing an already calculated matrix on another
                data set.
                
                
    returns: the data provided demixed and the p matrix for futere use.
    """
    
    n_exp = len(data)
    nombre = ['Protein 1', 'Protein 2', 'Test']

    #------ separar canales ------
    ch1 = [d[0] for d in data]
    ch2 = [d[1] for d in data]

    ch1mean = [np.mean(c) for c in ch1]
    ch2mean = [np.mean(c) for c in ch2]

    if calculop:
        p = np.array([[ch1mean[0], ch1mean[1]],
                      [ch2mean[0], ch2mean[1]]])

    p_inv = np.linalg.inv(p)

    ch1_filtrado = [np.zeros_like(c) for c in ch1]
    ch2_filtrado = [np.zeros_like(c) for c in ch2]

    # colormaps
    custom_cmap_green = LinearSegmentedColormap.from_list(
        'black_green', [(0, 0, 0), (0, 1, 0)], N=100)

    custom_cmap_red = LinearSegmentedColormap.from_list(
        'black_red', [(0, 0, 0), (1, 0, 0)], N=100)

    for i in range(n_exp):
        #------ demixing ------
        stack = np.stack([ch1[i], ch2[i]], axis=0)
        shp = stack.shape
        stack_flat = stack.reshape(2, -1)
        demixed = p_inv @ stack_flat
        ch1_f = demixed[0].reshape(shp[1:])
        ch2_f = demixed[1].reshape(shp[1:])

        #------ normalización para la visualización ------
        ch1_f *= np.max(ch1[i]) / np.max(ch1_f)
        ch2_f *= np.max(ch2[i]) / np.max(ch2_f)
        ch1_filtrado[i] = ch1_f
        ch2_filtrado[i] = ch2_f

        #------ plots CH1 ------
        fig, (ax0, ax1) = plt.subplots(1, 2)
        im0 = ax0.imshow(ch1[i], cmap=custom_cmap_green, vmin=0)
        im1 = ax1.imshow(ch1_f, cmap=custom_cmap_green, vmin=0)

        for ax, im in zip((ax0, ax1), (im0, im1)):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.axis('off')

        ax0.set_title(f'{nombre[i]}-CH1')
        ax1.set_title(f'{nombre[i]}-demixed CH1')
        plt.show()

        #------- plots CH2 ------
        fig, (ax0, ax1) = plt.subplots(1, 2)
        im0 = ax0.imshow(ch2[i], cmap=custom_cmap_red, vmin=0)
        im1 = ax1.imshow(ch2_f, cmap=custom_cmap_red, vmin=0)

        for ax, im in zip((ax0, ax1), (im0, im1)):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.axis('off')

        ax0.set_title(f'{nombre[i]}-CH2')
        ax1.set_title(f'{nombre[i]}-demixed CH2')
        plt.show()

    return ch1_filtrado, ch2_filtrado, p

