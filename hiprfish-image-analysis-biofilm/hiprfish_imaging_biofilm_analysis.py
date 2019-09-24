
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import joblib
import argparse
import itertools
import javabridge
import bioformats
import numpy as np
import pandas as pd
import skimage.filters
from sklearn import svm
from skimage import color
from ete3 import NCBITaxa
from skimage import exposure
from skimage import restoration
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.future import graph
from sklearn.cluster import KMeans
from neighbor import line_profile_v2
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from scipy.ndimage import binary_opening
from matplotlib.colors import hsv_to_rgb
from neighbor2d import line_profile_2d_v2
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import binary_fill_holes
from matplotlib.ticker import ScalarFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from neighbor import line_profile_memory_efficient_v2
from neighbor import line_profile_memory_efficient_v3
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(length):
    return(length/2.54)

def load_ztslice(filename, z_index, t_index):
    image = bioformats.load_image(filename, z = z_index, t = t_index)
    return(image)

def load_ztslice_tile(filename, z_index, t_index, tile):
    image = bioformats.load_image(filename, z = z_index, t = t_index, series = tile)
    return(image)

def get_x_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    x_range = ome.image(0).Pixels.get_SizeX()
    return(x_range)

def get_y_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    y_range = ome.image(0).Pixels.get_SizeY()
    return(y_range)

def get_c_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    c_range = ome.image(0).Pixels.get_SizeC()
    return(c_range)

def get_t_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    t_range = ome.image(0).Pixels.get_SizeT()
    return(t_range)

def get_z_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    z_range = ome.image(0).Pixels.get_SizeZ()
    return(z_range)

def get_tile_size(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    return(int(np.sqrt(ome.image_count)))

def get_image_count(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    return(ome.image_count)

def load_image_tile(filename):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice(filename, k, 0) for k in range(0, z_range)], axis = 2)
    return(image)

def load_image_zstack_fixed_t(filename, t):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice(filename, k, t) for k in range(0, z_range)], axis = 2)
    return(image)

def load_image_zstack_fixed_t_memory_efficient(filename, t, z_min, z_max):
    image = np.stack([load_ztslice(filename, k, t) for k in range(z_min, z_max)], axis = 2)
    return(image)

def load_image_zstack_fixed_t_tile(filename, t, tile):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice_tile(filename, k, t, tile) for k in range(0, z_range)], axis = 2)
    return(image)

def get_registered_image_from_tile(filename, tile_size, overlap):
    image_0 = load_ztslice_tile(filename, 0, 0, 0)
    full_size_x = image_0.shape[0]*tile_size[0] - overlap*(tile_size[0] - 1)
    full_size_y = image_0.shape[1]*tile_size[1] - overlap*(tile_size[1] - 1)
    image_full = np.zeros((full_size_x, full_size_y, image_0.shape[2]))
    for i in range(tile_size[0]):
        for j in range(tile_size[1]):
            overlap_compensation_x = 200*i
            overlap_compensation_y = 200*j
            image_full[i*image_0.shape[0] - overlap_compensation_x: (i+1)*image_0.shape[0] - overlap_compensation_x, j*image_0.shape[0] - overlap_compensation_y: (j+1)*image_0.shape[0] - overlap_compensation_y, :] = load_ztslice_tile(filename, 0, 0, i*tile_size[1] + j)
    return(image_full)

def get_registered_average_image_from_tstack(filename):
    image_0 = load_image_zstack_fixed_t(filename, 0)
    image_registered = image_0.copy()
    image_0_sum = np.sum(image_0, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    for i in range(1, nt):
        image_i = load_image_zstack_fixed_t(filename, i)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.feature.register_translation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_0.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_0.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_0.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_0.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_0.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_0.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_0.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        image_registered += image_registered_hold
    return(image_registered/nt)

def get_registered_image_from_tstack(filename):
    print('Loading {} at t = 0...'.format(os.path.basename(filename)))
    image_0 = load_image_zstack_fixed_t(filename, 0)
    image_registered = [image_0.copy()]
    image_0_sum = np.sum(image_0, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    for i in range(1, nt):
        print('Loading {} at t = {}...'.format(os.path.basename(filename), i))
        image_i = load_image_zstack_fixed_t(filename, i)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.feature.register_translation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_0.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_0.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_0.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_0.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_0.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_0.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_0.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        image_registered.append(image_registered_hold)
    image_registered = np.stack(image_registered, axis = 4)
    return(image_registered)

def get_registered_image_from_tstack_tile(filename, tile):
    image_0 = load_image_zstack_fixed_t_tile(filename, 0, tile)
    image_registered = image_0.copy()
    image_0_sum = np.sum(image_0, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    shift_filter_mask = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), True, dtype = bool)
    for i in range(1, nt):
        image_i = load_image_zstack_fixed_t_tile(filename, i, tile)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.feature.register_translation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask_hold = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_0.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_0.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_0.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_0.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_0.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_0.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_0.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        shift_filter_mask = shift_filter_mask*shift_filter_mask_hold
        image_registered += image_registered_hold
    return(image_registered, shift_filter_mask)

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', box_color = 'white')
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    segfilename = sample + '_seg.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    np.save(sample + '_seg', segmentation)
    return

def save_identification(image_identification, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_identification)
    scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    segfilename = sample + '_identification.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_identification_filtered(image_identification, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_identification)
    scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    segfilename = sample + '_identification_filtered.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_identification_bvox(image_identification, sample):
    nx, ny, nz, nframes = image_identification.shape[0], image_identification.shape[1], image_identification.shape[2], 1
    header = np.array([nx,ny,nz,nframes])
    color_filename = ['r', 'g', 'b']
    for i in range(3):
        pointdata = image_identification[:,:,:,i]
        binfile = open('{}_identification_{}.bvox'.format(sample, color_filename[i]),'wb')
        header.astype('<i4').tofile(binfile)
        pointdata.flatten('F').astype('<f4').tofile(binfile)
    return

def save_raw_image_bvox(image_registered_sum, sample):
    nx, ny, nz, nframes = image_registered_sum.shape[0], image_registered_sum.shape[1], image_registered_sum.shape[2], 1
    header = np.array([nx,ny,nz,nframes])
    binfile = open('{}_raw_image.bvox'.format(sample),'wb')
    header.astype('<i4').tofile(binfile)
    image_registered_sum.flatten('F').astype('<f4').tofile(binfile)
    return

def save_sum_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    segfilename = sample + '_sum.png'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_enhanced_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    plt.axis('off')
    segfilename = sample + '_enhanced.pdf'
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    return

def generate_2d_segmentation(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [bioformats.load_image(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]), np.log(image_sum[i]))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape[0]
    for i in range(len(image_stack)):
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        print(shift_row, shift_col)
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    image_channel = np.concatenate(image_registered, axis = 2)
    image_registered_sum = np.sum(image_channel, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:,:,:,None]
    image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
    image_lp_rnc = image_lp_rel_norm[:,:,:,5]
    image_lprns = np.average(image_lp_rnc, axis = 2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
    image_lprn_qcv = np.zeros(image_lprn_uq.shape)
    image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
    image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum_nl*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    image_bkg = image_bkg_filter_mask == 0
    image_bkg = skimage.morphology.remove_small_objects(image_bkg, 10000)
    image_bkg = binary_fill_holes(image_bkg)
    structuring_element = skimage.morphology.disk(100)
    image_bkg_bc = skimage.morphology.binary_closing(image_bkg, structuring_element)
    image_bkg_bc_objects = skimage.measure.label(image_bkg_bc)
    image_bkg_bc_objects_props = skimage.measure.regionprops(image_bkg_bc_objects)
    image_bkg_bc_areas = [x.area for x in image_bkg_bc_objects_props]
    image_bkg_final = image_bkg_bc_objects == image_bkg_bc_objects_props[np.argmax(image_bkg_bc_areas)].label
    image_bkg_final_bd = skimage.morphology.binary_dilation(image_bkg_final, structuring_element)
    image_objects_overall = skimage.measure.label(1 - image_bkg_final_bd)
    image_objects_overall_seg = skimage.segmentation.watershed(-image_registered_sum, image_objects_overall)
    image_objects_overall_props = skimage.measure.regionprops(image_objects_overall_seg)
    image_objects_areas = [x.area for x in image_objects_overall_props]
    image_epithelial_area = image_objects_overall_seg != image_objects_overall_props[np.argmax(image_objects_areas)].label
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg, adjacency_seg, image_epithelial_area)

def generate_3d_segmentation(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_channel = np.concatenate(image_registered, axis = 3)
    image_registered_sum = np.sum(image_channel, axis = (3,4))
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 0.03)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
    image_lp_rnc = line_profile_memory_efficient_v2(image_padded.astype(np.float64), 11, 9, 9)
    image_lprns = np.average(image_lp_rnc, axis = 3)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 3)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 3)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg_mask[image_final > 0] = (intensity_rough_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    intensity_rough_seg_mask = skimage.morphology.remove_small_holes(intensity_rough_seg_mask)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh*intensity_rough_seg_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl+1e-8)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum_nl*(image_bkg_filter == 0)
    image1 = image_registered_sum_nl*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    # image_final_bkg_filtered = image_registered_sum_nl_log*image_bkg_filter_mask
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    # image_watershed_sauvola = threshold_sauvola(image_registered_sum_nl, k = 0)
    # image_watershed_mask_bkg_filtered = (image_registered_sum_nl > image_watershed_sauvola)*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_lprns_rsfbosm_bfh*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    # image_seg = skimage.morphology.remove_small_objects(image_seg, size_limit)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_bkg_filter_mask)
    # adjacency_seg = skimage.morphology.remove_small_objects(adjacency_seg, size_limit)
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    # save_segmentation(image_seg, sample)
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg)

def get_image(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack[:,:,:,:,] = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
    image_stack = np.concatenate(image_stack, axis = 3)
    return(image_stack)

def get_t_average_image(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_average_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
    image_stack = np.concatenate(image_stack, axis = 3)
    return(image_stack)

def generate_2d_segmentation_from_tile(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    tile_size = (4,2)
    overlap = 200
    image_stack = [get_registered_image_from_tile(filename, tile_size, overlap) for filename in image_name]
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]), np.log(image_sum[i]))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1]), False, dtype = bool) for image in image_stack]
    image_shape_row = image_stack[0].shape[0]
    image_shape_col = image_stack[0].shape[1]
    for i in range(len(image_stack)):
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        print(shift_row, shift_col)
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape_row + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape_col + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape_row - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape_col - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    image_channel = np.concatenate(image_registered, axis = 2)
    image_registered_sum = np.sum(image_channel, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:,:,:,None]
    image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
    image_lp_rnc = image_lp_rel_norm[:,:,:,5]
    image_lprns = np.average(image_lp_rnc, axis = 2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
    image_lprn_qcv = np.zeros(image_lprn_uq.shape)
    image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
    image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum_nl*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)

    image_bkg = image_bkg_filter_mask == 0
    image_bkg = skimage.morphology.remove_small_objects(image_bkg, 10000)
    image_bkg = binary_fill_holes(image_bkg)
    structuring_element = skimage.morphology.disk(100)
    image_bkg_bc = skimage.morphology.binary_dilation(image_bkg, structuring_element)
    image_bkg_bc_objects = skimage.measure.label(image_bkg_bc)
    image_bkg_bc_objects_props = skimage.measure.regionprops(image_bkg_bc_objects)
    image_bkg_bc_areas = [x.area for x in image_bkg_bc_objects_props]
    image_bkg_final = image_bkg_bc_objects == image_bkg_bc_objects_props[np.argmax(image_bkg_bc_areas)].label
    image_bkg_final_bd = skimage.morphology.binary_dilation(image_bkg_final)
    image_objects_overall = skimage.measure.label(1 - image_bkg_final_bd)
    image_objects_overall_seg = skimage.segmentation.watershed(-image_registered_sum, image_objects_overall)
    image_objects_overall_props = skimage.measure.regionprops(image_objects_overall_seg)
    image_objects_areas = [x.area for x in image_objects_overall_props]
    image_epithelial_area = image_objects_overall_seg != image_objects_overall_props[np.argmax(image_objects_areas)].label
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg, adjacency_seg, epithelial_boundary)

def generate_2d_segmentation_from_zstack(image_stack, z):
    image_registered = image_stack[:,:,z,:,:]
    image_registered_sum = np.sum(image_registered, axis = (2,3))
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:,:,:,None]
    image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
    image_lp_rnc = image_lp_rel_norm[:,:,:,5]
    image_lprns = np.average(image_lp_rnc, axis = 2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    intensity_rough_seg_mask = skimage.morphology.remove_small_holes(intensity_rough_seg_mask)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh*intensity_rough_seg_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_bkg_filter_mask)
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, adjacency_seg)

def generate_2d_segmentation_from_zstack_t_sum(image_stack, z):
    image_registered = image_stack[:,:,z,:]
    image_registered_sum = np.sum(image_registered, axis = 2)
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:,:,:,None]
    image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
    image_lp_rnc = image_lp_rel_norm[:,:,:,5]
    image_lprns = np.average(image_lp_rnc, axis = 2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    intensity_rough_seg_mask = skimage.morphology.remove_small_holes(intensity_rough_seg_mask)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh*intensity_rough_seg_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_bkg_filter_mask)
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, adjacency_seg)

def generate_3d_segmentation_memory_efficient(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    image_registered_sum = np.sum(image_registered, axis = 3)
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_lp = line_profile_memory_efficient_v2(image_padded.astype(np.float64), 11, 9, 9)
    image_lprns = np.average(image_lp, axis = 3)
    image_lprn_lq = np.percentile(image_lp, 25, axis = 3)
    image_lprn_uq = np.percentile(image_lp, 75, axis = 3)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg[image_final > 0] = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    if i0 < i1:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0]
    else:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0] == 0
    image_final_seg = np.zeros(image_final.shape).astype(int)
    image_final_seg[image_final > 0] = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    image_lprns_rsf = np.zeros(image_final.shape).astype(int)
    image_lprns_rsf[image_final > 0] = (image_final_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    image_lprns_rsfbo = binary_opening(image_lprns_rsf)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_watershed_seeds = skimage.morphology.label(image_watershed_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum + 1)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_watershed_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def generate_3d_segmentation_tile(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [load_image_zstack_fixed_t(filename, t = 0) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    np.save('{}_registered.npy'.format(sample), image_registered)
    image_registered_sum = np.sum(image_registered, axis = 3)
    del image_registered
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_final = np.zeros(image_registered_sum.shape)
    for i in range(10):
        for j in range(10):
            print('Calculating tile {}, {}'.format(i,j))
            image_padded_temp = image_padded[i*200:(i+1)*200+10, j*200:(j+1)*200+10,:]
            image_lp = line_profile_v2(image_padded_temp.astype(np.float64), 11, 9, 9)
            image_lp = np.nan_to_num(image_lp)
            image_lp_min = np.min(image_lp, axis = 4)
            image_lp_max = np.max(image_lp, axis = 4)
            image_lp_max = image_lp_max - image_lp_min
            image_lp = image_lp - image_lp_min[:,:,:,:,None]
            image_lp_rel_norm = image_lp/image_lp_max[:,:,:,:,None]
            image_lp_rnc = image_lp_rel_norm[:,:,:,:,5]
            image_lprns = np.average(image_lp_rnc, axis = 3)
            image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 3)
            image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 3)
            image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
            image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
            image_final[i*200:(i+1)*200, j*200:(j+1)*200,:] = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    image2 = image_final*(intensity_rough_seg == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def generate_3d_segmentation_slice(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [load_image_zstack_fixed_t(filename, t = 0) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    np.save('{}_registered.npy'.format(sample), image_registered)
    image_registered_sum = np.sum(image_registered, axis = 3)
    del image_registered
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_final = np.zeros(image_registered_sum.shape)
    for i in range(image_registered_sum.shape[2]):
        print('Calculating slice {}'.format(i))
        image_padded = skimage.util.pad(image_registered_sum[:,:,i], 5, mode = 'edge')
        image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
        image_lp = np.nan_to_num(image_lp)
        image_lp_min = np.min(image_lp, axis = 3)
        image_lp_max = np.max(image_lp, axis = 3)
        image_lp_max = image_lp_max - image_lp_min
        image_lp = image_lp - image_lp_min[:,:,:,None]
        image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
        image_lp_rnc = image_lp_rel_norm[:,:,:,5]
        image_lprns = np.average(image_lp_rnc, axis = 2)
        image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
        image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
        image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
        image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
        image_final[:,:,i] = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_epithelial_area = np.zeros(image_seg.shape)
    for i in range(image_seg.shape[2]):
        print('Calculating slice {}'.format(i))
        image_bkg = image_bkg_filter_mask[:,:,i] == 0
        image_bkg = skimage.morphology.remove_small_objects(image_bkg, 10000)
        image_bkg = binary_fill_holes(image_bkg)
        structuring_element = skimage.morphology.disk(100)
        image_bkg_bc = skimage.morphology.binary_dilation(image_bkg, structuring_element)
        image_bkg_bc_objects = skimage.measure.label(image_bkg_bc)
        image_bkg_bc_objects_props = skimage.measure.regionprops(image_bkg_bc_objects)
        image_bkg_bc_areas = [x.area for x in image_bkg_bc_objects_props]
        image_bkg_final = image_bkg_bc_objects == image_bkg_bc_objects_props[np.argmax(image_bkg_bc_areas)].label
        image_bkg_final_bd = skimage.morphology.binary_dilation(image_bkg_final)
        image_objects_overall = skimage.measure.label(1 - image_bkg_final_bd)
        image_objects_overall_seg = skimage.segmentation.watershed(-image_registered_sum[:,:,i], image_objects_overall)
        image_objects_overall_props = skimage.measure.regionprops(image_objects_overall_seg)
        image_objects_areas = [x.area for x in image_objects_overall_props]
        if image_objects_areas:
            image_epithelial_area[:,:,i] = image_objects_overall_seg != image_objects_overall_props[np.argmax(image_objects_areas)].label

    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, image_epithelial_area)

def generate_3d_segmentation_tile_memory_efficient(sample):
    image_name = '{}_561.czi'.format(sample)
    image_tile_size = get_tile_size(image_name)
    image_sum_list = []
    shift_filter_mask_list = []
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            print('Analyzing tile [{}, {}]'.format(i,j))
            image_stack, shift_filter_mask = get_registered_image_from_tstack_tile(image_name, i*image_tile_size + j)
            image_sum = np.sum(image_stack, axis = 3)
            image_sum_list.append(image_sum)
            shift_filter_mask_list.append(shift_filter_mask)
    image_sum_filtered_list = [image_sum_list[i]*shift_filter_mask_list[i] for i in range(image_tile_size*image_tile_size)]
    shift_vector_full = np.zeros((image_tile_size, image_tile_size, 3))
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            if (i == 0) & (j == 0):
                shift_vector_full[i,j,:] = np.zeros(3)
            elif (i > 0) & (j == 0):
                shift_vector = skimage.feature.register_translation(image_sum_filtered_list[(i-1)*image_tile_size][450:500,:,:], image_sum_filtered_list[i*image_tile_size][0:50,:,:])
                shift_vector_full[i, j, :] = shift_vector[0]
            else:
                shift_vector = skimage.feature.register_translation(image_sum_filtered_list[i*image_tile_size + j - 1][:,450:500,:], image_sum_filtered_list[i*image_tile_size + j][:,0:50,:])
                shift_vector_full[i, j, :] = shift_vector[0]
    image_full = np.zeros((2020, 2020, 170))
    image_overlap_full = np.zeros((2020, 2020, 170))
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            x_min = int(i*500 - 50*i + np.sum(shift_vector_full[0:i+1, 0, 0]) + np.sum(shift_vector_full[i, 1:j+1, 0])) + 10
            x_max = int((i+1)*500 - 50*i + np.sum(shift_vector_full[0:i+1, 0, 0]) + np.sum(shift_vector_full[i, 1:j+1, 0])) + 10
            y_min = int(j*500 - 50*j + np.sum(shift_vector_full[i, 0:j+1, 1])) + 10
            y_max = int((j+1)*500 - 50*j + np.sum(shift_vector_full[i, 0:j+1, 1])) + 10
            z_min = int(np.sum(shift_vector_full[i, 0:j+1, 2])) + 10
            z_max = int(150 + np.sum(shift_vector_full[i, 0:j+1, 2])) + 10
            image_full[x_min:x_max,y_min:y_max,z_min:z_max] += image_sum_filtered_list[i*image_tile_size + j]*shift_filter_mask_list[i*image_tile_size + j]
            image_overlap_full[x_min:x_max,y_min:y_max,z_min:z_max][shift_filter_mask_list[i*image_tile_size + j] > 0] += 1
    image_overlap_full[image_overlap_full == 0] = 1
    image_full = image_full/image_overlap_full
    image_norm = image_full/np.max(image_full)
    image_padded = skimage.util.pad(image_norm, 5, mode = 'edge')
    image_final = np.zeros(image_norm.shape)
    for i in range(20):
        for j in range(20):
            x_start = i*100
            x_end = (i+1)*100 + 10
            y_start = j*100
            y_end = (j+1)*100 + 10
            image_chunk = image_padded[x_start: x_end, y_start: y_end, :]
            image_lp_chunk = line_profile_v2(image_chunk.astype(np.float64), 11, 9, 9)
            # image_lp_chunk = np.nan_to_num(image_lp_chunk)
            image_lp_chunk_min = np.min(image_lp_chunk, axis = 4)
            image_lp_chunk_max = np.max(image_lp_chunk, axis = 4)
            image_lp_chunk_max = image_lp_chunk_max - image_lp_chunk_min
            image_lp_chunk = image_lp_chunk - image_lp_chunk_min[:,:,:,:,None]
            image_lp_rel_norm_chunk = image_lp_chunk/(image_lp_chunk_max[:,:,:,:,None] + 1e-8)
            image_lp_rnc_chunk = image_lp_rel_norm_chunk[:,:,:,:,5]
            image_lprns_chunk = np.average(image_lp_rnc_chunk, axis = 3)
            image_lprn_lq_chunk = np.percentile(image_lp_rnc_chunk, 25, axis = 3)
            image_lprn_uq_chunk = np.percentile(image_lp_rnc_chunk, 75, axis = 3)
            image_lprn_qcv_chunk = (image_lprn_uq_chunk - image_lprn_lq_chunk)/(image_lprn_uq_chunk + image_lprn_lq_chunk + 1e-8)
            # image_lprn_qcv_chunk = np.nan_to_num(image_lprn_qcv_chunk)
            image_final_chunk = image_lprns_chunk*(1-image_lprn_qcv_chunk)
            image_final[i*100:(i+1)*100, j*100:(j+1)*100, :] = image_final_chunk
    intensity_rough_seg = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg[image_final > 0] = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    if i0 < i1:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0]
    else:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0] == 0
    image_final_seg = np.zeros(image_final.shape).astype(int)
    image_final_seg[image_final > 0] = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final[image_final > 0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    image_lprns_rsf = np.zeros(image_final.shape).astype(int)
    image_lprns_rsf[image_final > 0] = (image_final_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    image_lprns_rsfbo = binary_opening(image_lprns_rsf)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_watershed_seeds = skimage.morphology.label(image_watershed_mask)
    image_norm_log = np.log10(image_norm + 1e-8)
    image_bkg_filter = np.zeros(image_norm.shape)
    image_bkg_filter[image_norm > 0] =  KMeans(n_clusters = 2, random_state = 0).fit_predict(image_norm_log[image_norm > 0].reshape(-1, 1))
    image0 = image_norm[image_norm > 0]*(image_bkg_filter[image_norm > 0] == 0)
    image1 = image_norm[image_norm > 0]*(image_bkg_filter[image_norm > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter[~(image_norm > 0)] = 1
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_watershed_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def get_volume(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = (3,4)) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
    return(image_stack)

def measure_epithelial_distance(cx, cy, ebc):
    distance = np.zeros(ebc.shape[0])
    for i in range(ebc.shape[0]):
        distance[i] = np.sqrt((cx - ebc[i,0])**2 + (cy - ebc[i,1])**2)
    return(np.min(distance))

def measure_biofilm_images_2d(sample, umap_transform, scaler, clf_umap, clf, taxon_lookup):
    image_registered_sum, image_registered, image_final_bkg_filtered, segmentation, adjacency_seg, image_epithelial_area = generate_2d_segmentation(sample)
    np.save('{}_registered.npy'.format(sample), image_registered)
    np.save('{}_seg.npy'.format(sample), segmentation)
    np.save('{}_adjacency_seg.npy'.format(sample), adjacency_seg)
    np.save('{}_epithelial_area.npy'.format(sample), image_epithelial_area)
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[2]))
    for k in range(0, image_registered.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    pd.DataFrame(avgint).to_csv('{}_avgint.csv'.format(sample), index = None)
    avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm_scaled = scaler.transform(avgint_norm[:,0:63])
    avgint_norm[:,63] = clf[0].predict(avgint_norm_scaled[:,0:23])
    avgint_norm[:,64] = clf[1].predict(avgint_norm_scaled[:,23:43])
    avgint_norm[:,65] = clf[2].predict(avgint_norm_scaled[:,43:57])
    avgint_norm[:,66] = clf[3].predict(avgint_norm_scaled[:,57:63])
    avgint_umap_transformed = umap_transform.transform(avgint_norm)
    cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    cell_ids_norm_prob = clf_umap.predict_proba(avgint_umap_transformed)
    max_prob = np.max(cell_ids_norm_prob, axis = 1)
    cell_info = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None], max_prob[:,None], cell_ids_norm_prob), axis = 1))
    cell_info.columns = ['channel_{}'.format(i) for i in range(63)] + ['intensity_classification_{}'.format(i) for i in range(4)] + ['cell_barcode', 'max_probability'] + ['{}_prob'.format(x) for x in clf_umap.classes_]
    cell_info['sample'] = sample
    cell_info['label'] = np.asarray([x.label for x in cells])
    cell_info['centroid_x'] = np.asarray([x.centroid[0] for x in cells])
    cell_info['centroid_y'] = np.asarray([x.centroid[1] for x in cells])
    cell_info['major_axis'] = np.asarray([x.major_axis_length for x in cells])
    cell_info['minor_axis'] = np.asarray([x.minor_axis_length for x in cells])
    cell_info['eccentricity'] = np.asarray([x.eccentricity for x in cells])
    cell_info['orientation'] = np.asarray([x.orientation for x in cells])
    cell_info['area'] = np.asarray([x.area for x in cells])
    cell_info['epithelial_distance'] = 0
    cell_info['max_intensity'] = cell_info.loc[:, ['channel_{}'.format(i) for i in range(63)]].max(axis = 1).values
    cell_info['type'] = 'cell'
    cellinfofilename = sample + '_cell_information.csv'
    cell_info.to_csv(cellinfofilename, index = None)
    ids = list(set(cell_ids_norm))
    image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    image_identification_barcode = np.zeros(segmentation.shape)
    for q in range(0, len(ids)):
        cell_population = np.where(cell_ids_norm == ids[q])[0]
        for r in range(0, len(cell_population)):
            image_identification_barcode[segmentation == cell_population[r]+1] = int(ids[q], 2)
            if ids[q] in taxon_lookup.code.values:g
                image_identification[segmentation == cell_population[r]+1, :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == ids[q], ['H', 'S', 'V']].values)
            else:
                image_identification[segmentation == cell_population[r]+1, :] = np.array([1,1,1])
    save_identification(image_identification, sample)
    debris = segmentation*image_epithelial_area
    image_identification_filtered = image_identification.copy()
    image_identification_filtered[debris > 0] = [0.5,0.5,0.5]
    debris_labels = np.delete(np.unique(debris), 0)
    for i in range(cell_info.shape[0]):
        cell_label = cell_info.loc[i, 'label']
        cell_area = cell_info.loc[i, 'area']
        cell_prob = cell_info.loc[i, 'max_probability']
        if (cell_area > 10000) or (cell_label in debris_labels) or (cell_prob <= 0.95):
            cell_info.loc[i, 'type'] = 'debris'
            image_identification_filtered[segmentation == cell_label] = [0.5,0.5,0.5]
    np.save('{}_identification_filtered.npy'.format(sample), image_identification_filtered)
    save_identification_filtered(image_identification, sample)
    cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:].copy()
    cellinfofilteredfilename = sample + '_cell_information_filtered.csv'
    cell_info_filtered.to_csv(cellinfofilteredfilename, index = None)
    avgint_filtered = avgint[cell_info.type.values == 'cell', :]
    pd.DataFrame(avgint_filtered).to_csv('{}_avgint_filtered.csv'.format(sample), index = None)
    edge_map = skimage.filters.sobel(segmentation > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
    adjacency_matrix_filtered = adjacency_matrix.copy()
    for i in range(cell_info.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
            node_1 = e[0]
            node_2 = e[1]
            if (node_1 != 0) and (node_2 !=0):
                barcode_1 = cell_info.iloc[node_1-1,67]
                barcode_2 = cell_info.iloc[node_2-1,67]
                adjacency_matrix.loc[barcode_1, barcode_2] += 1
                if (cell_info.loc[node_1-1,'type'] == 'cell') and (cell_info.loc[node_2-1,'type'] == 'cell'):
                    adjacency_matrix_filtered.loc[barcode_1, barcode_2] += 1
    adjacencyfilename = sample + '_adjacency_matrix.csv'
    adjacencyfilteredfilename = sample + '_adjacency_matrix_filtered.csv'
    adjacency_matrix.to_csv(adjacencyfilename)
    adjacency_matrix_filtered.to_csv(adjacencyfilteredfilename)
    return

def measure_biofilm_images_2d_from_zstack(sample, image_stack, umap_transform, scaler, clf_umap, clf, taxon_lookup, z):
    image_registered_sum, image_registered, image_final_bkg_filtered, segmentation, adjacency_seg = generate_2d_segmentation_from_zstack_t_sum(image_stack, z)
    np.save('{}_z_{}_registered.npy'.format(sample, z), image_registered)
    np.save('{}_z_{}_seg.npy'.format(sample, z), segmentation)
    np.save('{}_z_{}_adjacency_seg.npy'.format(sample, z), adjacency_seg)
    save_segmentation(segmentation, '{}_z_{}'.format(sample, z))
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[2]))
    for k in range(0, image_registered.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm_scaled = scaler.transform(avgint_norm[:,0:63])
    avgint_norm[:,63] = clf[0].predict(avgint_norm_scaled[:,0:23])
    avgint_norm[:,64] = clf[1].predict(avgint_norm_scaled[:,23:43])
    avgint_norm[:,65] = clf[2].predict(avgint_norm_scaled[:,43:57])
    avgint_norm[:,66] = clf[3].predict(avgint_norm_scaled[:,57:63])
    avgint_umap_transformed = umap_transform.transform(avgint_norm)
    cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    cell_info = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None]), axis = 1))
    cell_info[68] = sample
    cell_info[69] = np.asarray([x.label for x in cells])
    cell_info[70] = np.asarray([x.centroid[0] for x in cells])
    cell_info[71] = np.asarray([x.centroid[1] for x in cells])
    cell_info[72] = np.asarray([x.major_axis_length for x in cells])
    cell_info[73] = np.asarray([x.minor_axis_length for x in cells])
    cell_info[74] = np.asarray([x.eccentricity for x in cells])
    cell_info[75] = np.asarray([x.orientation for x in cells])
    cell_info[76] = np.asarray([x.area for x in cells])
    cellinfofilename = '{}_z_{}_cell_information.csv'.format(sample, z)
    cell_info.to_csv(cellinfofilename, index = None, header = None)
    ids = list(set(cell_ids_norm))
    image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    image_identification_barcode = np.zeros(segmentation.shape)
    for q in range(0, len(ids)):
        cell_population = np.where(cell_ids_norm == ids[q])[0]
        for r in range(0, len(cell_population)):
            image_identification_barcode[segmentation == cell_population[r]+1] = int(ids[q], 2)
            if ids[q] in taxon_lookup.code.values:
                image_identification[segmentation == cell_population[r]+1, :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == ids[q], ['H', 'S', 'V']].values)
            else:
                image_identification[segmentation == cell_population[r]+1, :] = np.array([1,1,1])
    np.save('{}_z_{}_identification.npy'.format(sample, z), image_identification)
    save_identification(image_identification, '{}_z_{}'.format(sample, z))
    edge_map = skimage.filters.sobel(segmentation > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
    for i in range(cell_info.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
            node_1 = e[0]
            node_2 = e[1]
            if (node_1 != 0) and (node_2 !=0):
                barcode_1 = cell_info.iloc[node_1-1,67]
                barcode_2 = cell_info.iloc[node_2-1, 67]
                adjacency_matrix.loc[barcode_1, barcode_2] += 1
    adjacencyfilename = '{}_z_{}_adjacency_matrix.csv'.format(sample, z)
    adjacency_matrix.to_csv(adjacencyfilename)
    return

def measure_biofilm_images_3d(sample, umap_transform, clf_umap, clf, taxon_lookup, z):
    image_registered_sum, image_channel, image_final_bkg_filtered, image_seg = generate_3d_segmentation_slice(sample)
    np.save('{}_registered.npy'.format(sample), image_registered)
    np.save('{}_seg.npy'.format(sample), segmentation)
    # image_registered = np.average(image_registered, axis = 3)
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[3]))
    for k in range(0, image_registered.shape[3]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm[:,63] = clf[0].predict(avgint_norm[:,0:23])
    avgint_norm[:,64] = clf[1].predict(avgint_norm[:,23:43])
    avgint_norm[:,65] = clf[2].predict(avgint_norm[:,43:57])
    avgint_norm[:,66] = clf[3].predict(avgint_norm[:,57:63])
    avgint_umap_transformed = umap_transform.transform(avgint_norm)
    cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    cell_ids_norm_prob = clf_umap.predict_proba(avgint_umap_transformed)
    max_prob = np.max(cell_ids_norm_prob, axis = 1)
    cell_info = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None], max_prob[:,None], cell_ids_norm_prob), axis = 1))
    cell_info.columns = ['channel_{}'.format(i) for i in range(63)] + ['intensity_classification_{}'.format(i) for i in range(4)] + ['cell_barcode', 'max_probability'] + ['{}_prob'.format(x) for x in clf_umap.classes_]
    cell_info['sample'] = sample
    cell_info['label'] = np.asarray([x.label for x in cells])
    cell_info['centroid_x'] = np.asarray([x.centroid[0] for x in cells])
    cell_info['centroid_y'] = np.asarray([x.centroid[1] for x in cells])
    cell_info['centroid_z'] = np.asarray([x.centroid[1] for x in cells])
    cell_info['area'] = np.asarray([x.area for x in cells])
    cell_info['type'] = 'cell'
    cellinfofilename = sample + '_cell_information.csv'
    cell_info.to_csv(cellinfofilename, index = None, header = None)
    ids = list(set(cell_ids_norm))
    image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], segmentation.shape[2], 3))
    image_identification_barcode = np.zeros(segmentation.shape)
    for q in range(0, len(ids)):
        cell_population = np.where(cell_ids_norm == ids[q])[0]
        for r in range(0, len(cell_population)):
            image_identification_barcode[segmentation == cell_population[r]+1] = int(ids[q], 2)
            if ids[q] in taxon_lookup.code.values:
                image_identification[segmentation == cell_population[r]+1, :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == ids[q], ['H', 'S', 'V']].values)
            else:
                image_identification[segmentation == cell_population[r]+1, :] = np.array([1,1,1])
    np.save('{}_identification.npy'.format(sample), image_identification)
    save_identification_bvox(image_identification, sample)
    debris = segmentation*image_epithelial_area
    image_identification_filtered = image_identification.copy()
    image_identification_filtered[debris > 0] = [0.5,0.5,0.5]
    debris_labels = np.delete(np.unique(debris), 0)
    for i in range(cell_info.shape[0]):
        cell_label = cell_info.loc[i, 'label']
        cell_area = cell_info.loc[i, 'area']
        cell_prob = cell_info.loc[i, 'max_probability']
        if (cell_area > 100000) or (cell_prob <=0.95):
            cell_info.loc[i, 'type'] = 'debris'
            image_identification_filtered[segmentation == cell_label] = [0.5,0.5,0.5]

    np.save('{}_identification_filtered.npy'.format(sample), image_identification_filtered)
    save_identification_bvox(image_identification_filtered, sample)
    return

def main():
    parser = argparse.ArgumentParser('Mesure environmental microbial community spectral images')
    parser.add_argument('input_folder', type = str, help = 'Input folder containing spectral images')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, default = '', help = 'Probe design filename')
    parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    parser.add_argument('-d', '--d', dest = 'd', type = int, help = 'Dimension of images')
    parser.add_argument('-z', '--z', dest = 'z', nargs = '*', type = int, help = 'Indices of z slices to analyze')
    parser.add_argument('-sf', '--sf', dest = 'sf', type = str, help = 'Toggle switch for datasets that contains subfolders')
    args = parser.parse_args()
    probes = pd.read_csv(args.probe_design_filename, dtype = {'code': str})
    ncbi = NCBITaxa()
    taxon_lookup = probes.loc[:,['target_taxon', 'code']].drop_duplicates()
    taxon_lookup['H'] = np.arange(0,1,1/taxon_lookup.shape[0])
    taxon_lookup['S'] = 1
    taxon_lookup['V'] = 1
    taxon_sciname = pd.DataFrame.from_dict(ncbi.get_taxid_translator(taxon_lookup.target_taxon.values), orient = 'index').reset_index()
    taxon_sciname.columns = ['target_taxon', 'sci_name']
    taxon_lookup = taxon_lookup.merge(taxon_sciname, on = 'target_taxon')
    taxon_lookup.to_csv('{}/taxon_color_lookup.csv'.format(args.input_folder))
    umap_transform = joblib.load(args.ref_clf)
    scaler = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_scaler.pkl', args.ref_clf))
    clf_umap = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_svc.pkl', args.ref_clf))
    clf = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_check_svc.pkl', args.ref_clf))
    sub_folders = glob.glob('{}/*'.format(args.input_folder))
    if args.sf == 'T':
        for sf in sub_folders:
            if not 'zstack' in sf:
                filenames = glob.glob('{}/*.czi'.format(sf))
                samples = list(set([re.sub('_[0-9][0-9][0-9].czi', '', file) for file in filenames]))
                for s in samples:
                    if args.d == 2:
                        measure_biofilm_images_2d(s, umap_transform, scaler, clf_umap, clf, taxon_lookup)
                    elif args.z is not None:
                        image_stack = get_t_average_image(s)
                        for z in args.z:
                            measure_biofilm_images_2d_from_zstack(s, image_stack, umap_transform, scaler, clf_umap, clf, taxon_lookup, z)
                    else:
                        measure_biofilm_images_3d(s, umap_transform, clf_umap, clf, taxon_lookup, args.z)
    else:
        filenames = glob.glob('{}/*.czi'.format(args.input_folder))
        samples = list(set([re.sub('_[0-9][0-9][0-9].czi', '', file) for file in filenames]))
        for s in samples:
            if args.d == 2:
                measure_biofilm_images_2d(s, umap_transform, scaler, clf_umap, clf, taxon_lookup)
            elif args.z is not None:
                image_stack = get_t_average_image(s)
                for z in args.z:
                    measure_biofilm_images_2d_from_zstack(s, image_stack, umap_transform, scaler, clf_umap, clf, taxon_lookup, z)
            else:
                measure_biofilm_images_3d(s, umap_transform, clf_umap, clf, taxon_lookup, args.z)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
