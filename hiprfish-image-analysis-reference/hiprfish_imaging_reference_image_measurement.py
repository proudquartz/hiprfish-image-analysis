
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

import os
import re
import sys
import glob
import joblib
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
import skimage.filters
from skimage import color
from skimage import restoration
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

javabridge.start_vm(class_path=bioformats.JARS)

def correct_images(image, calibration_norm):
    image_ffc = image/calibration_norm
    return(image_ffc)

def load_calibration_images(filename):
    calibration_image = np.load(filename)
    calibration_full = np.ones((calibration_image.shape[0], calibration_image.shape[1], 95))
    for i in range(32):
        calibration_full[:,:,i] = calibration_image
    return(calibration_full)

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    segfilename = sample + '_seg.png'
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    np.save(sample + '_seg', segmentation)
    return

def save_identification(image_identification, sample):
    seg_color = color.label2rgb(image_identification, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    segfilename = sample + '_identification.png'
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    return

def plot_avg_int_reference(avgint, sample):
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', sample).group(0))
    figure_filename = '{}_avgint.png'.format(sample)
    fig = plt.figure()
    fig.set_size_inches(5,3)
    plt.plot(avgint.transpose(), 'b', alpha = 0.1)
    plt.plot(np.average(avgint, axis = 0), 'r', alpha = 0.8)
    plt.text(72, 1.05, 'Encoding {}'.format(enc))
    plt.ylim(-0.02,1.2)
    plt.xlabel('Channels [-]')
    plt.ylabel('Intensity [A.U.]')
    plt.tight_layout()
    plt.savefig(figure_filename, dpi = 300, format = 'png')
    plt.close()
    return

def segment_images(image_stack):
    image_channel_max = [np.max(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.feature.register_translation(image_channel_max[0], image_channel_max[i])[0] for i in range(1,len(image_stack))]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape[0]
    for i in range(len(image_stack)):
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        if np.abs(shift_row) > 15:
            shift_row = 0
        if np.abs(shift_col) > 15:
            shift_col = 0
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
    shift_filter_mask_final = np.prod(shift_filter_mask, axis = 0)
    image_registered = np.dstack(image_registered)*shift_filter_mask_final[:,:,None]
    image_cn = np.sum(image_registered, axis = 2)
    image_cn = np.log(np.sum(image_registered, axis = 2)+1e-2)
    rough = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_cn.reshape(image_cn.shape[0]*image_cn.shape[1],1))
    rough_seg = rough.reshape(image_cn.shape)
    image0 = image_cn*(rough_seg == 0)
    image1 = image_cn*(rough_seg == 1)
    i0 = np.average(image0[rough_seg == 0])
    i1 = np.average(image1[rough_seg == 1])
    if (i0 < i1):
        rough_seg_mask = (rough_seg == 1)
        bkg = (rough_seg == 0)
    else:
        rough_seg_mask = (rough_seg == 0)
        bkg = (rough_seg == 1)
    layers = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_cn.reshape(image_cn.shape[0]*image_cn.shape[1],1))
    cell_interior = layers.reshape(image_cn.shape)
    cell_interior_int_0 = skimage.measure.regionprops((cell_interior == 0)*1, intensity_image = image_cn)
    cell_interior_int_1 = skimage.measure.regionprops((cell_interior == 1)*1, intensity_image = image_cn)
    cell_interior_int_2 = skimage.measure.regionprops((cell_interior == 2)*1, intensity_image = image_cn)
    avgint_0 = [x.mean_intensity for x in cell_interior_int_0]
    avgint_1 = [x.mean_intensity for x in cell_interior_int_1]
    avgint_2 = [x.mean_intensity for x in cell_interior_int_2]
    layerint = [avgint_0[0], avgint_1[0], avgint_2[0]]
    index = np.argsort(layerint)[2]
    cell_interior_opening = skimage.morphology.binary_opening(skimage.morphology.remove_small_holes(cell_interior == index))
    cell_sm = skimage.morphology.remove_small_objects(cell_interior_opening, 50)
    cell_sm_label = skimage.morphology.label(cell_sm)
    dist_lab = skimage.morphology.label(cell_sm_label)
    markers = skimage.measure.regionprops(dist_lab)
    dist_be = np.zeros(dist_lab.shape)
    while(len(markers) > 0):
        for j in range(0,len(markers)):
            a = markers[j].area
            if (a < 600):
                dist_be[dist_lab == j+1] = 1
                dist_lab[dist_lab == j+1] = 0
        dist_bin_temp = skimage.morphology.binary_erosion(dist_lab)
        dist_bin_temp_sm = skimage.morphology.remove_small_objects(dist_bin_temp, 10)
        dist_lab = skimage.morphology.label(dist_bin_temp_sm)
        markers = skimage.measure.regionprops(dist_lab)
    dist_final = skimage.morphology.label(skimage.morphology.remove_small_objects(skimage.morphology.label(dist_be), 10))
    watershed_seeds = skimage.morphology.label(dist_final)
    segmentation = skimage.morphology.watershed(-image_cn, watershed_seeds, mask = rough_seg_mask)
    segmentation_sm = skimage.morphology.remove_small_objects(segmentation, 100)
    segmentation_smbc = skimage.segmentation.clear_border(segmentation_sm)
    cells = skimage.measure.regionprops(segmentation_smbc)
    segmentation_final = np.zeros(segmentation_smbc.shape).astype(int)
    for i in range(len(cells)):
        minor_axis_length = cells[i].minor_axis_length
        area = cells[i].area
        cell_seg_image = (segmentation_smbc == cells[i].label)
        cell_seg_image_be = skimage.morphology.binary_erosion(skimage.morphology.binary_erosion(cell_seg_image))
        if minor_axis_length < 15 or minor_axis_length > 35:
            segmentation_final[segmentation_smbc == cells[i].label] = 0
        else:
            segmentation_final[cell_seg_image_be] = cells[i].label
    return(segmentation_final, image_registered)

def measure_reference_images(image_name, cal_toggle, calibration_norm):
    sample = re.sub('_[0-9]*.czi', '', image_name[0])
    print('Analyzing sample {}...'.format(sample))
    image_stack = [bioformats.load_image(filename) for filename in image_name]
    segmentation, image_stack = segment_images(image_stack)
    if cal_toggle == 'T':
        image_ffc = correct_images(image_stack, calibration_norm)
    else:
        image_ffc = image_stack.copy()
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_ffc.shape[2]))
    for k in range(0, image_ffc.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_ffc[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    save_segmentation(segmentation, sample)
    avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    plot_avg_int_reference(avgint_norm, sample)
    avgintfilename = sample + '_avgint.csv'
    avgintnormfilename = sample + '_avgint_norm.csv'
    np.savetxt(avgintfilename, avgint, delimiter = ',')
    np.savetxt(avgintnormfilename, avgint_norm, delimiter = ',')
    return(segmentation, avgint)

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('-i', '--image_name', dest = 'image_name', nargs = '*', default = [], type = str, help = 'Input image names')
    parser.add_argument('-c', '--calibration', dest = 'cal_toggle', type = str, default = 'T', help = 'Toggle switch to calibrate images to correct for non-uniform illumination field, default is True')
    parser.add_argument('-cf', '--calibration_images_filename', dest = 'calibration_images_filename', type = str, default = '', help = 'Calibration image filename')
    args = parser.parse_args()
    if args.cal_toggle == 'T':
        calibration_norm = load_calibration_images(args.calibration_images_filename)
    else:
        calibration_norm = 0
    segmentation, avgint = measure_reference_images(args.image_name, args.cal_toggle, calibration_norm)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
