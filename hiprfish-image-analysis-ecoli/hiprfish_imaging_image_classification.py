
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
import umap
import glob
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

def classify_images(avgint_filename, ref_clf):
    sample = re.sub('_avgint.csv', '', avgint_filename)
    print('Classifying sample {}...'.format(sample))
    segmentation = np.load('{}_seg.npy'.format(sample))
    avgint = pd.read_csv(avgint_filename, header = None)
    avgint_norm = avgint.values/np.max(avgint.values, axis = 1)[:,None]
    umap_transform = joblib.load(ref_clf)
    clf_umap = joblib.load(re.sub('transform.pkl', 'transformed_svc.pkl', ref_clf))
    clf = joblib.load(re.sub('transform.pkl', 'transformed_check_svc.pkl', ref_clf))
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 37))), axis = 1)
    avgint_norm[:,95:126] = np.diff(avgint_norm[:,0:32], axis = 1)
    avgint_norm[:,126] = clf[0].predict(avgint_norm[:,0:32])
    avgint_norm[:,127] = clf[1].predict(avgint_norm[:,32:55])
    avgint_norm[:,128] = clf[2].predict(avgint_norm[:,55:75])
    avgint_norm[:,129] = clf[3].predict(avgint_norm[:,75:89])
    avgint_norm[:,130] = clf[4].predict(avgint_norm[:,89:95])
    avgint_norm[:,131] = clf[5].predict(avgint_norm[:,95:126])
    avgint_umap_transformed = umap_transform.transform(avgint_norm)
    cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    cell_ids_norm_filename = sample + '_cell_ids.txt'
    avgint_identification_filename = sample + '_avgint_ids.csv'
    avgint_identification = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None]), axis = 1))
    avgint_identification[133] = sample
    cells = skimage.measure.regionprops(segmentation)
    avgint_identification[134] = np.asarray([x.label for x in cells])
    np.savetxt(cell_ids_norm_filename, cell_ids_norm, fmt = '%s')
    avgint_identification.to_csv(avgint_identification_filename, header = None, index = None)
    ids = list(set(cell_ids_norm))
    image_identification = np.zeros(segmentation.shape)
    for q in range(0, len(ids)):
        cell_population = np.where(cell_ids_norm == ids[q])[0]
        for r in range(0, len(cell_population)):
            image_identification[segmentation == cell_population[r]+1] = int(ids[q], 2)
    save_identification(image_identification, sample)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('input_spectra', type = str, default = '', help = 'Average normalized single cell spectra filenname')
    parser.add_argument('-rf', '--reference_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()
    classify_images(args.input_spectra, args.ref_clf)
    return

if __name__ == '__main__':
    main()
