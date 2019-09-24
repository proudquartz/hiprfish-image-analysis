
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
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import measure

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def classify_spectra(input_spectra, umap_transform, scaler, clf_umap, clf):
    sample = re.sub('_avgint_norm.csv', '', input_spectra)
    avgint = pd.read_csv(input_spectra)
    segmentation = np.load('{}_seg.npy'.format(sample))
    avgint_norm = avgint.values/np.max(avgint.values, axis = 1)[:,None]
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
    cells = skimage.measure.regionprops(segmentation)
    cell_info[69] = np.asarray([x.label for x in cells])
    cell_info[70] = np.asarray([x.centroid[0] for x in cells])
    cell_info[71] = np.asarray([x.centroid[1] for x in cells])
    cell_info[72] = np.asarray([x.major_axis_length for x in cells])
    cell_info[73] = np.asarray([x.minor_axis_length for x in cells])
    cell_info[74] = np.asarray([x.eccentricity for x in cells])
    cell_info[75] = np.asarray([x.orientation for x in cells])
    cell_info[76] = np.asarray([x.area for x in cells])
    cellinfofilename = '{}_cell_information.csv'.format(sample)
    cell_info.to_csv(cellinfofilename, index = None, header = None)
    return

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-i', '--input_spectra', dest = 'input_spectra', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()
    umap_transform = joblib.load(args.ref_clf)
    scaler = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_scaler.pkl', args.ref_clf))
    clf_umap = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_svc.pkl', args.ref_clf))
    clf = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_check_svc.pkl', args.ref_clf))
    classify_spectra(args.input_spectra, umap_transform, scaler, clf_umap, clf)
    return

if __name__ == '__main__':
    main()
