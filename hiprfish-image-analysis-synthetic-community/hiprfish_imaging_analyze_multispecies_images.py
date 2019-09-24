
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from ete3 import NCBITaxa
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

ncbi = NCBITaxa()

def cm_to_inches(length):
    return(length*0.393701)

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def summarize_error_rate(input_folder, probe_design_filename):
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8.75), cm_to_inches(7.25))
    gs = GridSpec(2, 1)
    color_list = ['darkviolet', 'dodgerblue', 'orangered']
    encoding_set = ['B', 'C', 'A']
    labels = ['Least Complex', 'Most Complex', 'Random']
    for k in range(3):
        enc_set = encoding_set[k]
        filenames = glob.glob('{}/*_{}_*_cell_information.csv'.format(input_folder, enc_set))
        filenames.sort()
        samples = [re.sub('_cell_information.csv', '', f) for f in filenames]
        probes = pd.read_csv(probe_design_filename[k], dtype = {'code': str})
        multispecies_summary = probes.loc[:,['target_taxon', 'code']].drop_duplicates().reset_index().drop(columns = ['index'])
        multispecies_summary['ErrorRate'] = 0
        multispecies_summary['UpperLimit'] = 0
        taxid_sciname = pd.DataFrame.from_dict({564: 'E. coli',
                    1718: 'C. glutamicum',
                    1590: 'L. plantarum',
                    140100: 'V. albensis',
                    1580: 'L. brevis',
                    438: 'A. plantarum',
                    104102: 'A. tropicalis',
                    108981: 'A. schindleri',
                    285: 'C. testosteroni',
                    1353: 'E. gallinarum',
                    56459: 'X. vasicola'}, orient = 'index').reset_index()
        taxid_sciname.columns = ['target_taxon', 'sci_name']
        multispecies_summary = multispecies_summary.merge(taxid_sciname, on = 'target_taxon', how = 'left')
        taxid_list = [re.sub('_', '', re.sub('_fov_1', '', re.search('_.[0-9]*_fov_1', f).group(0))) for f in filenames]
        hamming_distance_list = []
        for i in range(len(samples)):
            s = samples[i]
            cell_info = pd.read_csv('{}_cell_information.csv'.format(s), header = None, dtype = {67: str})
            cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
            cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
            cell_info['Sample'] = s
            barcode_assignment = multispecies_summary.loc[multispecies_summary.target_taxon.values == int(taxid_list[i]), 'code'].values[0]
            max_intensity_mode = stats.mode(cell_info.max_intensity.values, axis = None)[0][0]
            error_rate = 1 - np.sum(cell_info.iloc[:,67].values == barcode_assignment)/cell_info.shape[0]
            cell_info['hamming_distance'] = cell_info[67].apply(hamming2, args = (barcode_assignment,))
            cell_info = cell_info.loc[cell_info.max_intensity.values > 0.75*max_intensity_mode, :]
            hamming_distance_list.append(cell_info.hamming_distance.values)
            multispecies_summary.loc[multispecies_summary.target_taxon.values == int(taxid_list[i]), 'samples'] = s
            if error_rate > 0:
                multispecies_summary.loc[multispecies_summary.target_taxon.values == int(taxid_list[i]), 'ErrorRate'] = error_rate
            else:
                multispecies_summary.loc[multispecies_summary.target_taxon.values == int(taxid_list[i]), 'ErrorRate'] = 1/cell_info.shape[0]
                multispecies_summary.loc[multispecies_summary.target_taxon.values == int(taxid_list[i]), 'UpperLimit'] = 1
        multispecies_summary = multispecies_summary.sort_values(['samples'], ascending = [samples])
        ax = plt.subplot(gs[0,0])
        ax.plot(np.arange(11)[multispecies_summary.UpperLimit.values == 0], multispecies_summary.loc[multispecies_summary.UpperLimit.values == 0, 'ErrorRate'], 'o', alpha = 0.8, markersize = 4, color = color_list[k], marker = 'o', markeredgewidth = 0)
        ax.plot(np.arange(11)[multispecies_summary.UpperLimit.values == 1], multispecies_summary.loc[multispecies_summary.UpperLimit.values == 1, 'ErrorRate'], 'o', alpha = 0.8, markersize = 4, color = color_list[k], marker = 'v', markeredgewidth = 0)
        ax = plt.subplot(gs[1,0])
        parts = ax.violinplot(hamming_distance_list, np.arange(1 + (k-1)*0.1, 12 + (k-1)*0.1, 1), showmedians = False, showmeans = True, showextrema = False, bw_method = 0.2, widths = 0.5, points = 100)
        for pc in parts['bodies']:
            pc.set_facecolor(color_list[k])
            pc.set_edgecolor(color_list[k])
            pc.set_linewidth(0.5)
            pc.set_alpha(0.8)
        parts['cmeans'].set_color(color_list[k])
        parts['cmeans'].set_linewidth(0.5)
    ax = plt.subplot(gs[0,0])
    plt.yscale('log')
    plt.ylabel('Error Rate', fontsize = 8, color = 'black')
    plt.tick_params(labelsize = 8, direction = 'in', colors = 'black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.xticks([])
    plt.ylim(1e-5,1)
    ax = plt.subplot(gs[1,0])
    patches = [mpatches.Patch(alpha = 0), mpatches.Patch(alpha = 0), mpatches.Patch(alpha = 0)]
    l = ax.legend(patches, labels, loc = 2, fontsize = 8, framealpha = 0, bbox_to_anchor = (-0.065,-0.065,1.1,1.1), bbox_transform = ax.transAxes)
    for k in range(3):
        l.get_texts()[k].set_color(color_list[k])
    plt.xticks(np.arange(1,12), multispecies_summary.sci_name.values, rotation = 30, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = 8, color = 'black', style = 'italic')
    plt.tick_params(labelsize = 8, direction = 'in', colors = 'black')
    plt.ylabel('Hamming distance', fontsize = 8, labelpad = 15, color = 'black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.2)
    plt.savefig('{}/multispecies_error_rate.pdf'.format(input_folder), dpi = 300, transparent = True)
    plt.close()
    return

def plot_representative_cell_image(input_folder, probe_design_filename):
    encoding_set = ['A', 'B', 'C']
    for k in range(3):
        enc_set = encoding_set[k]
        filenames = glob.glob('{}/*_{}_*_cell_information.csv'.format(input_folder, enc_set))
        samples = list(set([re.sub('_lp', '', re.sub('_cell_information.csv', '', f)) for f in filenames]))
        samples.sort()
        sci_name = {564: 'E. coli',
                    1718: 'C. glutamicum',
                    1590: 'L. plantarum',
                    140100: 'V. albensis',
                    1580: 'L. brevis',
                    438: 'A. plantarum',
                    104102: 'A. tropicalis',
                    108981: 'A. schindleri',
                    285: 'C. testosteroni',
                    1353: 'E. gallinarum',
                    56459: 'X. vasicola'}
        cell_info_list = [pd.read_csv(f, header = None, dtype = {67: str}) for f in filenames]
        tile_info = np.zeros((len(samples),2))
        probes = pd.read_csv(probe_design_filename[k], dtype = {'code': str})
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(9), cm_to_inches(7))
    gs = GridSpec(11, 6)
    for k in range(3):
        enc_set = encoding_set[k]
        filenames = glob.glob('{}/*_{}_*_cell_information.csv'.format(input_folder, enc_set))
        samples = list(set([re.sub('_lp', '', re.sub('_cell_information.csv', '', f)) for f in filenames]))
        samples.sort()
        ordered_taxid_list = [108981, 140100, 56459, 104102, 1580, 1590, 1353, 438, 1718, 285, 564]
        for i in range(len(samples)):
            s = samples[i]
            taxid = int(re.sub('_', '', re.sub('_fov_1', '', re.search('_[0-9]*_fov_1', s).group(0))))
            cell_information = pd.read_csv('{}_cell_information.csv'.format(s), header = None, dtype = {67: str})
            spec_average = np.average(cell_information.loc[:,0:62].values, axis = 0)
            spec_std = np.std(cell_information.loc[:,0:62].values, axis = 0)
            ax = plt.subplot(gs[i, 2*k:2*k+2])
            ax.errorbar(np.arange(0,23), spec_average[0:23], yerr = spec_std[0:23], color = 'limegreen', fmt = '-o', markersize = 0.1, ecolor = 'w', capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(23,43), spec_average[23:43], yerr = spec_std[23:43], color = 'yellowgreen', fmt = '-o', markersize = 0.1, ecolor = 'w', capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(43,57), spec_average[43:57], yerr = spec_std[43:57], color = 'darkorange', fmt = '-o', markersize = 0.1, ecolor = 'w', capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(57,63), spec_average[57:63], yerr = spec_std[57:63], color = 'red', fmt = '-o', markersize = 0.1, ecolor = 'w', capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.tick_params(colors = 'black', direction = 'in', labelsize = 6)
            if k == 0:
                ax.axes.get_yaxis().set_visible(True)
                plt.yticks([])
                plt.ylabel(sci_name[taxid], rotation = 0, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = 6, fontstyle = 'italic', color = 'black')
    ax = plt.subplot(gs[10,0:2])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    plt.yticks([0,1])
    plt.xticks([0,20,40,60])
    plt.xlabel('Channel', fontsize = 6)
    ax = plt.subplot(gs[0,0:2])
    ax.axes.get_xaxis().set_visible(True)
    plt.tick_params(axis = 'x', bottom = False, color = 'black')
    plt.xticks([])
    ax.xaxis.set_label_position('top')
    plt.xlabel('Random', fontsize = 6, color = 'black')
    ax = plt.subplot(gs[0,2:4])
    ax.axes.get_xaxis().set_visible(True)
    plt.tick_params(axis = 'x', bottom = False, color = 'black')
    plt.xticks([])
    ax.xaxis.set_label_position('top')
    plt.xlabel('Least Complex', fontsize = 6, color = 'black')
    ax = plt.subplot(gs[0,4:6])
    ax.axes.get_xaxis().set_visible(True)
    plt.tick_params(axis = 'x', bottom = False, color = 'black')
    plt.xticks([])
    ax.xaxis.set_label_position('top')
    plt.xlabel('Most Complex', fontsize = 6, color = 'black')
    plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.1)
    plt.savefig('{}/multispecies_representative_cell_spectra.pdf'.format(input_folder), dpi = 300, transparent = True)
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser('Summarize multispecies synthetic community measurement results')
    parser.add_argument('input_folder', type = str, help = 'Input folder containing image analysis results')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, nargs = '*', help = 'Probe design filenames')
    args = parser.parse_args()
    summarize_error_rate(args.input_folder, args.probe_design_filename)
    plot_representative_cell_image(args.input_folder, args.probe_design_filename)
    return

if __name__ == '__main__':
    main()
