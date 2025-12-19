#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 17 4:27 PM 2022
Created in PyCharm
Created as QGP_Scripts/analyze_binom_slice_plotter

@author: Dylan Neff, Dyn04
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.optimize import basinhopping
import warnings

import itertools
import inspect

from multiprocessing import Pool
import tqdm
import istarmap  # Needed for tqdm

from analyze_binom_slices import *



def main():
    plot_qa_example()
    # plot_paper_figs()
    # plot_qm_figs()
    # plot_method_paper_figs()

    # plot_star_var_sys()
    # make_models_csv()

    # plot_lyons_example()
    # plot_star_analysis_note_figs()

    # plot_raw_mix_sys_comp()
    print('donzo')


def plot_qa_example():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144

    # base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # v2_star_in_dir = 'F:/Research/Data/default/' \
    #                  'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    # df_path = f'{base_path}Binomial_Slice_Moments/binom_slice_vars_bes_sys.csv'
    # cent_ref_path = 'F:/Research/Results/Azimuth_Analysis/mean_cent_ref.csv'

    base_path = '/star/u/dneff/gpfs/'
    v2_star_in_dir = '/star/u/dneff/gpfs/tree_reader_data/Data/default/' \
                     'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    df_path = f'{base_path}Binomial_Slice_Moments/binom_slice_stats.csv'
    cent_ref_path = f'{base_path}Binomial_Slice_Moments/mean_cent_ref.csv'

    plot = True
    threads = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 180
    divs_all = [60, 72, 89, 90, 120, 180, 240, 270, 288, 300]
    cent_plt = 7
    cents = [1, 2, 3, 4, 5, 6, 7, 8]
    energies_fit = [11]
    samples = 72  # For title purposes only

    cent_ref_df = pd.read_csv(f'{cent_ref_path}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    data_sets_plt = ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'green', 'black', 'red', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']))

    df = pd.read_csv(df_path)
    df = df.dropna()
    df['name'] = df['name'].str.replace('bes_sys_', '')
    all_sets = pd.unique(df['name'])

    # For all sets where 'ampt' is not in the name, make a copy of all 'bes_def' rows in cent_ref_df for this data_set
    cent_ref_extra_sets = []
    for data_set_name in all_sets:
        print(data_set_name)
        if 'ampt' not in data_set_name and data_set_name != 'bes_def':
            cent_ref_extra_sets.append(cent_ref_df[cent_ref_df['data_set'] == 'bes_def'].assign(data_set=data_set_name))
    if len(cent_ref_extra_sets) > 0:
        cent_ref_extra_sets = pd.concat(cent_ref_extra_sets)
        cent_ref_df = pd.concat([cent_ref_df, cent_ref_extra_sets])

    v2_star_vals = {2: read_flow_values(v2_star_in_dir)}
    v2_sys_vals = {}
    v2_sys_vals.update({'bes_def': v2_star_vals})

    df = df[df['stat'] == stat_plot]

    # Calculate dsigma with k2 values and get systematics
    df = df[df['stat'] == 'k2']
    df = df.drop('stat', axis=1)
    print('Calc dsigma')
    df_raw, df_mix, df_diff = calc_dsigma(df, ['raw', 'mix', 'diff'])
    df_dsigma_types = pd.concat([df_raw, df_mix, df_diff])
    print('Calc diff nlo error')
    df_dsigma_types = add_diff_nlo_err(df_dsigma_types, group_cols=['energy', 'cent', 'name', 'total_protons'],
                                       exclude_divs=[356, 89])

    dvar_vs_protons(df_dsigma_types, div_plt, cent_plt, [11], ['raw', 'mix', 'diff'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.0015, 0.00085], print_data=True,
                    data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
                                                  'diff': 'Single Event - Mixed Event'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo
    dvar_vs_protons(df_dsigma_types, div_plt, 4, [11], ['raw', 'mix', 'diff'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.003, 0.00095],
                    data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
                                                  'diff': 'Single Event - Mixed Event'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo

    sets_run = all_sets if plot else ['bes_def']
    dsig_avgs_all, dsig_avgs_diff_v2sub = [], []
    print(sets_run)
    for energy in energies_fit:
        print(f'Energy {energy}GeV')
        jobs = [(df_dsigma_types[df_dsigma_types['name'] == set_i], divs_all, cents, energy, ['raw', 'mix', 'diff'],
                 [set_i], None, False, True) for set_i in sets_run]
        dsig_avgs_div_all = []
        with Pool(threads) as pool:
            for job_i in tqdm.tqdm(pool.istarmap(dvar_vs_protons_cents, jobs), total=len(jobs)):
                dsig_avgs_div_all.append(job_i)
        dsig_avgs_div_all = pd.concat(dsig_avgs_div_all, ignore_index=True)
        dsig_avgs_all.append(dsig_avgs_div_all)
        dsig_avgs_div_diff = dsig_avgs_div_all[dsig_avgs_div_all['data_type'] == 'diff']
        dsig_avgs_div_diff = dsig_avgs_div_diff.drop('data_type', axis=1)
        for data_set in sets_run:
            dsig_avgs_div_diff_set = subtract_dsigma_flow(dsig_avgs_div_diff, data_set,
                                                          data_set, v2_sys_vals[data_set], new_only=True)
            dsig_avgs_diff_v2sub.append(dsig_avgs_div_diff_set)

    dsig_avgs_diff_v2sub = pd.concat(dsig_avgs_diff_v2sub, ignore_index=True)

    dsig_avgs_v2_sub_div120 = dsig_avgs_diff_v2sub[(dsig_avgs_diff_v2sub['divs'] == 120) & (dsig_avgs_diff_v2sub['cent'] > -1)]
    data_sets_energies_colors = \
        {'bes_def': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'}}
    vs_cent_sets = list(data_sets_energies_colors.keys())

    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5),
                              data_sets_energies_colors=data_sets_energies_colors,
                              print_data=True)
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5),
                              data_sets_energies_colors=data_sets_energies_colors)

    plt.show()


def plot_qm_figs():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144

    presentation_mode = True
    if presentation_mode:
        # plt.rcParams['axes.labelsize'] = 14  # Adjust the value as needed
        # plt.rcParams['axes.titlesize'] = 16  # Adjust the value as needed
        # plt.rcParams['legend.fontsize'] = 14  # Adjust the value as needed
        plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1

    # base_path = 'F:/Research/Results/Azimuth_Analysis/'
    base_path = 'C:/Users/Dyn04/OneDrive - personalmicrosoftsoftware.ucla.edu/OneDrive - UCLA IT Services/Research/UCLA/Results/Azimuth_Analysis/'
    df_dsigma_v2sub_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma_v2sub.csv'
    df_dsigma_v2sub_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma_v2sub.csv'
    df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_bes.csv'
    df_def_avgs_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_model.csv'
    df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_bes.csv'
    df_def_avgs_v2sub_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_model.csv'

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 120
    exclude_divs = [89, 356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 8
    energies_fit = [7, 11, 19, 27, 39, 62]
    samples = 72  # For title purposes only

    data_sets_plt = ['bes_def', 'ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'red', 'blue', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['STAR', 'AMPT', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$']))
    data_sets_markers = dict(zip(data_sets_plt, [dict(zip(['raw', 'mix', 'diff'], [x, x, x]))
                                                 for x in ['o', 's', '^', '*']]))
    data_sets_bands = ['ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    legend_order = ['STAR', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$', 'AMPT']

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    df_dsigma_v2sub = pd.read_csv(f'{base_path}{df_dsigma_v2sub_name}')
    df_dsigma_v2sub_model = pd.read_csv(f'{base_path}{df_dsigma_v2sub_model_name}')
    df_dsigma_v2sub_model = df_dsigma_v2sub_model[df_dsigma_v2sub_model['err'] < 0.0001]
    df_dsigma_v2sub = pd.concat([df_dsigma_v2sub, df_dsigma_v2sub_model])

    dvar_vs_protons(df_dsigma_v2sub, div_plt, cent_plt, [39], ['diff'], data_sets_plt, ylabel=r'$\Delta\sigma^2$',
                    data_sets_colors=data_sets_colors, plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009],
                    data_sets_labels=data_sets_labels, marker_map=data_sets_markers, legend_pos='lower right',
                    kin_info_loc=(0.22, 0.13), star_prelim_loc=(0.68, 0.35), data_sets_bands=data_sets_bands,
                    legend_order=legend_order, title=f'{cent_map[8]} Centrality, {div_plt}° Partitions',
                    xlim=(0, 43))  # <---

    dsig_avgs_all = pd.read_csv(f'{base_path}{df_def_avgs_out_name}')
    dsig_avgs_all_model = pd.read_csv(f'{base_path}{df_def_avgs_out_model_name}')
    dsig_avgs_all = pd.concat([dsig_avgs_all, dsig_avgs_all_model])

    dsig_avgs_v2sub = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_name}')
    dsig_avgs_v2sub_model = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_model_name}')
    dsig_avgs_v2sub = pd.concat([dsig_avgs_v2sub, dsig_avgs_v2sub_model])

    dsig_avgs_v2sub['data_type'] = 'diff'

    plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.2
    dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
                             plot=True, avg=False, plot_avg=False, data_sets_colors=data_sets_colors, no_hydro_label=1,
                             data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0003], avgs_df=dsig_avgs_v2sub,
                             ylabel=r'$\Delta\sigma^2$', ylim=(-0.0014, 0.00035),
                             # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
                             kin_loc=(0.58, 0.45), star_prelim_loc=(2, 0.55, 0.2),
                             marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
                             # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
                             title=f''
                             )  # <---

    dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
                             plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors, no_hydro_label=1,
                             data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0003], avgs_df=dsig_avgs_v2sub,
                             ylabel=r'$\Delta\sigma^2$', ylim=(-0.0014, 0.00035),
                             # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
                             kin_loc=(0.58, 0.45), star_prelim_loc=(2, 0.55, 0.2),
                             marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
                             # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
                             title=f''
                             )  # <---

    dsig_avgs_v2_sub_cent8_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['cent'] == 8) & (dsig_avgs_v2sub['divs'] == 120)]
    plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, data_sets_plt, data_sets_colors=data_sets_colors,  # <---
                                data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.73, 0.45),
                                star_prelim_loc=(0.02, 0.75), marker_map=data_sets_markers,
                                data_sets_bands=data_sets_bands, legend_order=legend_order,
                                # title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')
                                title=f'{cent_map[8]} Centrality, {div_plt}° Partitions')
    plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, ['bes_def'], data_sets_colors=data_sets_colors,
                                data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.123, 0.68),
                                star_prelim_loc=(0.65, 0.9), leg_loc='lower right',
                                title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')

    dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[dsig_avgs_v2sub['divs'] == 120]
    data_sets_energies_colors = \
        {'bes_def': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'},
         'ampt_new_coal_epbins1': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'}}
    vs_cent_sets = list(data_sets_energies_colors.keys())
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
                              data_sets_energies_colors=data_sets_energies_colors)
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
                              data_sets_energies_colors=data_sets_energies_colors)

    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5), marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5), marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)

    data_sets_cent = ['ampt_new_coal_epbins1', 'bes_def']
    legend_order = ['7.7 GeV', '11.5 GeV', '19.6 GeV', '27 GeV', '39 GeV', '62.4 GeV', 'AMPT Fit']
    plot_dsig_avg_vs_cent_2panel(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
                                 # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                 title='',
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    plot_dsig_avg_vs_cent_2panel2(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                  cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
                                  # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                  title='',
                                  errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                  kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.62, 0.4), marker_map=data_sets_markers,
                                  data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.3
    plot_dsig_avg_vs_cent_2panel3(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                  cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
                                  # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                  title='',
                                  errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                  kin_info_loc=(0.3, 0.1), star_prelim_loc=(0.25, 0.3), marker_map=data_sets_markers,
                                  data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    # plt.savefig('C:/Users/Dyn04/Desktop/test.pdf', format='pdf')
    plt.show()


def plot_paper_figs():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144

    presentation_mode = False
    if presentation_mode:
        # plt.rcParams['axes.labelsize'] = 14  # Adjust the value as needed
        # plt.rcParams['axes.titlesize'] = 16  # Adjust the value as needed
        # plt.rcParams['legend.fontsize'] = 14  # Adjust the value as needed
        plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.2
    # else:
    #     plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.1

    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # base_path = 'C:/Users/Dyn04/OneDrive - personalmicrosoftsoftware.ucla.edu/OneDrive - UCLA IT Services/Research/UCLA/Results/Azimuth_Analysis/'
    # base_path = 'C:/Users/Dyn04/Research/'
    df_name = 'Bes_with_Sys/binom_slice_vars_bes.csv'
    df_model_name = 'Bes_with_Sys/binom_slice_vars_model.csv'
    df_dsigma_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma.csv'
    df_dsigma_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma.csv'
    df_dsigma_v2sub_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma_v2sub.csv'
    df_dsigma_v2sub_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma_v2sub.csv'
    df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_bes.csv'
    df_def_avgs_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_model.csv'
    df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_bes.csv'
    df_def_avgs_v2sub_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_model.csv'
    df_partitions_fits_name = 'Bes_with_Sys/partition_width_fits_bes.csv'
    df_partitions_fits_model_name = 'Bes_with_Sys/partition_width_fits_model.csv'

    save_dir = 'C:/Users/Dylan/OneDrive - UCLA IT Services/Research/UCLA/Presentations/STAR_Paper/Autogen_Figures/'

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 120
    exclude_divs = [89, 356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 8
    energies_fit = [7, 11, 19, 27, 39, 62]
    samples = 72  # For title purposes only

    data_sets_plt = ['bes_def', 'ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'red', 'blue', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['STAR', 'AMPT', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$']))
    data_sets_markers = dict(zip(data_sets_plt, [dict(zip(['raw', 'mix', 'diff'], [x, x, x]))
                                                 for x in ['o', 's', '^', '*']]))
    data_sets_bands = ['ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    legend_order = ['STAR', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$', 'AMPT']

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    df = pd.read_csv(f'{base_path}{df_name}')
    df_model = pd.read_csv(f'{base_path}{df_model_name}')
    df = pd.concat([df, df_model])
    df = df[df['stat'] == stat_plot]

    df_dsigma = pd.read_csv(f'{base_path}{df_dsigma_name}')
    df_dsigma_model = pd.read_csv(f'{base_path}{df_dsigma_model_name}')
    df_dsigma = pd.concat([df_dsigma, df_dsigma_model])

    dvar_vs_protons(df_dsigma, div_plt, cent_plt, [39], ['raw', 'mix', 'diff'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055], print_data=True,
                    data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
                                                  'diff': 'Single Event - Mixed Event'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo
    dvar_vs_protons(df_dsigma, div_plt, 4, [39], ['raw', 'mix', 'diff'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055],
                    data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
                                                  'diff': 'Single Event - Mixed Event'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo

    dvar_vs_protons(df_dsigma, div_plt, cent_plt, [62], ['raw', 'mix', 'diff'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055],
                    data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
                                                  'diff': 'Single Event - Mixed Event'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo

    # dvar_vs_protons(df_dsigma, div_plt, cent_plt, [39], ['diff'], data_sets_plt, data_sets_colors=data_sets_colors,
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009], ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels=data_sets_labels, marker_map=data_sets_markers, legend_pos='lower right',
    #                 data_sets_bands=data_sets_bands)

    df_dsigma_v2sub = pd.read_csv(f'{base_path}{df_dsigma_v2sub_name}')
    df_dsigma_v2sub_model = pd.read_csv(f'{base_path}{df_dsigma_v2sub_model_name}')
    df_dsigma_v2sub_model = df_dsigma_v2sub_model[df_dsigma_v2sub_model['err'] < 0.0001]
    df_dsigma_v2sub = pd.concat([df_dsigma_v2sub, df_dsigma_v2sub_model])

    dvar_vs_protons(df_dsigma_v2sub, div_plt, cent_plt, [39], ['diff'], data_sets_plt, ylabel=r'$\Delta\sigma^2$',
                    data_sets_colors=data_sets_colors, plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009],
                    data_sets_labels=data_sets_labels, marker_map=data_sets_markers, legend_pos='lower right',
                    kin_info_loc=(0.22, 0.13), star_prelim_loc=(0.65, 0.96), data_sets_bands=data_sets_bands,
                    legend_order=legend_order)  # 39 GeV mix and v2 subtract dsig2

    df_dsigma_v2sub_diffs = df_dsigma_v2sub[df_dsigma_v2sub['data_type'] == 'diff'].assign(data_type='v2_sub')
    df_dsigma_with_v2sub = pd.concat([df_dsigma, df_dsigma_v2sub_diffs])
    dvar_vs_protons(df_dsigma_with_v2sub, div_plt, cent_plt, [39], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009],
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}})

    dvar_vs_protons(df_dsigma_with_v2sub, 90, 4, [62], ['raw', 'diff', 'v2_sub'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
                    data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected',
                                                  'v2_sub': 'Mixed and Flow Corrected'}},
                    data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
                    # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
                    y_ranges=[-0.0039, 0.0009], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 4, [39], ['raw', 'diff', 'v2_sub'], ['bes_def'],
                    plot=True, avg=False, alpha=1.0, ylabel=r'$\Delta\sigma^2$', print_data=True,
                    data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected',
                                                  'v2_sub': 'Mixed and Flow Corrected'}},
                    data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black'}},
                    marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
                    # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
                    y_ranges=[-0.0039, 0.0009], kin_info_loc=(0.26, 0.94))  # v2 sub demo

    # plt.show()

    dsig_avgs_all = pd.read_csv(f'{base_path}{df_def_avgs_out_name}')
    dsig_avgs_all_model = pd.read_csv(f'{base_path}{df_def_avgs_out_model_name}')
    dsig_avgs_all = pd.concat([dsig_avgs_all, dsig_avgs_all_model])

    # dvar_vs_protons_energies(df_dsigma, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], ['bes_def'],
    #                          plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_all,
    #                          ylabel=r'$\Delta\sigma^2$', kin_loc=(0.65, 0.2), legend_order=legend_order)

    dsig_avgs_v2sub = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_name}')
    dsig_avgs_v2sub_model = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_model_name}')
    dsig_avgs_v2sub = pd.concat([dsig_avgs_v2sub, dsig_avgs_v2sub_model])

    dsig_avgs_v2sub['data_type'] = 'diff'
    print(df_dsigma_v2sub.columns)
    df_dsigma_v2sub = df_dsigma_v2sub[df_dsigma_v2sub['err'] < 0.0001]
    # dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], ['bes_def'],
    #                          plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_v2sub,
    #                          ylabel=r'$\Delta\sigma^2$',
    #                          # kin_loc=(0.55, 0.6), star_prelim_loc=(1, 0.54, 0.53)
    #                          kin_loc=(0.58, 0.45), star_prelim_loc=(1, 0.54, 0.4)
    #                          )

    # dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
    #                          plot=True, avg=False, plot_avg=False, data_sets_colors=data_sets_colors, no_hydro_label=1,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_v2sub,
    #                          ylabel=r'$\Delta\sigma^2$',
    #                          # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
    #                          kin_loc=(0.58, 0.45), star_prelim_loc=(1, 0.54, 0.4),
    #                          marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
    #                          # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
    #                          title=f''
    #                          )  # <---

    dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
                             plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors, no_hydro_label=1,
                             data_sets_labels=data_sets_labels, y_ranges=[-0.00124, 0.00033], avgs_df=dsig_avgs_v2sub,
                             ylabel=r'$\Delta\sigma^2$', print_data=True,
                             # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
                             kin_loc=(0.58, 0.43), star_prelim_loc=(1, 0.54, 0.37),
                             marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
                             # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
                             title=f''
                             )  # <---

    # plot_protons_avgs_vs_energy(dsig_avg, ['bes_def'], data_sets_colors=data_sets_colors,
    #                             data_sets_labels=data_sets_labels, title=f'{cent_map[cent_plt]} Centrality, {div_plt}° '
    #                                                                      f'Partitions, {samples} Samples per Event')

    dsig_avgs_v2_sub_cent8 = dsig_avgs_v2sub[dsig_avgs_v2sub['cent'] == 8]
    # plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, data_sets_plt, data_sets_colors=data_sets_colors, fit=False,  # <---
    #                     data_sets_labels=data_sets_labels, plot_energy_panels=True, legend_order=legend_order,
    #                     ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_bands=data_sets_bands,
    #                     plot_indiv=False, ylim=(-0.00079, 0.0001), leg_panel=5, no_hydro_label=True,
    #                     # star_prelim_loc=(1, 0.3, 0.7),
    #                     star_prelim_loc=(1, 0.5, 0.8),
    #                     # xlim=(-10, 370), title=f'0-5% Centrality, {samples} Samples per Event',
    #                     xlim=(-10, 370), title=f'',
    #                     exclude_divs=exclude_divs)
    plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, data_sets_plt, data_sets_colors=data_sets_colors, fit=True,  # <---
                        data_sets_labels=data_sets_labels, plot_energy_panels=True, legend_order=legend_order,
                        ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_bands=data_sets_bands, print_data=True,
                        plot_indiv=False, ylim=(-0.0009, 0.0001), leg_panel=5, no_hydro_label=True,
                        # star_prelim_loc=(1, 0.3, 0.7),
                        star_prelim_loc=(1, 0.5, 0.8),
                        # xlim=(-10, 370), title=f'0-5% Centrality, {samples} Samples per Event',
                        xlim=(-10, 370), title=f'',
                        exclude_divs=exclude_divs)
    # plt.show()

    # plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                     data_sets_labels=data_sets_labels, plot_energy_panels=True,
    #                     ylab=r'$\langle\Delta\sigma^2\rangle$',
    #                     plot_indiv=False, ylim=(-0.00079, 0.00019), leg_panel=5,
    #                     # star_prelim_loc=(1, 0.3, 0.7),
    #                     star_prelim_loc=(1, 0.5, 0.8),
    #                     xlim=(-10, 370), title='',
    #                     # title=f'0-5% Centrality, {samples} Samples per Event',
    #                     exclude_divs=exclude_divs)

    dsig_avgs_v2_sub_cent8_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['cent'] == 8) & (dsig_avgs_v2sub['divs'] == 120)]
    plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, data_sets_plt, data_sets_colors=data_sets_colors,  # <---
                                data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.02, 0.595),
                                star_prelim_loc=(0.25, 0.32), marker_map=data_sets_markers, ylim=(-0.00066, 0.00002),
                                data_sets_bands=data_sets_bands, legend_order=legend_order, leg_loc='lower right',
                                # title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')
                                title=f'{cent_map[8]} Centrality, {div_plt}° Partitions')
    # plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, ['bes_def'], data_sets_colors=data_sets_colors,
    #                             data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.123, 0.68),
    #                             star_prelim_loc=(0.65, 0.9), leg_loc='lower right',
    #                             title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')

    # dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[dsig_avgs_v2sub['divs'] == 120]
    dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['divs'] == 120) & (dsig_avgs_v2sub['cent'] > -1)]
    data_sets_energies_colors = \
        {'bes_def': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'},
         'ampt_new_coal_epbins1': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'}}
    vs_cent_sets = list(data_sets_energies_colors.keys())
    # plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                           data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
    #                           title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
    #                           kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
    #                           data_sets_energies_colors=data_sets_energies_colors)
    # plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                           data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
    #                           title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
    #                           kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
    #                           data_sets_energies_colors=data_sets_energies_colors)

    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5), marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands,
                              print_data=True)
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.3, 0.5), marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)

    data_sets_cent = ['ampt_new_coal_epbins1', 'bes_def']
    legend_order = ['7.7 GeV', '11.5 GeV', '19.6 GeV', '27 GeV', '39 GeV', '62.4 GeV', 'AMPT Fit']
    plot_dsig_avg_vs_cent_2panel(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
                                 # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                 title='',
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    # plot_dsig_avg_vs_cent_2panel2(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
    #                               cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
    #                               # title=f'{div_plt}° Partitions, {samples} Samples per Event',
    #                               title='',
    #                               errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
    #                               kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
    #                               data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    plot_dsig_avg_vs_cent_2panel62ref(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors,
                                      fit=False, cent_ref=cent_ref_df, ref_type=ref_type, legend_order=None,
                                      title='', errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8, kin_info_loc=(0.2, 0.75),
                                      star_prelim_loc=None, marker_map=data_sets_markers,
                                      data_sets_energies_colors=data_sets_energies_colors,
                                      data_sets_bands=data_sets_bands)

    cent_fits = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors,
                                          fit=True, cent_ref=cent_ref_df, ref_type=ref_type, legend_order=None,
                                          title='', errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8, kin_info_loc=(0.2, 0.75),
                                          star_prelim_loc=None, marker_map=data_sets_markers,
                                          data_sets_energies_colors=data_sets_energies_colors,
                                          data_sets_bands=None)
    print(cent_fits)
    # plt.show()

    dsig_avgs_62ref = dsig_avgs_v2_sub_div120.rename(columns={'name': 'data_set'})
    plot_div_fits_vs_cent_62res(dsig_avgs_62ref, data_sets_cent, data_sets_colors, data_sets_labels, ref_type=ref_type,
                                cent_ref=cent_ref_df, val_col='avg', err_col='avg_err')

    df_fits = pd.read_csv(f'{base_path}{df_partitions_fits_name}')
    df_fits_model = pd.read_csv(f'{base_path}{df_partitions_fits_model_name}')
    df_fits = pd.concat([df_fits, df_fits_model])

    # data_sets_energies_cmaps = dict(zip(data_sets_cent, ['winter', 'copper']))
    # data_sets_markers2 = dict(zip(data_sets_cent, ['s', 'o']))
    # plot_div_fits_vs_cent(df_fits, data_sets_cent,  # data_sets_energies_cmaps=data_sets_energies_cmaps,
    #                       data_sets_labels=data_sets_labels, title=None, fit=False, cent_ref=cent_ref_df,
    #                       ref_type=ref_type, data_sets_colors=data_sets_energies_colors,
    #                       data_sets_markers=data_sets_markers2)

    plot_div_fits_vs_cent_2panel(df_fits, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                                 # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                 title='', print_data=True,
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.03), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors,
                                 # data_sets_bands=data_sets_bands
                                 )
    plot_baseline_vs_cent_fit(df_fits, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                                 title='', ls='',
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.03), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors,
                                 )
    plt.show()

    # plot_slope_div_fits(df_fits, data_sets_colors, data_sets_labels, data_sets=data_sets_plt)
    # plot_slope_div_fits_simpars(df_fits)

    # Plot avg dsig2 vs refmult for mixed events. Wierd stuff at most peripheral bin or two

    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 0, [7], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=True, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected', 'mix': 'Mixed',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black', 'mix': 'purple'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.017, 0.007], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    #
    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 2, [7], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=True, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected', 'mix': 'Mixed',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black', 'mix': 'purple'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.017, 0.007], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    #
    # df_mix = []
    # for cent in range(0, 9):
    #     df_mix_cent = dvar_vs_protons(df_dsigma, div_plt, cent, [7, 11, 19, 27, 39, 62],
    #                                   ['mix'], data_sets_plt, plot=False, avg=True)
    #     print(df_mix_cent)
    #     df_mix.append(df_mix_cent)
    # print(df_mix)
    # df_mix = pd.concat(df_mix)
    # print(df_mix)
    # print(df_mix.columns)
    # df_mix = df_mix[(df_mix['divs'] == 120) & (df_mix['data_type'] == 'mix')]
    # plot_dsig_avg_vs_cent_2panel(df_mix, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
    #                              cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
    #                              # title=f'{div_plt}° Partitions, {samples} Samples per Event',
    #                              title='',
    #                              errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
    #                              kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
    #                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)

    # Save all open figures
    for i in plt.get_fignums():
        plt.figure(i)
        window_title = plt.gcf().canvas.manager.get_window_title().replace(' ', '_')
        plt.savefig(f'{save_dir}{window_title}_{i}.png')
        plt.savefig(f'{save_dir}{window_title}_{i}.pdf')

    plt.show()


def plot_method_paper_figs():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144

    presentation_mode = False
    if presentation_mode:
        # plt.rcParams['axes.labelsize'] = 14  # Adjust the value as needed
        # plt.rcParams['axes.titlesize'] = 16  # Adjust the value as needed
        # plt.rcParams['legend.fontsize'] = 14  # Adjust the value as needed
        plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.2

    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # base_path = 'C:/Users/Dyn04/Research/'
    df_model_name = 'Bes_with_Sys/binom_slice_vars_model.csv'
    df_dsigma_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma.csv'
    df_dsigma_v2sub_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma_v2sub.csv'
    # df_def_avgs_v2sub_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_model.csv'
    df_def_avgs_v2sub_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_raw_model.csv'
    # df_partitions_fits_model_name = 'Bes_with_Sys/partition_width_fits_model.csv'
    df_partitions_fits_model_name = 'Bes_with_Sys/partition_width_fits_raw_model.csv'
    df_partitions_fits_model_nov2sub_name = 'Bes_with_Sys/partition_width_fits_nov2sub_model.csv'

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 120
    exclude_divs = [89, 356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 8
    data_type_plt = 'diff'  # 'raw' 'diff'
    samples = 72  # For title purposes only

    data_sets_plt = ['ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    data_sets_colors = dict(zip(data_sets_plt, ['red', 'blue', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['AMPT', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$']))
    data_sets_markers = dict(zip(data_sets_plt, [dict(zip(['raw', 'mix', 'diff'], [x, x, x]))
                                                 for x in ['o', 's', '^', '*']]))
    data_sets_bands = []
    legend_order = ['MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$', 'AMPT']

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    df = pd.read_csv(f'{base_path}{df_model_name}')
    df = df[df['stat'] == stat_plot]

    df_dsigma = pd.read_csv(f'{base_path}{df_dsigma_model_name}')

    # plot_method_paper_event()

    # plt.rcParams.update({'font.size': 12})
    # method_paper_plot()  # From Plotter/presentation_plots
    # cluster_voids_example_plots()  # From Anti_Clustering/pdf_examples. Take time to generate.

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(8, 4), dpi=144)
    stat_binom_vs_protons(df, stat_plot, div_plt, cent_plt, 39, ['raw', 'mix'], 'ampt_new_coal_epbins1',
                          data_sets_labels=data_sets_labels, ax_in=ax1)
    # ax1.set_title(f'AMPT 39 GeV, 0-5% Centrality, 120° Partitions, 72 Samples/Event', pad=-20)
    ax1.text(0.5, 0.9, 'AMPT 39 GeV, 0-5% Centrality, 120° Partitions', ha='center',
             va='center', transform=ax1.transAxes, fontsize=16)
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.1)
    ax1.set_ylabel(r'$\sigma^2$', fontsize=15)
    ax1.text(0.005, 0.9, '(a)', ha='left', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.legend(loc='lower right')

    markers = {'ampt_new_coal_epbins1': {'raw': 'o', 'mix': 's', 'diff': '^'}}
    dvar_vs_protons(df_dsigma, div_plt, cent_plt, [39], ['raw', 'mix'], ['ampt_new_coal_epbins1'],
                    data_sets_colors=None, data_sets_labels=data_sets_labels, ax_in=ax2, title='', marker_map=markers,
                    plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009], ylabel=r'$\Delta\sigma^2$',
                    kin_info_loc=(0.35, 0.35), leg=False)
    ax2.set_xlabel('Total Protons in Event', fontsize=15)
    ax2.set_ylabel(r'$\Delta\sigma^2$', fontsize=15)
    ax2.text(0.005, 0.9, '(b)', ha='left', va='center', transform=ax2.transAxes, fontsize=12)
    fig.subplots_adjust(hspace=0.0, left=0.126, right=0.995, top=0.995, bottom=0.115)
    fig.canvas.manager.set_window_title('variance_dsig2_vs_tprotons_ampt_39gev')

    df_dsigma_v2sub = pd.read_csv(f'{base_path}{df_dsigma_v2sub_model_name}')

    dsig_avgs_v2sub = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_model_name}')

    dsig_avgs_v2sub['data_type'] = data_type_plt
    subplot_adjust = {'wspace': 0.0, 'hspace': 0.0, 'left': 0.075, 'right': 0.995, 'top': 0.995, 'bottom': 0.09}
    dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], [data_type_plt], data_sets_plt,
                             plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors, no_hydro_label=1,
                             data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.00053], avgs_df=dsig_avgs_v2sub,
                             ylabel=r'$\Delta\sigma^2$', kin_loc=(0.65, 0.41), legend_panel=3,
                             marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
                             title=f'', alpha=0.8, fig_splt_adjust=subplot_adjust, plot_letters=True)

    dsig_avgs_v2_sub_cent8 = dsig_avgs_v2sub[dsig_avgs_v2sub['cent'] == 8]
    subplot_adjust = {'wspace': 0.0, 'hspace': 0.0, 'left': 0.075, 'right': 0.995, 'top': 0.995, 'bottom': 0.095}
    plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, data_sets_plt, data_sets_colors=data_sets_colors, fit=True,
                        data_sets_labels=data_sets_labels, plot_energy_panels=True, legend_order=legend_order,
                        ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_bands=data_sets_bands,
                        plot_indiv=False, ylim=(-0.00054, 0.00009), leg_panel=5, no_hydro_label=True,
                        xlim=(-10, 370), title=f'', leg_frameon=True, exclude_divs=exclude_divs, alpha=0.8,
                        data_sets_markers=data_sets_markers, fig_splt_adj=subplot_adjust, panel_letters=True)

    plt.rcParams["figure.figsize"] = (7, 3.5)
    subplot_adjust = {'left': 0.142, 'right': 0.995, 'top': 0.995, 'bottom': 0.140}
    dsig_avgs_v2_sub_cent8_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['cent'] == 8) & (dsig_avgs_v2sub['divs'] == 120)]
    plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, data_sets_plt, data_sets_colors=data_sets_colors,
                                data_sets_labels=data_sets_labels, alpha=0.6, kin_info_loc=(0.02, 0.45),
                                marker_map=data_sets_markers, fig_splt_adj=subplot_adjust,
                                data_sets_bands=data_sets_bands, legend_order=legend_order, title=f'')

    dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[dsig_avgs_v2sub['divs'] == 120]
    data_sets_energies_colors = \
        {'ampt_new_coal_epbins1': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'}}
    data_sets_energies_markers = {'ampt_new_coal_epbins1': {7: 'o', 11: 's', 19: '^', 27: 'p', 39: 'D', 62: '*'}}
    vs_cent_sets = list(data_sets_energies_colors.keys())

    splt_adjust = {'left': 0.142, 'right': 0.995, 'top': 0.995, 'bottom': 0.11}
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,
                              title=f'', alpha=0.8, errbar_alpha=0.3, xerr=False, xlim=(-10, 699), figsize=(7, 4.5),
                              kin_info_loc=(0.2, 0.1), marker_map=data_sets_energies_markers, fig_splt_adj=splt_adjust,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)

    df_fits = pd.read_csv(f'{base_path}{df_partitions_fits_model_name}')
    data_sets_cent = ['ampt_new_coal_epbins1']

    # plot_div_fits_vs_cent(df_fits, data_sets_cent,  # data_sets_energies_cmaps=data_sets_energies_cmaps,
    #                       data_sets_labels=data_sets_labels, title=None, fit=False, cent_ref=cent_ref_df, ls='-',
    #                       ref_type=ref_type, data_sets_colors=data_sets_energies_colors,
    #                       data_sets_markers=data_sets_energies_markers)

    plot_div_fits_vs_cent_single_set(df_fits, data_sets_cent, data_sets_labels=data_sets_labels, title=None, ls='-',
                                     data_sets_colors=data_sets_energies_colors, cent_ref=cent_ref_df, show_xerr=False,
                                     data_sets_markers=data_sets_energies_markers, ref_type=ref_type)

    df_fits_nov2sub = pd.read_csv(f'{base_path}{df_partitions_fits_model_nov2sub_name}')
    plot_div_fits_vs_cent_single_set(df_fits_nov2sub, data_sets_cent, data_sets_labels=data_sets_labels, title=None,
                                     data_sets_colors=data_sets_energies_colors, cent_ref=cent_ref_df, show_xerr=False,
                                     data_sets_markers=data_sets_energies_markers, ref_type=ref_type, ls='-')

    df_fits_amptv2rp = df_fits[df_fits['data_set'] == 'ampt_new_coal_epbins1_v2rp']
    df_fits_amptv2rp['data_set'] = 'ampt_new_coal_epbins1'
    plot_div_fits_vs_cent_single_set(df_fits_amptv2rp, data_sets_cent, data_sets_labels=data_sets_labels, title=None,
                                     data_sets_colors=data_sets_energies_colors, cent_ref=cent_ref_df, show_xerr=False,
                                     data_sets_markers=data_sets_energies_markers, ref_type=ref_type, ls='-')

    df_fits_amptv2cum = df_fits[df_fits['data_set'] == 'ampt_new_coal_epbins1_v2cum']
    df_fits_amptv2cum['data_set'] = 'ampt_new_coal_epbins1'
    plot_div_fits_vs_cent_single_set(df_fits_amptv2cum, data_sets_cent, data_sets_labels=data_sets_labels, title=None,
                                     data_sets_colors=data_sets_energies_colors, cent_ref=cent_ref_df, show_xerr=False,
                                     data_sets_markers=data_sets_energies_markers, ref_type=ref_type, ls='-')

    plt.show()

    # Gaussian correlation model simulations
    sim_base_path = f'{base_path}Binomial_Slice_Moments/'
    df_sim_dsig = pd.read_csv(f'{sim_base_path}binom_slice_stats_sim_demos_dsigma.csv')
    df_sim_dsig_avgs = pd.read_csv(f'{sim_base_path}binom_slice_stats_sim_demos_dsigma_avg.csv')
    df_sim_width_fits = pd.read_csv(f'{sim_base_path}binom_slice_stats_sim_demos_width_fits.csv')

    amps = ['002', '006', '008', '01']
    spreads = ['04', '05', '06', '07', '08', '09', '1', '11', '12']
    amp_spreads_all = [amps, spreads]

    amps = ['002', '006', '01']
    spreads = ['04', '05', '06', '07', '08', '09', '1', '11', '12']
    amp_spreads_z_w = [amps, spreads]

    amps = ['002', '006', '01']
    spreads = ['1']
    amp_spreads_dsig_n = [amps, spreads]

    amps = ['002', '008']
    spreads = ['04', '08']
    amp_spreads_avg_w = [amps, spreads]

    sim_set_defs = [amp_spreads_all, amp_spreads_dsig_n, amp_spreads_avg_w, amp_spreads_z_w]
    sim_sets_out = []
    for (amps, spreads) in sim_set_defs:
        sim_sets = []
        for amp in amps:
            for spread in spreads:
                sim_sets.append(f'sim_aclmul_amp{amp}_spread{spread}')
                sim_sets.append(f'sim_clmul_amp{amp}_spread{spread}')
        sim_sets = sorted(sim_sets, reverse=True)
        sim_sets = sim_sets[:int(len(sim_sets) / 2)] + sorted(sim_sets[int(len(sim_sets) / 2):])
        sim_sets_out.append(sim_sets)
    sim_sets_all, sim_sets_dsig_n, sim_sets_avg_w, sim_sets_z_w = sim_sets_out

    amp_spread_markers = {0.002: {0.4: 'o', 0.8: 's'}, 0.006: {0.4: '^', 0.8: 'v'}, 0.008: {0.4: '^', 0.8: 'v'},
                          0.01: {0.4: 'D', 0.8: 'x'}}
    amp_spread_colors = {0.002: {0.4: 'black', 0.8: 'red'}, 0.006: {0.4: 'green', 0.8: 'orange'},
                         0.008: {0.4: 'black', 0.8: 'red'}, 0.01: {0.4: 'purple', 0.8: 'black'}}
    amp_spread_ls = {0.002: {0.4: '-', 0.8: '--'}, 0.006: {0.4: '-', 0.8: '--'}, 0.008: {0.4: '-', 0.8: '--'},
                     0.01: {0.4: '-', 0.8: '--'}}
    data_sets_labels_sim, data_sets_colors_sim, data_sets_fill_sim, data_sets_markers_sim = {}, {}, {}, {}
    data_sets_ls_sim = {}
    for sim_set in sim_sets_all:
        label, fillstyle, marker = '', '', ''
        amp, spread = get_name_amp_spread(sim_set)
        if '_clmul_' in sim_set:
            label += 'Attractive ' + rf'$A=+{amp:.3f}$ $Σ={spread}$'
            fillstyle = 'none'
        elif '_aclmul_' in sim_set:
            label += 'Repulsive ' + rf'$A=-{amp:.3f}$ $Σ={spread}$'
            fillstyle = 'full'
        data_sets_labels_sim.update({sim_set: label})
        data_sets_fill_sim.update({sim_set: fillstyle})
        if amp in amp_spread_markers and spread in amp_spread_markers[amp]:
            marker = amp_spread_markers[amp][spread]
        else:
            marker = 'o'
        if amp in amp_spread_colors and spread in amp_spread_colors[amp]:
            color = amp_spread_colors[amp][spread]
        else:
            color = 'black'
        if amp in amp_spread_ls and spread in amp_spread_ls[amp]:
            ls = amp_spread_ls[amp][spread]
        else:
            ls = '-'
        data_sets_markers_sim.update({sim_set: marker})
        data_sets_colors_sim.update({sim_set: color})
        data_sets_ls_sim.update({sim_set: ls})

    fig, ax = plt.subplots(figsize=(8, 4), dpi=144)
    dvar_vs_protons(df_sim_dsig, div_plt, cent_plt, ['sim'], ['raw'], sim_sets_dsig_n, plot=True, avg=True,
                    data_sets_labels=data_sets_labels_sim, ylabel=r'$\Delta\sigma^2$', data_sets_bands=sim_sets_dsig_n,
                    title=None, ax_in=ax, ylim=(-0.00099, 0.00124), kin_info_loc=(0.4, 0.17), leg=False)
    title = f'Gaussian Correlation Model - {div_plt}° Partitions'
    ax.text(0.5, 0.98, title, ha='center', va='top', transform=ax.transAxes, fontsize=16)
    ax.set_xlabel('Total Protons in Event', fontsize=14)
    ax.set_ylabel(r'$\Delta\sigma^2$', fontsize=14)
    ax.axhline(0, color='black', zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    attractive_handles = [handle for handle, label in zip(handles, labels) if 'Attractive' in label]
    repulsive_handles = [handle for handle, label in zip(handles, labels) if 'Repulsive' in label]
    attractive_label = ax.legend(attractive_handles,
                                 [label.replace('Attractive ', '') for label in labels if 'Attractive' in label],
                                 loc='upper right', title='Attractive', bbox_to_anchor=(0.99, 0.935), title_fontsize=11)
    repulsive_label = ax.legend(repulsive_handles,
                                [label.replace('Repulsive ', '') for label in labels if 'Repulsive' in label],
                                loc='upper right', title='Repulsive', bbox_to_anchor=(0.99, 0.27), title_fontsize=11)
    ax.add_artist(attractive_label)

    fig.subplots_adjust(left=0.135, right=0.995, top=0.995, bottom=0.115)
    fig.canvas.manager.set_window_title('dsigma_vs_Total_Protons_simGeV_120_example')

    plt.rcParams["figure.figsize"] = (9, 4.5)
    # plt.rcParams['figure.subplot.left'], plt.rcParams['figure.subplot.right'] = 0.125, 0.995
    # plt.rcParams['figure.subplot.bottom'], plt.rcParams['figure.subplot.top'] = 0.1, 0.94
    subplot_adjust = {'left': 0.111, 'right': 0.995, 'top': 0.995, 'bottom': 0.105}
    plot_dvar_avgs_divs(df_sim_dsig_avgs, sim_sets_avg_w, data_sets_colors=data_sets_colors_sim, fit=True,
                        data_sets_labels=data_sets_labels_sim, plot_energy_panels=False, ylim=(-0.00045, 0.00045),
                        data_sets_markers=data_sets_markers_sim, data_sets_ls=data_sets_ls_sim, xlim=(-15, 485),
                        ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_fills=data_sets_fill_sim, alpha=0.8,
                        exclude_divs=exclude_divs, fig_splt_adj=subplot_adjust, title='', leg=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    attractive_handles = [handle for handle, label in zip(handles, labels) if 'Attractive' in label]
    repulsive_handles = [handle for handle, label in zip(handles, labels) if 'Repulsive' in label]
    attractive_label = ax.legend(attractive_handles,
                                 [label.replace('Attractive ', '') for label in labels if 'Attractive' in label],
                                 loc='center right', title='Attractive', bbox_to_anchor=(1.0, 0.75), title_fontsize=12)
    repulsive_label = ax.legend(repulsive_handles,
                                [label.replace('Repulsive ', '') for label in labels if 'Repulsive' in label],
                                loc='center right', title='Repulsive', bbox_to_anchor=(1.0, 0.25), title_fontsize=12)
    ax.add_artist(attractive_label)

    plot_b_vs_amp(df_sim_width_fits, alpha=0.8, ylim=(-0.00046, 0.00053))

    amp_shifts = {0.002: -0.01, 0.006: 0.0, 0.01: +0.01}
    amp_colors = {0.002: 'red', 0.006: 'black', 0.01: 'blue'}
    amp_markers = {0.002: 's', 0.006: 'o', 0.01: '^'}
    plot_z_vs_spread(df_sim_width_fits, amps=list(amp_colors.keys()), amps_colors=amp_colors, amps_markers=amp_markers,
                     amps_x_shifts=amp_shifts, alpha=0.7)

    # plot_b_vs_amp_sig_dep(df_sim_width_fits, alpha=0.8)

    # plot_slope_div_fits_simpars(df_sim_width_fits)
    # plot_full_test_from_file()  # Momentum_Conservation_Sim.momentum_conservation_model.py
    # Get current figure and set window title
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('dsig2_vs_total_particles_fit')

    plt.show()


def plot_star_var_sys():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144
    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # v2_star_in_dir = 'F:/Research/Data/default_resample_epbins1_calcv2_qaonly_test/' \
    #                  'rapid05_resample_norotate_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_qaonly_test_0/'
    v2_star_in_dir = 'F:/Research/Data/default/' \
                     'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    sys_dir = 'F:/Research/Data/default_sys/'
    sys_default_dir = 'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    df_name = 'Binomial_Slice_Moments/binom_slice_vars_bes_sys.csv'

    plot = True
    # plot = False
    calc_finals = False
    # calc_finals = True
    threads = 11
    sys_pdf_out_path = f'{base_path}systematic_plots.pdf'
    indiv_pdf_out_path = f'F:/Research/Results/BES_QA_Plots/Systematics/'
    df_def_out_name = 'Bes_with_Sys/binom_slice_vars_bes.csv'
    # df_def_out_name = None
    df_def_dsigma_out_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma.csv'
    # df_def_dsigma_out_name = None
    df_def_dsigma_v2sub_out_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma_v2sub.csv'
    df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_bes.csv'
    df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_bes.csv'
    fits_out_base = 'Base_Zero_Fits'
    df_partitions_fits_name = 'Bes_with_Sys/partition_width_fits_bes.csv'
    df_path = base_path + df_name

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 180
    divs_all = [60, 72, 89, 90, 120, 180, 240, 270, 288, 300]
    # divs_all = [60, 120, 180]
    exclude_divs = [356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 7
    cents = [1, 2, 3, 4, 5, 6, 7, 8]
    energies_fit = [7, 11, 19, 27, 39, 62]
    # energies_fit = [7, 11]
    samples = 72  # For title purposes only

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    # sys_info_dict = {  # Old systematics (in my opinion, better) used in thesis. For all gaussian.
    #     'vz': {'name': 'vz range', 'title': 'vz', 'decimal': None, 'default': None,
    #            'sys_vars': ['low7', 'high-7', 'low-5_vzhigh5'], 'val_unit': ' cm',
    #            'sys_var_order': ['low7', 'low-5_vzhigh5', 'high-7']},
    #     'Efficiency': {'name': 'efficiency', 'title': 'efficiency', 'decimal': 2, 'default': 0,
    #                    'sys_vars': [95.0, 90.0], 'val_unit': '%', 'sys_var_order': [95.0, 90.0, 85.0, 80.0]},
    #     'dca': {'name': 'dca', 'title': 'dca', 'decimal': 1, 'default': 1, 'sys_vars': [0.8, 1.2], 'val_unit': ' cm',
    #             'sys_var_order': [0.5, 0.8, 1.2, 1.5]},
    #     'nsprx': {'name': r'n$\sigma$ proton', 'title': r'n$\sigma$ proton', 'decimal': 1, 'default': 1,
    #               'sys_vars': [0.9, 1.1], 'val_unit': '', 'sys_var_order': [0.75, 0.9, 1.1, 1.25]},
    #     'm2r': {'name': r'$m^2$ range', 'title': 'm2 range', 'decimal': 0, 'default': 0.6, 'sys_vars': [0.4, 0.8],
    #             'val_unit': ' GeV', 'sys_var_order': [0.2, 0.4, 0.8, 1.0]},
    #     'nhfit': {'name': 'nHits fit', 'title': 'nhits fit', 'decimal': 2, 'default': 20, 'sys_vars': [15, 25],
    #               'val_unit': '', 'sys_var_order': [15, 25]},
    #     'sysrefshift': {'name': 'refmult3 shift', 'title': 'ref3 shift', 'decimal': None, 'default': 0,
    #                     'sys_vars': ['-1', '1'], 'val_unit': '', 'sys_var_order': ['-1', '1']},
    #     'dcxyqa': {'name': 'dcaxy qa', 'title': 'dcaxy qa', 'decimal': None, 'default': None,
    #                'sys_vars': ['tight', 'loose'], 'val_unit': '',
    #                'sys_var_order': ['2tight', 'tight', 'loose', '2loose']},
    #     'pileupqa': {'name': 'pile-up qa', 'title': 'pile-up qa', 'decimal': None, 'default': None,
    #                  'sys_vars': ['tight', 'loose'], 'val_unit': '',
    #                  'sys_var_order': ['2tight', 'tight', 'loose', '2loose']},
    #     'mix_rand_': {'name': 'mix rand', 'title': 'mix rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
    #                   'val_unit': '', 'sys_var_order': None},
    #     'all_rand_': {'name': 'all rand', 'title': 'all rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
    #                   'val_unit': '', 'sys_var_order': None},
    # }

    sys_info_dict = {
        'vz': {'name': 'vz range', 'title': 'vz', 'decimal': None, 'default': None,
               'sys_vars': ['low7', 'high-7'], 'val_unit': ' cm',
               'sys_var_order': ['low7', 'low-5_vzhigh5', 'high-7'], 'prior': 'flat_two_side'},
        'Efficiency': {'name': 'efficiency', 'title': 'efficiency', 'decimal': 2, 'default': 0,
                       'sys_vars': [90.0], 'val_unit': '%', 'sys_var_order': [95.0, 90.0, 85.0, 80.0],
                       'prior': 'flat_one_side'},
        'dca': {'name': 'dca', 'title': 'dca', 'decimal': 1, 'default': 1, 'sys_vars': [0.5], 'val_unit': ' cm',
                'sys_var_order': [0.5, 0.8, 1.2, 1.5], 'prior': 'flat_one_side'},
        'nsprx': {'name': r'n$\sigma$ proton', 'title': r'n$\sigma$ proton', 'decimal': 1, 'default': 1,
                  'sys_vars': [0.75], 'val_unit': '', 'sys_var_order': [0.75, 0.9, 1.1, 1.25],
                  'prior': 'flat_one_side'},
        'm2r': {'name': r'$m^2$ range', 'title': 'm2 range', 'decimal': 0, 'default': 0.6, 'sys_vars': [0.4],
                'val_unit': ' GeV', 'sys_var_order': [0.2, 0.4, 0.8, 1.0], 'prior': 'flat_one_side'},
        'nhfit': {'name': 'nHits fit', 'title': 'nhits fit', 'decimal': 2, 'default': 20, 'sys_vars': [25],
                  'val_unit': '', 'sys_var_order': [15, 25], 'prior': 'flat_one_side'},
        'sysrefshift': {'name': 'refmult3 shift', 'title': 'ref3 shift', 'decimal': None, 'default': 0,
                        'sys_vars': ['-1', '1'], 'val_unit': '', 'sys_var_order': ['-1', '1'],
                        'prior': 'flat_two_side'},
        'dcxyqa': {'name': 'dcaxy qa', 'title': 'dcaxy qa', 'decimal': None, 'default': None,
                   'sys_vars': ['tight'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
                   'prior': 'flat_one_side'},
        'pileupqa': {'name': 'pile-up qa', 'title': 'pile-up qa', 'decimal': None, 'default': None,
                     'sys_vars': ['tight'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
                     'prior': 'flat_one_side'},
        'mix_rand_': {'name': 'mix rand', 'title': 'mix rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
                      'val_unit': '', 'sys_var_order': None, 'prior': None},
        'all_rand_': {'name': 'all rand', 'title': 'all rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
                      'val_unit': '', 'sys_var_order': None, 'prior': None},
    }

    sys_include_sets = sys_info_dict_to_var_names(sys_info_dict)
    sys_priors = sys_info_dict_to_priors(sys_info_dict)
    sys_include_names = [y for x in sys_include_sets.values() for y in x]

    data_sets_plt = ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'green', 'black', 'red', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']))

    df = pd.read_csv(df_path)
    df = df.dropna()
    df['name'] = df['name'].str.replace('bes_sys_', '')
    all_sets = pd.unique(df['name'])
    print(all_sets)
    print(sys_include_names)

    # For all sets where 'ampt' is not in the name, make a copy of all 'bes_def' rows in cent_ref_df for this data_set
    cent_ref_extra_sets = []
    for data_set_name in all_sets:
        print(data_set_name)
        if 'ampt' not in data_set_name and data_set_name != 'bes_def':
            cent_ref_extra_sets.append(cent_ref_df[cent_ref_df['data_set'] == 'bes_def'].assign(data_set=data_set_name))
    cent_ref_extra_sets = pd.concat(cent_ref_extra_sets)
    cent_ref_df = pd.concat([cent_ref_df, cent_ref_extra_sets])

    rand_sets = [set_name for set_name in all_sets if '_rand' in set_name]
    non_rand_sets = [set_name for set_name in all_sets if '_rand' not in set_name]

    v2_star_vals = {2: read_flow_values(v2_star_in_dir)}
    v2_sys_vals = {name: {2: read_flow_values(get_set_dir(name, sys_default_dir, sys_dir))} for name in all_sets
                   if name != 'bes_def'}
    v2_sys_vals.update({'bes_def': v2_star_vals})

    df = df[df['stat'] == stat_plot]

    # Get k2 raw, mix, diff systematics
    if df_def_out_name is not None and calc_finals:
        df_def_with_sys = get_sys(df, 'bes_def', sys_include_sets, sys_priors,
                                  group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        df_def_with_sys.to_csv(f'{base_path}{df_def_out_name}', index=False)

    # Calculate dsigma with k2 values and get systematics
    df = df[df['stat'] == 'k2']
    df = df.drop('stat', axis=1)
    print('Calc dsigma')
    df_raw, df_mix, df_diff = calc_dsigma(df, ['raw', 'mix', 'diff'])
    df_dsigma_types = pd.concat([df_raw, df_mix, df_diff])
    print('Calc diff nlo error')
    df_dsigma_types = add_diff_nlo_err(df_dsigma_types, group_cols=['energy', 'cent', 'name', 'total_protons'],
                                       exclude_divs=[356, 89])

    if df_def_dsigma_out_name is not None and calc_finals:
        df_def_dsigma = get_sys(df_dsigma_types, 'bes_def', sys_include_sets, sys_priors,
                                group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        print(df_def_dsigma)
        df_def_dsigma.to_csv(f'{base_path}{df_def_dsigma_out_name}', index=False)

    # Calculate v2 subtraction for each total_protons value
    if df_def_dsigma_v2sub_out_name is not None and calc_finals:
        print('Calc v2 sub')
        df_dsigma_types['meas'] = df_dsigma_types.apply(lambda row: Measure(row['val'], row['err']), axis=1)
        df_def_dsigma_v2sub = []
        for set_name in sys_include_names + ['bes_def']:
            set_v2sub = subtract_dsigma_flow(df_dsigma_types, set_name, set_name, v2_sys_vals[set_name],
                                             new_only=True, val_col='val', err_col='err', meas_col='meas')
            df_def_dsigma_v2sub.append(set_v2sub)
        df_def_dsigma_v2sub = pd.concat(df_def_dsigma_v2sub)
        df_def_dsigma_v2sub = get_sys(df_def_dsigma_v2sub, 'bes_def', sys_include_sets, sys_priors,
                                      group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        df_def_dsigma_v2sub.to_csv(f'{base_path}{df_def_dsigma_v2sub_out_name}', index=False)

    sets_run = all_sets if plot else sys_include_names + ['bes_def']
    dsig_avgs_all, dsig_avgs_diff_v2sub = [], []
    print(sets_run)
    for energy in energies_fit:
        print(f'Energy {energy}GeV')
        jobs = [(df_dsigma_types[df_dsigma_types['name'] == set_i], divs_all, cents, energy, ['raw', 'mix', 'diff'],
                 [set_i], None, False, True) for set_i in sets_run]
        dsig_avgs_div_all = []
        with Pool(threads) as pool:
            for job_i in tqdm.tqdm(pool.istarmap(dvar_vs_protons_cents, jobs), total=len(jobs)):
                dsig_avgs_div_all.append(job_i)
        dsig_avgs_div_all = pd.concat(dsig_avgs_div_all, ignore_index=True)
        dsig_avgs_all.append(dsig_avgs_div_all)
        dsig_avgs_div_diff = dsig_avgs_div_all[dsig_avgs_div_all['data_type'] == 'diff']
        dsig_avgs_div_diff = dsig_avgs_div_diff.drop('data_type', axis=1)
        for data_set in sets_run:
            dsig_avgs_div_diff_set = subtract_dsigma_flow(dsig_avgs_div_diff, data_set,
                                                          data_set, v2_sys_vals[data_set], new_only=True)
            dsig_avgs_diff_v2sub.append(dsig_avgs_div_diff_set)

    if df_def_avgs_out_name is not None and calc_finals:
        dsig_avg_all = pd.concat(dsig_avgs_all, ignore_index=True)
        dsig_avgs_def_sys = get_sys(dsig_avg_all, 'bes_def', sys_include_sets, sys_priors, val_col='avg',
                                    err_col='avg_err', group_cols=['divs', 'energy', 'cent', 'data_type'])
        dsig_avgs_def_sys.to_csv(f'{base_path}{df_def_avgs_out_name}', index=False)

    dsig_avgs_diff_v2sub = pd.concat(dsig_avgs_diff_v2sub, ignore_index=True)
    # if plot:
    #     # dsig_avgs_diff_v2sub = dsig_avgs_diff_v2sub[(dsig_avgs_diff_v2sub['divs'] == 120) &
    #     #                                             (dsig_avgs_diff_v2sub['cent'] == 8)]
    #     plot_sys(dsig_avgs_diff_v2sub, 'bes_def', non_rand_sets, sys_info_dict, val_col='avg', err_col='avg_err',
    #              group_cols=['divs', 'energy', 'cent'], y_label=r'$\langle \Delta \sigma^2 \rangle$',
    #              # pdf_out_path=None)
    #              pdf_out_path=sys_pdf_out_path)
    #     # plot_sys(dsig_avgs_diff_v2sub, 'bes_def', rand_sets, sys_info_dict, val_col='avg', err_col='avg_err',
    #     #          group_cols=['divs', 'energy', 'cent'], plot_bars=False, y_label=r'$\Delta \sigma^2$',
    #     #          pdf_out_path=None)
    #     #          # pdf_out_path=sys_pdf_out_path.replace('.pdf', '_rands.pdf'))
    #     # plt.show()

    if df_def_avgs_v2sub_out_name is not None and calc_finals:
        dsig_avg_diff_v2sub_out = get_sys(dsig_avgs_diff_v2sub, 'bes_def', sys_include_sets, sys_priors,
                                          val_col='avg', err_col='avg_err', group_cols=['divs', 'energy', 'cent'])
        dsig_avg_diff_v2sub_out.to_csv(f'{base_path}{df_def_avgs_v2sub_out_name}', index=False)

    dsig_avgs_v2_sub_div120 = dsig_avgs_diff_v2sub[dsig_avgs_diff_v2sub['divs'] == 120]
    cent_fits_120_refmult = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, all_sets, fit=True, cent_ref=cent_ref_df,
                                                        ref_type=ref_type, plot=False)
    cent_fits_120_npart = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, all_sets, fit=True, cent_ref=cent_ref_df,
                                                      ref_type='npart', plot=False)
    param_names = ['a', 'b', 'c']
    for ref_type_i, ref_type_df in zip(['refmult', 'npart'], [cent_fits_120_refmult, cent_fits_120_npart]):
        for param_name in param_names:
            cent_fits_120_ref_type = ref_type_df.copy()
            for other_param_name in param_names:
                if other_param_name == param_name:
                    continue
                cent_fits_120_ref_type.drop(columns=[other_param_name, f'{other_param_name}_err'], inplace=True)
            cent_fits_ref_param_out = get_sys(cent_fits_120_ref_type, 'bes_def', sys_include_sets, sys_priors,
                                          val_col=param_name, err_col=f'{param_name}_err', group_cols=['energy'],
                                          name_col='data_set')
            cent_fits_ref_param_out.to_csv(f'{base_path}/Bes_with_Sys/centrality_fits/{ref_type_i}_fits_120_{param_name}.csv',
                                           index=False)

    # plot_sys(dsig_avgs_diff_v2sub, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='avg', err_col='avg_err',
    #          group_cols=['divs', 'energy', 'cent'], y_label=r'$\langle \Delta \sigma^2 \rangle$',
    #          pdf_out_path=sys_pdf_out_path, indiv_pdf_path=indiv_pdf_out_path)
    plot_sys(dsig_avgs_diff_v2sub, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='avg',
             err_col='avg_err', group_cols=['cent', 'energy', 'divs'], y_label=r'$\langle \Delta \sigma^2 \rangle$',
             pdf_out_path=sys_pdf_out_path, indiv_pdf_path=indiv_pdf_out_path)
    df_120_4_up = dsig_avgs_diff_v2sub[(dsig_avgs_diff_v2sub['cent'] > 4) & (dsig_avgs_diff_v2sub['divs'] == 120)]
    plot_sys_table(df_120_4_up, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='avg',
                   err_col='avg_err', name_col='name', indiv_pdf_path=f'{indiv_pdf_out_path}sys_table_high_cent.pdf')
    df_120_4_down = dsig_avgs_diff_v2sub[(dsig_avgs_diff_v2sub['cent'] <= 4) & (dsig_avgs_diff_v2sub['divs'] == 120)]
    plot_sys_table(df_120_4_down, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='avg',
                   err_col='avg_err', name_col='name', indiv_pdf_path=f'{indiv_pdf_out_path}sys_table_low_cent.pdf')

    df_fits = plot_dvar_avgs_divs(dsig_avgs_diff_v2sub, all_sets, fit=True, plot_energy_panels=False, plot=False)
    if df_partitions_fits_name is not None:
        df_baselines = get_sys(df_fits, 'bes_def', sys_include_sets,  sys_priors, val_col='baseline',
                               err_col='base_err', name_col='data_set', sys_col='base_sys',
                               group_cols=['energy', 'cent'])
        df_zeros = get_sys(df_fits, 'bes_def', sys_include_sets, sys_priors, val_col='zero_mag',
                           err_col='zero_mag_err', name_col='data_set', sys_col='zero_sys',
                           group_cols=['energy', 'cent'])
        df_baselines = df_baselines.drop(columns=['zero_mag', 'zero_mag_err'])
        df_fits_out = pd.concat([df_baselines, df_zeros[['zero_mag', 'zero_mag_err', 'zero_sys']]], axis=1)
        df_fits_out.to_csv(f'{base_path}{df_partitions_fits_name}', index=False)

        print(df_fits_out)
        if plot:
            plot_slope_div_fits(df_fits_out, data_sets_colors, data_sets_labels)
            plot_slope_div_fits_simpars(df_fits_out)

    dsig_baseline_fits_refmult = plot_baseline_vs_cent_fit(df_fits, all_sets, plot=False, cent_ref=cent_ref_df,
                                                           ref_type=ref_type)
    dsig_baseline_fits_npart = plot_baseline_vs_cent_fit(df_fits, all_sets, plot=False, cent_ref=cent_ref_df,
                                                           ref_type='npart')
    param_names = ['a', 'b', 'c']
    for ref_type_i, ref_type_df in zip(['refmult', 'npart'], [dsig_baseline_fits_refmult, dsig_baseline_fits_npart]):
        for param_name in param_names:
            baseline_cent_fits_ref_type = ref_type_df.copy()
            for other_param_name in param_names:
                if other_param_name == param_name:
                    continue
                baseline_cent_fits_ref_type.drop(columns=[other_param_name, f'{other_param_name}_err'], inplace=True)
            baseline_cent_fits_ref_param_out = get_sys(baseline_cent_fits_ref_type, 'bes_def', sys_include_sets,
                                                       sys_priors, val_col=param_name, err_col=f'{param_name}_err',
                                                       group_cols=['energy'], name_col='data_set')
            baseline_cent_fits_ref_param_out.to_csv(f'{base_path}/Bes_with_Sys/baseline_cent_fits/{ref_type_i}_baseline_fits_{param_name}.csv',)

    df_fits = df_fits.rename(columns={'data_set': 'name'})
    plot_sys(df_fits, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='baseline', err_col='base_err',
             group_cols=['cent', 'energy'], name_col='name', indiv_pdf_path=indiv_pdf_out_path,
             pdf_out_path=sys_pdf_out_path.replace('.pdf', '_baseline.pdf'), y_label='Baseline')
    plot_sys(df_fits, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='zero_mag',
             err_col='zero_mag_err', group_cols=['cent', 'energy'], name_col='name',
             pdf_out_path=sys_pdf_out_path.replace('.pdf', '_zeros.pdf'), y_label='Curvature')
    df_fits_4_up = df_fits[df_fits['cent'] > 4]
    plot_sys_table(df_fits_4_up, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='baseline',
                   err_col='base_err', name_col='name',
                   indiv_pdf_path=f'{indiv_pdf_out_path}baseline_sys_table_high_cent.pdf')
    df_fits_4_down = df_fits[df_fits['cent'] <= 4]
    plot_sys_table(df_fits_4_down, 'bes_def', non_rand_sets, sys_info_dict, sys_priors, val_col='baseline',
                   err_col='base_err', name_col='name',
                   indiv_pdf_path=f'{indiv_pdf_out_path}baseline_sys_table_low_cent.pdf')

    plt.show()


def plot_lyons_example():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144
    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # v2_star_in_dir = 'F:/Research/Data/default_resample_epbins1_calcv2_qaonly_test/' \
    #                  'rapid05_resample_norotate_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_qaonly_test_0/'
    v2_star_in_dir = 'F:/Research/Data/default/' \
                     'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    sys_dir = 'F:/Research/Data/default_sys/'
    sys_default_dir = 'rapid05_resample_norotate_seed_dca1_nsprx1_m2r6_m2s0_nhfit20_epbins1_calcv2_0/'
    df_name = 'Binomial_Slice_Moments/binom_slice_vars_bes_sys.csv'

    plot = True
    # plot = False
    calc_finals = False
    # calc_finals = True
    threads = 11
    sys_pdf_out_path = f'{base_path}systematic_lyons_plots.pdf'
    indiv_pdf_out_path = f'F:/Research/Results/BES_QA_Plots/Systematics/'
    # df_def_out_name = 'Bes_with_Sys/binom_slice_vars_bes.csv'
    df_def_out_name = None
    # df_def_dsigma_out_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma.csv'
    df_def_dsigma_out_name = None
    # df_def_dsigma_v2sub_out_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma_v2sub.csv'
    df_def_dsigma_v2sub_out_name = None
    # df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_bes.csv'
    df_def_avgs_out_name = None
    # df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_bes.csv'
    df_def_avgs_v2sub_out_name = None
    fits_out_base = 'Base_Zero_Fits'
    df_partitions_fits_name = 'Bes_with_Sys/partition_width_fits_bes.csv'
    df_path = base_path + df_name

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 180
    divs_all = [60, 72, 89, 90, 120, 180, 240, 270, 288, 300]
    # divs_all = [60, 120, 180]
    exclude_divs = [356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 7
    cents = [1, 2, 3, 4, 5, 6, 7, 8]
    energies_fit = [7, 11, 19, 27, 39, 62]
    # energies_fit = [7, 11]
    samples = 72  # For title purposes only

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    # sys_info_dict = {
    #     'vz': {'name': 'vz range', 'title': 'vz', 'decimal': None, 'default': None,
    #            'sys_vars': ['low7', 'high-7'], 'val_unit': ' cm',
    #            'sys_var_order': ['low7', 'low-5_vzhigh5', 'high-7'], 'prior': 'flat_two_side'},
    #     'Efficiency': {'name': 'efficiency', 'title': 'efficiency', 'decimal': 2, 'default': 0,
    #                    'sys_vars': [90.0], 'val_unit': '%', 'sys_var_order': [95.0, 90.0, 85.0, 80.0],
    #                    'prior': 'flat_one_side'},
    #     'dca': {'name': 'dca', 'title': 'dca', 'decimal': 1, 'default': 1, 'sys_vars': [0.5], 'val_unit': ' cm',
    #             'sys_var_order': [0.5, 0.8, 1.2, 1.5], 'prior': 'flat_one_side'},
    #     'nsprx': {'name': r'n$\sigma$ proton', 'title': r'n$\sigma$ proton', 'decimal': 1, 'default': 1,
    #               'sys_vars': [0.75], 'val_unit': '', 'sys_var_order': [0.75, 0.9, 1.1, 1.25],
    #               'prior': 'flat_one_side'},
    #     'm2r': {'name': r'$m^2$ range', 'title': 'm2 range', 'decimal': 0, 'default': 0.6, 'sys_vars': [0.4],
    #             'val_unit': ' GeV', 'sys_var_order': [0.2, 0.4, 0.8, 1.0], 'prior': 'flat_one_side'},
    #     'nhfit': {'name': 'nHits fit', 'title': 'nhits fit', 'decimal': 2, 'default': 20, 'sys_vars': [25],
    #               'val_unit': '', 'sys_var_order': [15, 25], 'prior': 'flat_one_side'},
    #     'sysrefshift': {'name': 'refmult3 shift', 'title': 'ref3 shift', 'decimal': None, 'default': 0,
    #                     'sys_vars': ['-1', '1'], 'val_unit': '', 'sys_var_order': ['-1', '1'],
    #                     'prior': 'flat_two_side'},
    #     'dcxyqa': {'name': 'dcaxy qa', 'title': 'dcaxy qa', 'decimal': None, 'default': None,
    #                'sys_vars': ['tight'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
    #                'prior': 'flat_one_side'},
    #     'pileupqa': {'name': 'pile-up qa', 'title': 'pile-up qa', 'decimal': None, 'default': None,
    #                  'sys_vars': ['tight'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
    #                  'prior': 'flat_one_side'},
    #     'mix_rand_': {'name': 'mix rand', 'title': 'mix rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
    #                   'val_unit': '', 'sys_var_order': None, 'prior': None},
    #     'all_rand_': {'name': 'all rand', 'title': 'all rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
    #                   'val_unit': '', 'sys_var_order': None, 'prior': None},
    # }

    sys_info_dict = {
        'vz': {'name': 'vz range', 'title': 'vz', 'decimal': None, 'default': None,
               'sys_vars': ['low7', 'low-5_vzhigh5', 'high-7'], 'val_unit': ' cm',
               'sys_var_order': ['low7', 'low-5_vzhigh5', 'high-7'], 'prior': 'flat_two_side'},
        'Efficiency': {'name': 'efficiency', 'title': 'efficiency', 'decimal': 2, 'default': 0,
                       'sys_vars': [95.0, 90.0, 85.0, 80.0], 'val_unit': '%', 'sys_var_order': [95.0, 90.0, 85.0, 80.0],
                       'prior': 'flat_one_side'},
        'dca': {'name': 'dca', 'title': 'dca', 'decimal': 1, 'default': 1, 'sys_vars': [0.5, 0.8, 1.2, 1.5], 'val_unit': ' cm',
                'sys_var_order': [0.5, 0.8, 1.2, 1.5], 'prior': 'flat_one_side'},
        'nsprx': {'name': r'n$\sigma$ proton', 'title': r'n$\sigma$ proton', 'decimal': 1, 'default': 1,
                  'sys_vars': [0.75, 0.9, 1.1, 1.25], 'val_unit': '', 'sys_var_order': [0.75, 0.9, 1.1, 1.25],
                  'prior': 'flat_one_side'},
        'm2r': {'name': r'$m^2$ range', 'title': 'm2 range', 'decimal': 0, 'default': 0.6, 'sys_vars': [0.2, 0.4, 0.8, 1.0],
                'val_unit': ' GeV', 'sys_var_order': [0.2, 0.4, 0.8, 1.0], 'prior': 'flat_one_side'},
        'nhfit': {'name': 'nHits fit', 'title': 'nhits fit', 'decimal': 2, 'default': 20, 'sys_vars': [15, 25],
                  'val_unit': '', 'sys_var_order': [15, 25], 'prior': 'flat_one_side'},
        'sysrefshift': {'name': 'refmult3 shift', 'title': 'ref3 shift', 'decimal': None, 'default': 0,
                        'sys_vars': ['-1', '1'], 'val_unit': '', 'sys_var_order': ['-1', '1'],
                        'prior': 'flat_two_side'},
        'dcxyqa': {'name': 'dcaxy qa', 'title': 'dcaxy qa', 'decimal': None, 'default': None,
                   'sys_vars': ['2tight', 'tight', 'loose', '2loose'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
                   'prior': 'flat_one_side'},
        'pileupqa': {'name': 'pile-up qa', 'title': 'pile-up qa', 'decimal': None, 'default': None,
                     'sys_vars': ['2tight', 'tight', 'loose', '2loose'], 'val_unit': '', 'sys_var_order': ['2tight', 'tight', 'loose', '2loose'],
                     'prior': 'flat_one_side'},
        'mix_rand_': {'name': 'mix rand', 'title': 'mix rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
                      'val_unit': '', 'sys_var_order': None, 'prior': None},
        'all_rand_': {'name': 'all rand', 'title': 'all rand', 'decimal': 1, 'default': 0, 'sys_vars': None,
                      'val_unit': '', 'sys_var_order': None, 'prior': None},
    }

    sys_include_sets = sys_info_dict_to_var_names(sys_info_dict)
    sys_priors = sys_info_dict_to_priors(sys_info_dict)
    sys_include_names = [y for x in sys_include_sets.values() for y in x]

    data_sets_plt = ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'green', 'black', 'red', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['bes_def', 'dca08', 'nsprx09', 'm2r4', 'nhfit25']))

    df = pd.read_csv(df_path)
    df = df.dropna()
    df['name'] = df['name'].str.replace('bes_sys_', '')
    all_sets = pd.unique(df['name'])
    print(all_sets)
    print(sys_include_names)

    # For all sets where 'ampt' is not in the name, make a copy of all 'bes_def' rows in cent_ref_df for this data_set
    cent_ref_extra_sets = []
    for data_set_name in all_sets:
        print(data_set_name)
        if 'ampt' not in data_set_name and data_set_name != 'bes_def':
            cent_ref_extra_sets.append(cent_ref_df[cent_ref_df['data_set'] == 'bes_def'].assign(data_set=data_set_name))
    cent_ref_extra_sets = pd.concat(cent_ref_extra_sets)
    cent_ref_df = pd.concat([cent_ref_df, cent_ref_extra_sets])

    rand_sets = [set_name for set_name in all_sets if '_rand' in set_name]
    non_rand_sets = [set_name for set_name in all_sets if '_rand' not in set_name]

    v2_star_vals = {2: read_flow_values(v2_star_in_dir)}
    v2_sys_vals = {name: {2: read_flow_values(get_set_dir(name, sys_default_dir, sys_dir))} for name in all_sets
                   if name != 'bes_def'}
    v2_sys_vals.update({'bes_def': v2_star_vals})

    df = df[df['stat'] == stat_plot]

    # plot_lyons_sys(df, 'bes_def', sys_include_sets, sys_priors,
    #         group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])

    # Get k2 raw, mix, diff systematics
    if df_def_out_name is not None and calc_finals:
        df_def_with_sys = get_sys(df, 'bes_def', sys_include_sets, sys_priors,
                                  group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        df_def_with_sys.to_csv(f'{base_path}{df_def_out_name}', index=False)

    # Calculate dsigma with k2 values and get systematics
    df = df[df['stat'] == 'k2']
    df = df.drop('stat', axis=1)
    print('Calc dsigma')
    df_raw, df_mix, df_diff = calc_dsigma(df, ['raw', 'mix', 'diff'])
    df_dsigma_types = pd.concat([df_raw, df_mix, df_diff])
    print('Calc diff nlo error')
    df_dsigma_types = add_diff_nlo_err(df_dsigma_types, group_cols=['energy', 'cent', 'name', 'total_protons'],
                                       exclude_divs=[356, 89])

    if df_def_dsigma_out_name is not None and calc_finals:
        df_def_dsigma = get_sys(df_dsigma_types, 'bes_def', sys_include_sets, sys_priors,
                                group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        print(df_def_dsigma)
        df_def_dsigma.to_csv(f'{base_path}{df_def_dsigma_out_name}', index=False)

    # Calculate v2 subtraction for each total_protons value
    if df_def_dsigma_v2sub_out_name is not None and calc_finals:
        print('Calc v2 sub')
        df_dsigma_types['meas'] = df_dsigma_types.apply(lambda row: Measure(row['val'], row['err']), axis=1)
        df_def_dsigma_v2sub = []
        for set_name in sys_include_names + ['bes_def']:
            set_v2sub = subtract_dsigma_flow(df_dsigma_types, set_name, set_name, v2_sys_vals[set_name],
                                             new_only=True, val_col='val', err_col='err', meas_col='meas')
            df_def_dsigma_v2sub.append(set_v2sub)
        df_def_dsigma_v2sub = pd.concat(df_def_dsigma_v2sub)
        df_def_dsigma_v2sub = get_sys(df_def_dsigma_v2sub, 'bes_def', sys_include_sets, sys_priors,
                                      group_cols=['divs', 'energy', 'cent', 'data_type', 'total_protons'])
        df_def_dsigma_v2sub.to_csv(f'{base_path}{df_def_dsigma_v2sub_out_name}', index=False)

    sets_run = all_sets if plot else sys_include_names + ['bes_def']
    dsig_avgs_all, dsig_avgs_diff_v2sub = [], []
    print(sets_run)
    for energy in energies_fit:
        print(f'Energy {energy}GeV')
        jobs = [(df_dsigma_types[df_dsigma_types['name'] == set_i], divs_all, cents, energy, ['raw', 'mix', 'diff'],
                 [set_i], None, False, True) for set_i in sets_run]
        dsig_avgs_div_all = []
        with Pool(threads) as pool:
            for job_i in tqdm.tqdm(pool.istarmap(dvar_vs_protons_cents, jobs), total=len(jobs)):
                dsig_avgs_div_all.append(job_i)
        dsig_avgs_div_all = pd.concat(dsig_avgs_div_all, ignore_index=True)
        dsig_avgs_all.append(dsig_avgs_div_all)
        dsig_avgs_div_diff = dsig_avgs_div_all[dsig_avgs_div_all['data_type'] == 'diff']
        dsig_avgs_div_diff = dsig_avgs_div_diff.drop('data_type', axis=1)
        for data_set in sets_run:
            dsig_avgs_div_diff_set = subtract_dsigma_flow(dsig_avgs_div_diff, data_set,
                                                          data_set, v2_sys_vals[data_set], new_only=True)
            dsig_avgs_diff_v2sub.append(dsig_avgs_div_diff_set)

    if df_def_avgs_out_name is not None and calc_finals:
        dsig_avg_all = pd.concat(dsig_avgs_all, ignore_index=True)
        dsig_avgs_def_sys = get_sys(dsig_avg_all, 'bes_def', sys_include_sets, sys_priors, val_col='avg',
                                    err_col='avg_err', group_cols=['divs', 'energy', 'cent', 'data_type'])
        dsig_avgs_def_sys.to_csv(f'{base_path}{df_def_avgs_out_name}', index=False)

    dsig_avgs_diff_v2sub = pd.concat(dsig_avgs_diff_v2sub, ignore_index=True)
    # if plot:
    #     # dsig_avgs_diff_v2sub = dsig_avgs_diff_v2sub[(dsig_avgs_diff_v2sub['divs'] == 120) &
    #     #                                             (dsig_avgs_diff_v2sub['cent'] == 8)]
    #     plot_sys(dsig_avgs_diff_v2sub, 'bes_def', non_rand_sets, sys_info_dict, val_col='avg', err_col='avg_err',
    #              group_cols=['divs', 'energy', 'cent'], y_label=r'$\langle \Delta \sigma^2 \rangle$',
    #              # pdf_out_path=None)
    #              pdf_out_path=sys_pdf_out_path)
    #     # plot_sys(dsig_avgs_diff_v2sub, 'bes_def', rand_sets, sys_info_dict, val_col='avg', err_col='avg_err',
    #     #          group_cols=['divs', 'energy', 'cent'], plot_bars=False, y_label=r'$\Delta \sigma^2$',
    #     #          pdf_out_path=None)
    #     #          # pdf_out_path=sys_pdf_out_path.replace('.pdf', '_rands.pdf'))
    #     # plt.show()

    if df_def_avgs_v2sub_out_name is not None and calc_finals:
        dsig_avg_diff_v2sub_out = get_sys(dsig_avgs_diff_v2sub, 'bes_def', sys_include_sets, sys_priors,
                                          val_col='avg', err_col='avg_err', group_cols=['divs', 'energy', 'cent'])
        dsig_avg_diff_v2sub_out.to_csv(f'{base_path}{df_def_avgs_v2sub_out_name}', index=False)

    dsig_avgs_v2_sub_div120 = dsig_avgs_diff_v2sub[dsig_avgs_diff_v2sub['divs'] == 120]
    cent_fits_120_refmult = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, all_sets, fit=True, cent_ref=cent_ref_df,
                                                        ref_type=ref_type, plot=False)
    cent_fits_120_npart = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, all_sets, fit=True, cent_ref=cent_ref_df,
                                                      ref_type='npart', plot=False)
    param_names = ['a', 'b', 'c']
    for ref_type_i, ref_type_df in zip(['refmult', 'npart'], [cent_fits_120_refmult, cent_fits_120_npart]):
        for param_name in param_names:
            cent_fits_120_ref_type = ref_type_df.copy()
            for other_param_name in param_names:
                if other_param_name == param_name:
                    continue
                cent_fits_120_ref_type.drop(columns=[other_param_name, f'{other_param_name}_err'], inplace=True)
            cent_fits_ref_param_out = get_sys(cent_fits_120_ref_type, 'bes_def', sys_include_sets, sys_priors,
                                          val_col=param_name, err_col=f'{param_name}_err', group_cols=['energy'],
                                          name_col='data_set')
            cent_fits_ref_param_out.to_csv(f'{base_path}/Bes_with_Sys/centrality_fits/{ref_type_i}_fits_120_{param_name}.csv',
                                           index=False)

    df_fits = plot_dvar_avgs_divs(dsig_avgs_diff_v2sub, all_sets, fit=True, plot_energy_panels=False, plot=False)

    plot_lyons_sys(df_fits, 'bes_def', sys_include_sets, sys_priors, val_col='baseline',
            err_col='base_err', name_col='data_set', sys_col='base_sys',
            group_cols=['energy', 'cent'])
    plt.show()

    if df_partitions_fits_name is not None:
        df_baselines = get_sys(df_fits, 'bes_def', sys_include_sets,  sys_priors, val_col='baseline',
                               err_col='base_err', name_col='data_set', sys_col='base_sys',
                               group_cols=['energy', 'cent'])
        df_zeros = get_sys(df_fits, 'bes_def', sys_include_sets, sys_priors, val_col='zero_mag',
                           err_col='zero_mag_err', name_col='data_set', sys_col='zero_sys',
                           group_cols=['energy', 'cent'])
        df_baselines = df_baselines.drop(columns=['zero_mag', 'zero_mag_err'])
        df_fits_out = pd.concat([df_baselines, df_zeros[['zero_mag', 'zero_mag_err', 'zero_sys']]], axis=1)
        df_fits_out.to_csv(f'{base_path}{df_partitions_fits_name}', index=False)

        print(df_fits_out)
        if plot:
            plot_slope_div_fits(df_fits_out, data_sets_colors, data_sets_labels)
            plot_slope_div_fits_simpars(df_fits_out)

    plt.show()


def make_models_csv():
    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    v2_ampt_in_dir = 'F:/Research/Data_Ampt_New_Coal/default_resample_epbins1/Ampt_rapid05_resample_norotate_epbins1_0/'
    v2_cf_in_dir = 'F:/Research/Data_CF/default_resample_epbins1/CF_rapid05_resample_norotate_epbins1_0/'
    v2_cfev_in_dir = 'F:/Research/Data_CFEV/default_resample_epbins1/CFEV_rapid05_resample_norotate_epbins1_0/'
    v2_cfevb342_in_dir = 'F:/Research/Data_CFEVb342/default_resample_epbins1/' \
                         'CFEVb342_rapid05_resample_norotate_epbins1_0/'
    df_name = 'Binomial_Slice_Moments/binom_slice_stats_var_epbins1.csv'

    threads = 15
    df_def_out_name = 'Bes_with_Sys/binom_slice_vars_model.csv'
    # df_def_out_name = None
    df_def_dsigma_out_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma.csv'
    # df_def_dsigma_out_name = None
    df_def_dsigma_v2sub_out_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma_v2sub.csv'
    df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_model.csv'
    df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_model.csv'
    df_def_avgs_v2sub_raw_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_raw_model.csv'
    df_partitions_fits_name = 'Bes_with_Sys/partition_width_fits_model.csv'
    df_partitions_fits_raw_name = 'Bes_with_Sys/partition_width_fits_raw_model.csv'
    df_partitions_fits_nov2sub_name = 'Bes_with_Sys/partition_width_fits_nov2sub_model.csv'

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    divs_all = [60, 72, 89, 90, 120, 180, 240, 270, 288, 300]
    # divs_all = [60, 120, 180]
    exclude_divs = [356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cents = [1, 2, 3, 4, 5, 6, 7, 8]
    energies_fit = [7, 11, 19, 27, 39, 62]
    # energies_fit = [7, 11]

    df = pd.read_csv(f'{base_path}{df_name}')
    df = df.dropna()
    df = df[df['name'] != 'bes_resample_epbins1']
    all_sets = pd.unique(df['name'])
    print(all_sets)

    v2_ampt_vals = {2: read_flow_values(v2_ampt_in_dir)}
    v2_rp_ampt_vals = {2: read_flow_values(v2_ampt_in_dir, 'v2_rp')}
    v2_cum_ampt_vals = {1: read_flow_values(v2_ampt_in_dir, 'v1_cum_new'),
                        2: read_flow_values(v2_ampt_in_dir, 'v2_cum_new')}
    v2_cf_vals = {2: read_flow_values(v2_cf_in_dir)}
    v2_cfev_vals = {2: read_flow_values(v2_cfev_in_dir)}
    v2_cfevb342_vals = {2: read_flow_values(v2_cfevb342_in_dir)}
    v2_sys_vals = {'ampt_new_coal_epbins1': v2_ampt_vals, 'cf_resample_epbins1': v2_cf_vals,
                   'cfev_resample_epbins1': v2_cfev_vals, 'cfevb342_resample_epbins1': v2_cfevb342_vals,
                   'ampt_new_coal_epbins1_v2rp': v2_rp_ampt_vals, 'ampt_new_coal_epbins1_v2cum': v2_cum_ampt_vals}

    df['energy'] = df.apply(lambda row: 'sim' if 'sim_' in row['name'] else row['energy'], axis=1)
    df = df[df['stat'] == stat_plot]
    df['sys'] = 0

    # Get k2 raw, mix, diff systematics
    df.to_csv(f'{base_path}{df_def_out_name}', index=False)

    # Calculate dsigma with k2 values
    df = df[df['stat'] == 'k2']
    df = df.drop('stat', axis=1)
    df_raw, df_mix, df_diff = calc_dsigma(df, ['raw', 'mix', 'diff'])
    print('Calc dsigma')
    df_dsigma_types = pd.concat([df_raw, df_mix, df_diff])

    # Add v2rp dataset which is just copy of ampt_new_coal_epbins1
    df_ampt_v2rp = df_dsigma_types[df_dsigma_types['name'] == 'ampt_new_coal_epbins1'].copy()
    df_ampt_v2rp['name'] = 'ampt_new_coal_epbins1_v2rp'
    df_ampt_v2cum = df_dsigma_types[df_dsigma_types['name'] == 'ampt_new_coal_epbins1'].copy()
    df_ampt_v2cum['name'] = 'ampt_new_coal_epbins1_v2cum'
    all_sets = np.append(all_sets, ['ampt_new_coal_epbins1_v2rp', 'ampt_new_coal_epbins1_v2cum'])
    df_dsigma_types = pd.concat([df_dsigma_types, df_ampt_v2rp, df_ampt_v2cum], ignore_index=True)

    print('Calc diff nlo error')
    df_dsigma_types = add_diff_nlo_err(df_dsigma_types, group_cols=['energy', 'cent', 'name', 'total_protons'],
                                       exclude_divs=[356, 89])
    df_dsigma_types.to_csv(f'{base_path}{df_def_dsigma_out_name}', index=False)

    # Calculate dsigma with v2 subtracted
    # df_dsigma_types['meas'] = df_dsigma_types.apply(lambda row: Measure(row['val'], row['err']), axis=1)
    # df_def_dsigma_v2sub = subtract_dsigma_flow(df_dsigma_types, 'bes_def', 'bes_def', v2_sys_vals['bes_def'],
    #                                            new_only=True, val_col='val', err_col='err', meas_col='meas')
    # df_def_dsigma_v2sub.to_csv(f'{base_path}{df_def_dsigma_v2sub_out_name}', index=False)

    # Calculate v2 subtraction for each total_protons value
    df_dsigma_types['meas'] = df_dsigma_types.apply(lambda row: Measure(row['val'], row['err']), axis=1)
    df_def_dsigma_v2sub = []
    for set_name in all_sets:
        set_v2sub = subtract_dsigma_flow(df_dsigma_types, set_name, set_name, v2_sys_vals[set_name],
                                         new_only=True, val_col='val', err_col='err', meas_col='meas')
        df_def_dsigma_v2sub.append(set_v2sub)
    df_def_dsigma_v2sub = pd.concat(df_def_dsigma_v2sub)
    df_def_dsigma_v2sub.to_csv(f'{base_path}{df_def_dsigma_v2sub_out_name}', index=False)

    sets_run = all_sets
    dsig_avgs_all, dsig_avgs_diff_v2sub, dsig_avgs_raw_v2sub = [], [], []
    print(sets_run)
    for energy in energies_fit:
        print(f'Energy {energy}GeV')
        jobs = [(df_dsigma_types[df_dsigma_types['name'] == set_i], divs_all, cents, energy, ['raw', 'mix', 'diff'],
                 [set_i], None, False, True) for set_i in sets_run]
        dsig_avgs_div_all = []
        with Pool(threads) as pool:
            for job_i in tqdm.tqdm(pool.istarmap(dvar_vs_protons_cents, jobs), total=len(jobs)):
                dsig_avgs_div_all.append(job_i)
        dsig_avgs_div_all = pd.concat(dsig_avgs_div_all, ignore_index=True)
        dsig_avgs_div_all['sys'] = 0
        dsig_avgs_all.append(dsig_avgs_div_all)
        dsig_avgs_div_diff = dsig_avgs_div_all[dsig_avgs_div_all['data_type'] == 'diff']
        dsig_avgs_div_diff = dsig_avgs_div_diff.drop('data_type', axis=1)
        dsig_avgs_div_raw = dsig_avgs_div_all[dsig_avgs_div_all['data_type'] == 'raw']
        dsig_avgs_div_raw = dsig_avgs_div_raw.drop('data_type', axis=1)
        for data_set in sets_run:
            dsig_avgs_div_diff_set = subtract_dsigma_flow(dsig_avgs_div_diff, data_set,
                                                          data_set, v2_sys_vals[data_set], new_only=True)
            dsig_avgs_diff_v2sub.append(dsig_avgs_div_diff_set)
            dsig_avgs_div_raw_set = subtract_dsigma_flow(dsig_avgs_div_raw, data_set,
                                                         data_set, v2_sys_vals[data_set], new_only=True)
            dsig_avgs_raw_v2sub.append(dsig_avgs_div_raw_set)

    dsig_avg_all = pd.concat(dsig_avgs_all, ignore_index=True)
    dsig_avg_all.to_csv(f'{base_path}{df_def_avgs_out_name}', index=False)

    dsig_avgs_diff_v2sub = pd.concat(dsig_avgs_diff_v2sub, ignore_index=True)
    dsig_avgs_diff_v2sub.to_csv(f'{base_path}{df_def_avgs_v2sub_out_name}', index=False)

    dsig_avgs_raw_v2sub = pd.concat(dsig_avgs_raw_v2sub, ignore_index=True)
    dsig_avgs_raw_v2sub.to_csv(f'{base_path}{df_def_avgs_v2sub_raw_out_name}', index=False)

    df_fits = plot_dvar_avgs_divs(dsig_avgs_diff_v2sub, all_sets, fit=True, plot=True)
    df_fits.to_csv(f'{base_path}{df_partitions_fits_name}', index=False)

    df_fits_raw = plot_dvar_avgs_divs(dsig_avgs_raw_v2sub, all_sets, fit=True, plot=True)
    df_fits_raw.to_csv(f'{base_path}{df_partitions_fits_raw_name}', index=False)

    dsig_avgs_raw = dsig_avg_all[dsig_avg_all['data_type'] == 'raw']
    dsig_avgs_raw = dsig_avgs_raw.drop('data_type', axis=1)

    df_fits_nov2sub = plot_dvar_avgs_divs(dsig_avgs_raw, all_sets, fit=True, plot=True)
    df_fits_nov2sub.to_csv(f'{base_path}{df_partitions_fits_nov2sub_name}', index=False)

    plt.show()


def plot_star_analysis_note_figs():
    plt.rcParams["figure.figsize"] = (6.66, 5)
    plt.rcParams["figure.dpi"] = 144

    presentation_mode = False
    if presentation_mode:
        # plt.rcParams['axes.labelsize'] = 14  # Adjust the value as needed
        # plt.rcParams['axes.titlesize'] = 16  # Adjust the value as needed
        # plt.rcParams['legend.fontsize'] = 14  # Adjust the value as needed
        plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.2
    # else:
    #     plt.rcParams['font.size'] = plt.rcParams['font.size'] * 1.1

    base_path = 'F:/Research/Results/Azimuth_Analysis/'
    # base_path = 'C:/Users/Dyn04/OneDrive - personalmicrosoftsoftware.ucla.edu/OneDrive - UCLA IT Services/Research/UCLA/Results/Azimuth_Analysis/'
    # base_path = 'C:/Users/Dyn04/Research/'
    df_name = 'Bes_with_Sys/binom_slice_vars_bes.csv'
    df_model_name = 'Bes_with_Sys/binom_slice_vars_model.csv'
    df_dsigma_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma.csv'
    df_dsigma_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma.csv'
    df_dsigma_v2sub_name = 'Bes_with_Sys/binom_slice_vars_bes_dsigma_v2sub.csv'
    df_dsigma_v2sub_model_name = 'Bes_with_Sys/binom_slice_vars_model_dsigma_v2sub.csv'
    df_def_avgs_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_bes.csv'
    df_def_avgs_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_model.csv'
    df_def_avgs_v2sub_out_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_bes.csv'
    df_def_avgs_v2sub_out_model_name = 'Bes_with_Sys/dsig_tprotons_avgs_v2sub_model.csv'
    df_partitions_fits_name = 'Bes_with_Sys/partition_width_fits_bes.csv'
    df_partitions_fits_model_name = 'Bes_with_Sys/partition_width_fits_model.csv'

    save_dir = 'C:/Users/Dylan/OneDrive - UCLA IT Services/Research/UCLA/Presentations/STAR_Paper/Autogen_Figures/Analysis_Note/'

    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    stat_plot = 'k2'  # 'standard deviation', 'skewness', 'non-excess kurtosis'
    div_plt = 120
    exclude_divs = [89, 356]  # [60, 72, 89, 90, 180, 240, 270, 288, 300, 356]
    cent_plt = 8
    energies_fit = [7, 11, 19, 27, 39, 62]
    samples = 72  # For title purposes only

    data_sets_plt = ['bes_def', 'ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    data_sets_colors = dict(zip(data_sets_plt, ['black', 'red', 'blue', 'purple']))
    data_sets_labels = dict(zip(data_sets_plt, ['STAR', 'AMPT', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$']))
    data_sets_markers = dict(zip(data_sets_plt, [dict(zip(['raw', 'mix', 'diff'], [x, x, x]))
                                                 for x in ['o', 's', '^', '*']]))
    data_sets_bands = ['ampt_new_coal_epbins1', 'cf_resample_epbins1', 'cfev_resample_epbins1']
    legend_order = ['STAR', 'MUSIC+FIST', 'MUSIC+FIST EV $1fm^3$', 'AMPT']

    cent_ref_name = 'mean_cent_ref.csv'
    cent_ref_df = pd.read_csv(f'{base_path}{cent_ref_name}')
    ref_type = 'refn'  # 'refn'
    cent_ref_df = cent_ref_df.replace('bes_resample_def', 'bes_def')
    cent_ref_df = cent_ref_df.replace('ampt_new_coal_resample_def', 'ampt_new_coal_epbins1')

    df = pd.read_csv(f'{base_path}{df_name}')
    df_model = pd.read_csv(f'{base_path}{df_model_name}')
    df = pd.concat([df, df_model])
    df = df[df['stat'] == stat_plot]

    df_dsigma = pd.read_csv(f'{base_path}{df_dsigma_name}')
    df_dsigma_model = pd.read_csv(f'{base_path}{df_dsigma_model_name}')
    df_dsigma = pd.concat([df_dsigma, df_dsigma_model])

    # dvar_vs_protons(df_dsigma, div_plt, cent_plt, [39], ['raw', 'mix', 'diff'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055], print_data=True,
    #                 data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
    #                                               'diff': 'Single Event - Mixed Event'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo
    # dvar_vs_protons(df_dsigma, div_plt, 4, [39], ['raw', 'mix', 'diff'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055],
    #                 data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
    #                                               'diff': 'Single Event - Mixed Event'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo
    #
    # dvar_vs_protons(df_dsigma, div_plt, cent_plt, [62], ['raw', 'mix', 'diff'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00085, 0.00055],
    #                 data_sets_labels={'bes_def': {'raw': 'Single Event', 'mix': 'Mixed Event',
    #                                               'diff': 'Single Event - Mixed Event'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^'}})  # Mix subtraction demo

    # dvar_vs_protons(df_dsigma, div_plt, cent_plt, [39], ['diff'], data_sets_plt, data_sets_colors=data_sets_colors,
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009], ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels=data_sets_labels, marker_map=data_sets_markers, legend_pos='lower right',
    #                 data_sets_bands=data_sets_bands)

    df_dsigma_v2sub = pd.read_csv(f'{base_path}{df_dsigma_v2sub_name}')
    df_dsigma_v2sub_model = pd.read_csv(f'{base_path}{df_dsigma_v2sub_model_name}')
    # df_dsigma_v2sub_model = df_dsigma_v2sub_model[df_dsigma_v2sub_model['err'] < 0.0001]
    df_dsigma_v2sub = pd.concat([df_dsigma_v2sub, df_dsigma_v2sub_model])

    # dvar_vs_protons(df_dsigma_v2sub, div_plt, cent_plt, [39], ['diff'], data_sets_plt, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_colors=data_sets_colors, plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009],
    #                 data_sets_labels=data_sets_labels, marker_map=data_sets_markers, legend_pos='lower right',
    #                 kin_info_loc=(0.22, 0.13), star_prelim_loc=(0.65, 0.96), data_sets_bands=data_sets_bands,
    #                 legend_order=legend_order)  # 39 GeV mix and v2 subtract dsig2

    df_dsigma_v2sub_diffs = df_dsigma_v2sub[df_dsigma_v2sub['data_type'] == 'diff'].assign(data_type='v2_sub')
    df_dsigma_with_v2sub = pd.concat([df_dsigma, df_dsigma_v2sub_diffs])
    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, cent_plt, [39], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, y_ranges=[-0.00124, 0.0009],
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}})

    # dvar_vs_protons(df_dsigma_with_v2sub, 90, 4, [62], ['raw', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.0039, 0.0009], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 4, [39], ['raw', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=False, alpha=1.0, ylabel=r'$\Delta\sigma^2$', print_data=True,
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.0039, 0.0009], kin_info_loc=(0.26, 0.94))  # v2 sub demo

    dsig_avgs_all = pd.read_csv(f'{base_path}{df_def_avgs_out_name}')
    dsig_avgs_all_model = pd.read_csv(f'{base_path}{df_def_avgs_out_model_name}')
    dsig_avgs_all = pd.concat([dsig_avgs_all, dsig_avgs_all_model])

    # dvar_vs_protons_energies(df_dsigma, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], ['bes_def'],
    #                          plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_all,
    #                          ylabel=r'$\Delta\sigma^2$', kin_loc=(0.65, 0.2), legend_order=legend_order)

    dsig_avgs_v2sub = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_name}')
    dsig_avgs_v2sub_model = pd.read_csv(f'{base_path}{df_def_avgs_v2sub_out_model_name}')
    dsig_avgs_v2sub = pd.concat([dsig_avgs_v2sub, dsig_avgs_v2sub_model])

    dsig_avgs_v2sub['data_type'] = 'diff'
    print(df_dsigma_v2sub.columns)
    # df_dsigma_v2sub = df_dsigma_v2sub[df_dsigma_v2sub['err'] < 0.0001]
    # dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], ['bes_def'],
    #                          plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_v2sub,
    #                          ylabel=r'$\Delta\sigma^2$',
    #                          # kin_loc=(0.55, 0.6), star_prelim_loc=(1, 0.54, 0.53)
    #                          kin_loc=(0.58, 0.45), star_prelim_loc=(1, 0.54, 0.4)
    #                          )

    # dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
    #                          plot=True, avg=False, plot_avg=False, data_sets_colors=data_sets_colors, no_hydro_label=1,
    #                          data_sets_labels=data_sets_labels, y_ranges=[-0.00099, 0.0005], avgs_df=dsig_avgs_v2sub,
    #                          ylabel=r'$\Delta\sigma^2$',
    #                          # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
    #                          kin_loc=(0.58, 0.45), star_prelim_loc=(1, 0.54, 0.4),
    #                          marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
    #                          # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
    #                          title=f''
    #                          )  # <---

    dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_plt, [7, 11, 19, 27, 39, 62], ['diff'], data_sets_plt,
                             plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors, no_hydro_label=1,
                             data_sets_labels=data_sets_labels, y_ranges=[-0.00124, 0.00033], avgs_df=dsig_avgs_v2sub,
                             ylabel=r'$\Delta\sigma^2$', print_data=False,
                             # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
                             kin_loc=(0.58, 0.43), star_prelim_loc=(1, 0.54, 0.37),
                             marker_map=data_sets_markers, data_sets_bands=data_sets_bands, legend_order=legend_order,
                             # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
                             title=f''
                             )  # <---

    # for cent_i in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
    #     df_cent_i = df_dsigma_with_v2sub[(df_dsigma_with_v2sub['cent'] == cent_i) & (df_dsigma_with_v2sub['divs'] == 120)]
    #     min_val = np.percentile(df_cent_i['val'], 5)
    #     dvar_vs_protons_energies(df_dsigma_v2sub, [120], cent_i, [7, 11, 19, 27, 39, 62], ['diff'], ['bes_def', 'ampt_new_coal_epbins1'],
    #                              plot=True, avg=True, plot_avg=True, data_sets_colors=data_sets_colors,
    #                              no_hydro_label=None,
    #                              data_sets_labels=data_sets_labels, y_ranges=[min_val * 1.5, 0.00033],
    #                              avgs_df=dsig_avgs_v2sub,
    #                              ylabel=r'$\Delta\sigma^2$', print_data=False,
    #                              # kin_loc=(0.65, 0.2), star_prelim_loc=(1, 0.54, 0.78),
    #                              kin_loc=(0.58, 0.43), star_prelim_loc=None,
    #                              marker_map=data_sets_markers, data_sets_bands=data_sets_bands,
    #                              legend_order=None,
    #                              # title=f'{cent_map[cent_plt]} Centrality, 120° Partitions'
    #                              title=f''
    #                              )  # <---
        # plt.savefig(f'{save_dir}dsig_vs_nprotons_cent{cent_i}.png')
        # plt.savefig(f'{save_dir}dsig_vs_nprotons_cent{cent_i}.pdf')

    # plt.show()

    # plot_protons_avgs_vs_energy(dsig_avg, ['bes_def'], data_sets_colors=data_sets_colors,
    #                             data_sets_labels=data_sets_labels, title=f'{cent_map[cent_plt]} Centrality, {div_plt}° '
    #                                                                      f'Partitions, {samples} Samples per Event')

    dsig_avgs_v2_sub_cent8 = dsig_avgs_v2sub[dsig_avgs_v2sub['cent'] == 8]
    # plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, data_sets_plt, data_sets_colors=data_sets_colors, fit=False,  # <---
    #                     data_sets_labels=data_sets_labels, plot_energy_panels=True, legend_order=legend_order,
    #                     ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_bands=data_sets_bands,
    #                     plot_indiv=False, ylim=(-0.00079, 0.0001), leg_panel=5, no_hydro_label=True,
    #                     # star_prelim_loc=(1, 0.3, 0.7),
    #                     star_prelim_loc=(1, 0.5, 0.8),
    #                     # xlim=(-10, 370), title=f'0-5% Centrality, {samples} Samples per Event',
    #                     xlim=(-10, 370), title=f'',
    #                     exclude_divs=exclude_divs)
    plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, data_sets_plt, data_sets_colors=data_sets_colors, fit=True,  # <---
                        data_sets_labels=data_sets_labels, plot_energy_panels=True, legend_order=legend_order,
                        ylab=r'$\langle\Delta\sigma^2\rangle$', data_sets_bands=data_sets_bands, print_data=True,
                        plot_indiv=False, ylim=(-0.0009, 0.0001), leg_panel=5, no_hydro_label=True,
                        # star_prelim_loc=(1, 0.3, 0.7),
                        star_prelim_loc=(1, 0.5, 0.8),
                        # xlim=(-10, 370), title=f'0-5% Centrality, {samples} Samples per Event',
                        xlim=(-10, 370), title=f'',
                        exclude_divs=exclude_divs)
    # plt.show()

    # plot_dvar_avgs_divs(dsig_avgs_v2_sub_cent8, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                     data_sets_labels=data_sets_labels, plot_energy_panels=True,
    #                     ylab=r'$\langle\Delta\sigma^2\rangle$',
    #                     plot_indiv=False, ylim=(-0.00079, 0.00019), leg_panel=5,
    #                     # star_prelim_loc=(1, 0.3, 0.7),
    #                     star_prelim_loc=(1, 0.5, 0.8),
    #                     xlim=(-10, 370), title='',
    #                     # title=f'0-5% Centrality, {samples} Samples per Event',
    #                     exclude_divs=exclude_divs)

    dsig_avgs_v2_sub_cent8_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['cent'] == 8) & (dsig_avgs_v2sub['divs'] == 120)]
    plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, data_sets_plt, data_sets_colors=data_sets_colors,  # <---
                                data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.02, 0.595),
                                star_prelim_loc=(0.25, 0.32), marker_map=data_sets_markers, ylim=(-0.00066, 0.00002),
                                data_sets_bands=data_sets_bands, legend_order=legend_order, leg_loc='lower right',
                                # title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')
                                title=f'{cent_map[8]} Centrality, {div_plt}° Partitions')
    # plot_protons_avgs_vs_energy(dsig_avgs_v2_sub_cent8_div120, ['bes_def'], data_sets_colors=data_sets_colors,
    #                             data_sets_labels=data_sets_labels, alpha=1, kin_info_loc=(0.123, 0.68),
    #                             star_prelim_loc=(0.65, 0.9), leg_loc='lower right',
    #                             title=f'{cent_map[8]} Centrality, {div_plt}° Partitions, {samples} Samples per Event')

    # dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[dsig_avgs_v2sub['divs'] == 120]
    dsig_avgs_v2_sub_div120 = dsig_avgs_v2sub[(dsig_avgs_v2sub['divs'] == 120) & (dsig_avgs_v2sub['cent'] > -1)]
    dsig_avgs_all_div120 = dsig_avgs_all[(dsig_avgs_all['divs'] == 120) & (dsig_avgs_all['cent'] > -1) & (dsig_avgs_all['data_type'] == 'diff')]
    data_sets_energies_colors = \
        {'bes_def': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'},
         'ampt_new_coal_epbins1': {7: 'red', 11: 'blue', 19: 'green', 27: 'orange', 39: 'purple', 62: 'black'}}
    vs_cent_sets = list(data_sets_energies_colors.keys())
    # plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                           data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
    #                           title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
    #                           kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
    #                           data_sets_energies_colors=data_sets_energies_colors)
    # plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, ['bes_def'], data_sets_colors=data_sets_colors, fit=False,
    #                           data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
    #                           title=f'{div_plt}° Partitions, {samples} Samples per Event', alpha=0.8, errbar_alpha=0.3,
    #                           kin_info_loc=(0.2, 0.1), star_prelim_loc=(0.6, 0.5),
    #                           data_sets_energies_colors=data_sets_energies_colors)

    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title='Elliptic Flow Corrected', alpha=0.8, errbar_alpha=0.3, ylim=(-0.0051, 0.0005),
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=None, marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands,
                              fig_splt_adj={'top': 0.95, 'right': 0.995, 'bottom': 0.1, 'left': 0.14})
    plt.savefig(f'{save_dir}dsig2_vs_refmult_v2_cor.png')
    plt.savefig(f'{save_dir}dsig2_vs_refmult_v2_cor.pdf')
    plot_protons_avgs_vs_cent(dsig_avgs_v2_sub_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title='Elliptic Flow Corrected', alpha=0.8, errbar_alpha=0.3, ylim=(-0.0051, 0.0005),
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=None, marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands,
                              fig_splt_adj={'top': 0.95, 'right': 0.995, 'bottom': 0.1, 'left': 0.14})
    plt.savefig(f'{save_dir}dsig2_vs_npart_v2_cor.png')
    plt.savefig(f'{save_dir}dsig2_vs_npart_v2_cor.pdf')
    plot_protons_avgs_vs_cent(dsig_avgs_all_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                              title='No Elliptic Flow Correction', alpha=0.8, errbar_alpha=0.3, ylim=(-0.0051, 0.0005),
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=None, marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands,
                              fig_splt_adj={'top': 0.95, 'right': 0.995, 'bottom': 0.1, 'left': 0.14})
    plt.savefig(f'{save_dir}dsig2_vs_refmult_v2_uncor.png')
    plt.savefig(f'{save_dir}dsig2_vs_refmult_v2_uncor.pdf')
    plot_protons_avgs_vs_cent(dsig_avgs_all_div120, vs_cent_sets, data_sets_colors=data_sets_colors, fit=False,
                              data_sets_labels=data_sets_labels, cent_ref=cent_ref_df, ref_type='npart',  # <---
                              title='No Elliptic Flow Correction', alpha=0.8, errbar_alpha=0.3, ylim=(-0.0051, 0.0005),
                              kin_info_loc=(0.2, 0.1), star_prelim_loc=None, marker_map=data_sets_markers,
                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands,
                              fig_splt_adj={'top': 0.95, 'right': 0.995, 'bottom': 0.1, 'left': 0.14})
    plt.savefig(f'{save_dir}dsig2_vs_npart_v2_uncor.png')
    plt.savefig(f'{save_dir}dsig2_vs_npart_v2_uncor.pdf')

    plt.show()

    data_sets_cent = ['ampt_new_coal_epbins1', 'bes_def']
    legend_order = ['7.7 GeV', '11.5 GeV', '19.6 GeV', '27 GeV', '39 GeV', '62.4 GeV', 'AMPT Fit']
    plot_dsig_avg_vs_cent_2panel(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
                                 # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                 title='',
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    # plot_dsig_avg_vs_cent_2panel2(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
    #                               cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
    #                               # title=f'{div_plt}° Partitions, {samples} Samples per Event',
    #                               title='',
    #                               errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
    #                               kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
    #                               data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)
    plot_dsig_avg_vs_cent_2panel62ref(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors,
                                      fit=False, cent_ref=cent_ref_df, ref_type=ref_type, legend_order=None,
                                      title='', errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8, kin_info_loc=(0.2, 0.75),
                                      star_prelim_loc=None, marker_map=data_sets_markers,
                                      data_sets_energies_colors=data_sets_energies_colors,
                                      data_sets_bands=data_sets_bands)

    cent_fits = plot_dsig_avg_vs_cent_fit(dsig_avgs_v2_sub_div120, data_sets_cent, data_sets_colors=data_sets_colors,
                                          fit=True, cent_ref=cent_ref_df, ref_type=ref_type, legend_order=None,
                                          title='', errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8, kin_info_loc=(0.2, 0.75),
                                          star_prelim_loc=None, marker_map=data_sets_markers,
                                          data_sets_energies_colors=data_sets_energies_colors,
                                          data_sets_bands=None)
    print(cent_fits)
    # plt.show()

    dsig_avgs_62ref = dsig_avgs_v2_sub_div120.rename(columns={'name': 'data_set'})
    plot_div_fits_vs_cent_62res(dsig_avgs_62ref, data_sets_cent, data_sets_colors, data_sets_labels, ref_type=ref_type,
                                cent_ref=cent_ref_df, val_col='avg', err_col='avg_err')

    df_fits = pd.read_csv(f'{base_path}{df_partitions_fits_name}')
    df_fits_model = pd.read_csv(f'{base_path}{df_partitions_fits_model_name}')
    df_fits = pd.concat([df_fits, df_fits_model])

    # data_sets_energies_cmaps = dict(zip(data_sets_cent, ['winter', 'copper']))
    # data_sets_markers2 = dict(zip(data_sets_cent, ['s', 'o']))
    # plot_div_fits_vs_cent(df_fits, data_sets_cent,  # data_sets_energies_cmaps=data_sets_energies_cmaps,
    #                       data_sets_labels=data_sets_labels, title=None, fit=False, cent_ref=cent_ref_df,
    #                       ref_type=ref_type, data_sets_colors=data_sets_energies_colors,
    #                       data_sets_markers=data_sets_markers2)

    plot_div_fits_vs_cent_2panel(df_fits, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                                 # title=f'{div_plt}° Partitions, {samples} Samples per Event',
                                 title='', print_data=True,
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.03), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors,
                                 # data_sets_bands=data_sets_bands
                                 )
    plot_baseline_vs_cent_fit(df_fits, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
                                 cent_ref=cent_ref_df, ref_type=ref_type,  # <---
                                 title='', ls='',
                                 errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
                                 kin_info_loc=(0.45, 0.03), marker_map=data_sets_markers,
                                 data_sets_energies_colors=data_sets_energies_colors,
                                 )
    plt.show()

    # plot_slope_div_fits(df_fits, data_sets_colors, data_sets_labels, data_sets=data_sets_plt)
    # plot_slope_div_fits_simpars(df_fits)

    # Plot avg dsig2 vs refmult for mixed events. Wierd stuff at most peripheral bin or two

    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 0, [7], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=True, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected', 'mix': 'Mixed',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black', 'mix': 'purple'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.017, 0.007], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    #
    # dvar_vs_protons(df_dsigma_with_v2sub, div_plt, 2, [7], ['raw', 'mix', 'diff', 'v2_sub'], ['bes_def'],
    #                 plot=True, avg=True, alpha=1.0, ylabel=r'$\Delta\sigma^2$',
    #                 data_sets_labels={'bes_def': {'raw': 'Uncorrected', 'diff': 'Mixed Corrected', 'mix': 'Mixed',
    #                                               'v2_sub': 'Mixed and Flow Corrected'}},
    #                 data_sets_colors={'bes_def': {'raw': 'blue', 'diff': 'red', 'v2_sub': 'black', 'mix': 'purple'}},
    #                 marker_map={'bes_def': {'raw': 'o', 'mix': 's', 'diff': '^', 'v2_sub': '*'}},
    #                 # y_ranges=[-0.00124, 0.0009])  # v2 sub demo
    #                 y_ranges=[-0.017, 0.007], kin_info_loc=(0.26, 0.94))  # v2 sub demo
    #
    # df_mix = []
    # for cent in range(0, 9):
    #     df_mix_cent = dvar_vs_protons(df_dsigma, div_plt, cent, [7, 11, 19, 27, 39, 62],
    #                                   ['mix'], data_sets_plt, plot=False, avg=True)
    #     print(df_mix_cent)
    #     df_mix.append(df_mix_cent)
    # print(df_mix)
    # df_mix = pd.concat(df_mix)
    # print(df_mix)
    # print(df_mix.columns)
    # df_mix = df_mix[(df_mix['divs'] == 120) & (df_mix['data_type'] == 'mix')]
    # plot_dsig_avg_vs_cent_2panel(df_mix, data_sets_cent, data_sets_colors=data_sets_colors, fit=False,
    #                              cent_ref=cent_ref_df, ref_type=ref_type, legend_order=legend_order,  # <---
    #                              # title=f'{div_plt}° Partitions, {samples} Samples per Event',
    #                              title='',
    #                              errbar_alpha=0.3, xlim=(-20, 720), alpha=0.8,
    #                              kin_info_loc=(0.45, 0.1), star_prelim_loc=(0.4, 0.5), marker_map=data_sets_markers,
    #                              data_sets_energies_colors=data_sets_energies_colors, data_sets_bands=data_sets_bands)

    # Save all open figures
    # for i in plt.get_fignums():
    #     plt.figure(i)
    #     window_title = plt.gcf().canvas.manager.get_window_title().replace(' ', '_')
    #     plt.savefig(f'{save_dir}{window_title}_{i}.png')
    #     plt.savefig(f'{save_dir}{window_title}_{i}.pdf')

    plt.show()


if __name__ == '__main__':
    main()
