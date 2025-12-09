#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 18 5:16 PM 2021
Created in PyCharm
Created as QGP_Scripts/analyze_binom_slices

@author: Dylan Neff, dylan
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit as cf
from scipy.optimize import brentq
from scipy import odr
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.interpolate as interp
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from calc_binom_slices import get_name_amp_spread, get_name_amp_spread_pm

from multiprocessing import Pool
import tqdm
import istarmap  # Needed for tqdm

from DistStats import DistStats
from Measure import Measure


def line(x, a, b):
    return a * x + b


def line_yint1(x, a):
    return a * (x - 1) + 1


def line_yint0(x, a):
    return a * x


def quad_180(x, a, c):
    return a * (x - 180) ** 2 + c


def quad_180_rad(x, a, c):
    return a * (x - np.pi) ** 2 + c


def quad_180_zparam(x, z, c):
    return -c * ((x - 180) / z) ** 2 + c


def double_quad_180_zparam(x, z1, c1, z2, c2):
    return quad_180_zparam(x, z1, c1) * quad_180_zparam(x, z2, c2)


def inv_x(x, a):
    return a / x


def inv_x_const(x, a, b):
    return a / x + b


def inv_pow_x_const(x, a, b, c):
    return a / x ** c + b


def inv_pow_x_lin(x, a, b, c, d):
    return a / x ** c + b + d * x


def inv_pow_x_quad(x, a, b, c, d, e):
    return a / x ** c + b + d * x + e * x ** 2


def inv_x_var(x, a, b, c):
    return a / (x * (np.exp(-c * x) + 1)) + b


def inv_x_plus_sqrt(x, a, b, c):
    return a / (x + c * np.sqrt(x)) + b


def inv_sqrtx(x, a, c):
    return a / np.sqrt(x) + c


def inv_pow_m(m, a, b, c):
    return a + b / np.power(m, c)


def inv_sqrtx_noconst_odr(pars, x):
    return pars[0] / np.sqrt(x)


def inv_sqrtx_odr(pars, x):
    return pars[0] / np.sqrt(x) + pars[1]


def inv_powx_odr(pars, x):
    return pars[0] / x ** pars[1] + pars[2]


def inv_sqrtx_poly2_odr(pars, x):
    return pars[0] / np.sqrt(x) + pars[1] + pars[2] * x + pars[3] * x ** 2


def poly2_odr(pars, x):
    return pars[0] + pars[1] * x + pars[2] * x ** 2


def inv_sqrtx_sigmoid_odr(pars, x):  # Not fitting well
    return pars[0] / np.sqrt(x) + pars[1] * (1 + pars[2] / (1 + np.exp(-(x - pars[3]))))


def inv_sqrtx_heaviside_odr(pars, x):  # Not fitting well
    return pars[0] / np.sqrt(x) + pars[1] + pars[2] * np.heaviside(x - pars[3], 1)


def inv_sqrtxlin_odr(pars, x):
    return pars[0] / np.sqrt(x) + pars[1] + pars[2] * x


def inv_invxpow_odr(pars, x):
    return pars[0] / np.power(x, pars[2]) + pars[1]


def inv_invxpow_noc_odr(pars, x):
    return pars[0] / np.power(x, pars[1])


def inv_invx_odr(pars, x):
    return pars[0] / x + pars[1]


def quad_func(x, a, b, c):
    return a * x ** 2 + b * x + c


def x2(x, a):
    return a * x ** 2


def v2_divs_binom_norm(w, v2):
    return v2 ** 2 * np.sin(w) ** 2 / (np.pi * w * (1 - w / (2 * np.pi)))


def v2_divs(w, v2):
    return v2 ** 2 * np.sin(w) ** 2 / (2 * np.pi ** 2)


def v3_divs(w, v3):
    return 2 * v3 ** 2 * np.sin(3 * w / 2) ** 2 / (9 * np.pi ** 2)


def v4_divs(w, v4):
    return v4 ** 2 * np.sin(2 * w) ** 2 / (8 * np.pi ** 2)


def v5_divs(w, v5):
    return 2 * v5 ** 2 * np.sin(5 * w / 2) ** 2 / (25 * np.pi ** 2)


def v6_divs(w, v6):
    return v6 ** 2 * np.sin(3 * w) ** 2 / (18 * np.pi ** 2)


def vn_divs(w, v, n=2):  # Numpy doesn't like Measure for np.sign
    return (v / np.abs(v)) * 2 * v ** 2 * np.sin(n * w / 2) ** 2 / (n ** 2 * np.pi ** 2)


def stat_vs_protons(df, stat, div, cent, energies, data_types, data_sets_plt, y_ranges=None, plot=False, fit=False,
                    hist=False, data_sets_colors=None, data_sets_labels=None, star_prelim=False):
    data = []
    for data_type in data_types:
        for data_set in data_sets_plt:
            for energy in energies:
                df_pre = df
                if 'data_type' in df_pre:
                    df_pre = df_pre[df_pre['data_type'] == data_type]
                if 'cent' in df_pre:
                    df_pre = df_pre[df_pre['cent'] == cent]
                if 'stat' in df_pre:
                    df_pre = df_pre[df_pre['stat'] == stat]
                df_set = df_pre[(df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                if len(df_set) == 0:
                    continue
                if data_sets_labels is None:
                    if energy == 'sim':
                        lab = f'{data_set}_{data_type}'
                    else:
                        lab = f'{data_set}_{data_type}_{energy}GeV'
                else:
                    lab = data_sets_labels[data_set]
                df_set = df_set.sort_values(by=['total_protons'])
                if '_clmultipm_' in data_set:
                    ap, am = df_set['ampplus'].iloc[0], df_set['ampminus'].iloc[0]
                    sp, sm = df_set['spreadplus'].iloc[0], df_set['spreadminus'].iloc[0]
                    amp, spread = 0, 0
                else:
                    amp, spread = df_set['amp'].iloc[0], df_set['spread'].iloc[0]
                    ap, am, sp, sm = 0, 0, 0, 0
                data.append((df_set, lab, data_set, amp, spread, ap, am, sp, sm, energy))

    if plot:
        fig, ax = plt.subplots(figsize=(6.66, 5), dpi=144)
        fig.canvas.manager.set_window_title(f'Raw Div Mix {stat.title()} vs Total Protons {energy}GeV {div}°')
        ax.set_title(f'{energy} GeV, 0-5% Centrality, {div}° Partitions, 72 Samples/Event')
        ax.set_xlabel('Total Protons in Event')
        ax.set_ylabel(f'Raw / Mix {stat.title()}')
        ax.axhline(1, ls='--', color='black')
        if y_ranges:
            ax.set_ylim(y_ranges[stat])
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))

    fits = []
    for i, (df, lab, data_set, amp, spread, ap, am, sp, sm, energy) in enumerate(data):
        zo = len(data) - i + 4
        if plot:
            if data_sets_colors is None:
                c = next(color)
            else:
                c = data_sets_colors[data_set]
            if 'sim_' in data_set:
                ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'], label=lab, color=c,
                                alpha=0.4)
            else:
                ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab,
                            marker='o', ls='', color=c, alpha=0.7, zorder=zo)
                if 'sys' in df:
                    ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='', elinewidth=3,
                                color=c, alpha=0.4, zorder=zo)

        if fit and len(df) > 1:
            popt, pcov = cf(line_yint1, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
            fits.append({'name': data_set, 'energy': energy, 'divs': div, 'amp': amp, 'spread': spread, 'ap': ap,
                         'am': am, 'sp': sp, 'sm': sm, 'slope': popt[0], 'slope_err': np.sqrt(np.diag(pcov))[0],
                         'slope_meas': Measure(popt[0], np.sqrt(np.diag(pcov))[0]),
                         'int': 1, 'int_err': 0})
            if plot:
                ax.plot(df['total_protons'], line_yint1(df['total_protons'], *popt), ls='--', color=c)
                if hist:
                    sigs = (df['val'] - line_yint1(df['total_protons'], *popt)) / df['err']
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.set_title(f'{lab}')
                    ax_hist.set_xlabel('Standard Deviations from Linear Fit')
                    sns.histplot(sigs, stat='density', kde=True)
                    x_norm = np.linspace(min(sigs), max(sigs), 1000)
                    ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                    ax_hist.legend()
                    fig_hist.tight_layout()

    if plot:
        if star_prelim:
            ax.text(10, 0.955, 'STAR Preliminary', fontsize='large')
            eta_line = r'|$\eta$| < 1'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax.text(10, 0.962, f'Au+Au\n{eta_line}\n{pt_line}')
            ax.text(31.6, 1.041, 'statistical uncertainty only', fontsize=8)
            ax.legend(loc='upper center', framealpha=1.0).set_zorder(10)
        else:
            ax.legend(loc='upper left')
        fig.tight_layout()

    return pd.DataFrame(fits)


def raw_to_mix_stat_err(df, div_plt, cent_plt, energy_plt, data_set_plt):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    df = df[(df['divs'] == div_plt) & (df['cent'] == cent_plt) & (df['energy'] == energy_plt) &
            (df['name'] == data_set_plt)]
    df = df.sort_values(by=['total_protons'])
    df_raw = df[df['data_type'] == 'raw'].drop(columns=['data_type'])
    df_mix = df[df['data_type'] == 'mix'].drop(columns=['data_type'])
    df_sub = subtract_avgs(df_raw, df_mix, val_col='val', err_col='err', meas_col='meas')
    # df_sub = df_raw.set_index('total_protons') - df_mix.set_index('total_protons')

    fig, ax = plt.subplots(dpi=144)
    title = r'$\Delta\sigma^2$' f' vs Total Protons {data_set_plt} {energy_plt}GeV {div_plt}° {cent_map[cent_plt]}'
    fig.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.grid()
    ax.errorbar(df_raw['total_protons'], df_raw['val'], df_raw['err'], label='Single Event', marker='o', ls='',
                color='blue', alpha=0.7)
    ax.errorbar(df_mix['total_protons'], df_mix['val'], df_mix['err'], label='Mixed Event', marker='o', ls='',
                color='green', alpha=0.7)
    ax.errorbar(df_sub['total_protons'], df_sub['val'], df_sub['err'], label='Single - Mixed', marker='o', ls='',
                color='red', alpha=0.7)
    ax.set_xlabel('Total Protons in Event')
    ax.set_ylabel(r'$\Delta\sigma^2$')
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(dpi=144)
    title = r'$\Delta\sigma^2$ Error' \
            f' vs Total Protons {data_set_plt} {energy_plt}GeV {div_plt}° {cent_map[cent_plt]}'
    fig.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.grid()
    ax.scatter(df_raw['total_protons'], df_raw['err'], label='Single Event', color='blue', alpha=0.7)
    ax.scatter(df_mix['total_protons'], df_mix['err'], label='Mixed Event', color='green', alpha=0.7)
    ax.scatter(df_sub['total_protons'], df_sub['err'], label='Single - Mixed', color='red', alpha=0.7)
    ax.set_xlabel('Total Protons in Event')
    ax.set_ylabel(r'$\Delta\sigma^2$ Error')
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(dpi=144)
    title = r'$\Delta\sigma^2$ Error Contribution' \
            f' vs Total Protons {data_set_plt} {energy_plt}GeV {div_plt}° {cent_map[cent_plt]}'
    fig.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.axhline(0, color='gray')
    ax.axhline(100, color='gray')
    ax.grid()
    df_raw = df_raw.assign(err2=lambda x: x['err'] ** 2)
    df_mix = df_mix.assign(err2=lambda x: x['err'] ** 2)
    df_sub = df_sub.assign(err2=lambda x: x['err'] ** 2)
    df_ratio_raw = pd.merge(df_raw, df_sub, on='total_protons')
    df_ratio_raw = df_ratio_raw.assign(err_ratio=lambda x: x['err2_x'] / x['err2_y'])
    df_ratio_mix = pd.merge(df_mix, df_sub, on='total_protons')
    df_ratio_mix = df_ratio_mix.assign(err_ratio=lambda x: x['err2_x'] / x['err2_y'])
    ax.scatter(df_ratio_raw['total_protons'], 100 * df_ratio_raw['err_ratio'], label='Single Event', color='blue',
               alpha=0.7)
    ax.scatter(df_ratio_mix['total_protons'], 100 * df_ratio_mix['err_ratio'], label='Mixed Event', color='green',
               alpha=0.7)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Total Protons in Event')
    ax.set_ylabel(r'$\Delta\sigma^2$ Error Contribution')
    ax.legend()
    fig.tight_layout()


def stat_binom_vs_protons(df, stat, div, cent, energy, data_types, data_set_plt, data_sets_labels=None, y_ranges=None,
                          ax_in=None, leg=True):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    data = []
    tp_min, tp_max = 1000, -1
    for data_type in data_types:
        df_pre = df
        if 'data_type' in df_pre:
            df_pre = df_pre[df_pre['data_type'] == data_type]
        if 'cent' in df_pre:
            df_pre = df_pre[df_pre['cent'] == cent]
        if 'stat' in df_pre:
            df_pre = df_pre[df_pre['stat'] == stat]
        df_set = df_pre[(df_pre['name'] == data_set_plt) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
        if data_type == 'raw':
            lab = 'Single Event'
        elif data_type == 'mix':
            lab = 'Mixed Event'
        else:
            lab = '?'
        df_set = df_set.sort_values(by=['total_protons'])
        data.append((df_set, lab, data_type))
        set_tp_min, set_tp_max = df_set['total_protons'].min(), df_set['total_protons'].max()
        tp_min = tp_min if set_tp_min > tp_min else set_tp_min
        tp_max = tp_max if set_tp_max < tp_max else set_tp_max

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6.66, 5), dpi=144)
        fig.canvas.manager.set_window_title(f'Binomial Comparison {stat.title()} vs Total Protons {energy}GeV {div}°')
    else:
        ax = ax_in
    if data_sets_labels is not None:
        name = data_sets_labels[data_set_plt] + ' '
    else:
        name = data_set_plt + '\n'

    ax.set_ylabel(f'{stat.title()}')
    if y_ranges:
        ax.set_ylim(y_ranges[stat])

    xs = np.linspace(tp_min, tp_max, 1000)
    p = div / 360
    if stat in ['variance', 'k2']:
        binom_ys = xs * p * (1 - p)
    else:
        print(f'Stat {stat} not implemented!')
        binom_ys = xs * 0
    ax.plot(xs, binom_ys, color='red', label='Binomial')

    for df, lab, data_type in data:
        c = 'black'
        m = 'o'
        if data_type == 'raw':
            c, m = 'blue', 'o'
        elif data_type == 'mix':
            c, m = 'green', 's'

        ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab, marker=m, ls='', color=c, alpha=0.7)

    if leg:
        ax.legend()
    if ax_in is None:
        ax.set_title(f'{name}{energy} GeV, {cent_map[cent]} Centrality, {div}° Partitions, 72 Samples/Event')
        ax.set_xlabel('Total Protons in Event')
        fig.tight_layout()


def calc_nlo(df, exclude_divs):
    df_og = df.copy()
    df_mix_div_mean = df.loc[(df['data_type'] == 'mix') & (~df['divs'].isin(exclude_divs)), 'val'].mean()
    df_diff_errs = (df.loc[df['data_type'] == 'diff', 'err'] ** 2 +
                    abs(df.loc[df['data_type'] == 'diff', 'val'] * df_mix_div_mean) ** 1.5) ** 0.5
    df.loc[df['data_type'] == 'diff', 'err'] = df_diff_errs
    df_og['new_err'] = df['err']
    # print(df_og.loc[:, ['data_type', 'divs', 'val', 'err', 'new_err']])
    return df


def add_diff_nlo_err(df, group_cols, exclude_divs):
    df_group = df.groupby(group_cols).apply(calc_nlo, exclude_divs)
    return df_group.reset_index(drop=True)


def calc_sys(def_val, def_err, sys_vals, sys_errs, return_vals='both'):
    """
    Calculate systematic uncertainties of all input systematics with respect to single default value.
    Combine uncertainties (nominally just sum the squares). Return this combined uncertainty along with optionally the
    individual uncertainties.
    :param def_val: Default value
    :param def_err: Default statistical uncertainty
    :param sys_vals: Array of systematic variation values
    :param sys_errs: Array of systematic variation statistical uncertainties
    :param return_vals: String specifying what to return. If 'both' return individual systematic uncertainties along
    with their combination. If 'combo' just return the combination of systematic uncertainties. If 'indiv' return
    individual systematic uncertainties only.
    :return: Depending on return_vals return individual systematic uncertainties and/or their combination.
    """
    try:
        barlow_i = np.sqrt(
            np.clip((def_val - sys_vals) ** 2 - np.abs(def_err ** 2 - sys_errs ** 2), a_min=0, a_max=None))
    except TypeError:
        print(def_val, sys_vals, def_err, sys_errs)

    if 'indiv' in return_vals.lower():
        return barlow_i

    barlow = np.sqrt(np.sum(barlow_i ** 2))

    if 'combo' in return_vals.lower():
        return barlow

    return barlow, barlow_i


def get_sys(df, df_def_name, df_sys_dict, sys_prior_dict=None, group_cols=None,
            val_col='val', err_col='err', name_col='name', sys_col='sys'):
    """
    Calculate systematic uncertainties on the default set in df.
    Only ready to take single variation per systematic variable.
    :param df: Dataframe
    :param df_def_name: Default set name
    :param df_sys_dict: Dictionary of systematic sets to be analyzed. {set_type: [set_names]}
    :param sys_prior_dict: Dictionary of systematic prior distributions. If None, all gaus. {set_name: prior}
            gaus: Gaussian prior. Variation(s) correspond to 1 sigma estimate. If 2 variations, take max.
            flat_one_side: Uniform prior. Variation corresponds to edge of distribution, with default on other edge.
                            Divide by 1/sqrt(12) to get 1 sigma estimate.
            flat_two_side: Uniform prior. Variation(s) correspond to edge of distribution, with default in the center.
                            Divide by 2/sqrt(12)=1/sqrt(3) to get 1 sigma estimate.
    :param group_cols: Name of columns by which to group values.
    For example, energies and centralities should be separate
    :param val_col: Name of column containing the values of interest
    :param err_col: Name of column containing the statistical uncertainties on the values of interest
    :param name_col: Name of column for the name of the data sets
    :param sys_col: Name of the column for the systematic to output
    :return: Dataframe with default values, statistical uncertainties, and estimated systematic uncertainties
    """
    if group_cols is None:
        group_cols = ['divs', 'energy', 'cent', 'data_type', 'total_protons']

    sys_set_names = [y for x in df_sys_dict.values() for y in x]
    df_filtered = df[df[name_col].isin(list(sys_set_names) + [df_def_name])]
    df_set = df_filtered.groupby(group_cols)
    df_def_sys = []

    for group_name, group_df in df_set:
        if df_def_name not in group_df[name_col].values:
            continue  # No default value so no systematic

        group_df_def = group_df[group_df[name_col] == df_def_name]
        assert len(group_df_def) == 1
        def_val, def_err = group_df_def[val_col].values[0], group_df_def[err_col].values[0]
        barlow = 0

        if len(group_df) <= 1:
            pass  # No systematic values so sys = 0
        else:
            group_df_sys = group_df[group_df[name_col] != df_def_name]

            barlow = 0
            for sys_type, sys_names in df_sys_dict.items():
                group_df_sys_type = group_df_sys[group_df_sys[name_col].isin(sys_names)]
                sys_val = group_df_sys_type[val_col].values
                sys_err = group_df_sys_type[err_col].values

                barlow_i = calc_sys(def_val, def_err, sys_val, sys_err, 'indiv') ** 2
                if sys_prior_dict is None or sys_type not in sys_prior_dict or sys_prior_dict[sys_type] == 'gaus':
                    barlow_i = barlow_i  # Do nothing, treat as gaussian 1 sigma. Dumb if statement but hopefully clear.
                elif sys_prior_dict[sys_type] == 'flat_one_side':
                    barlow_i = barlow_i / 12  # 1 sigma estimate for a uniform distribution with default at one edge
                elif sys_prior_dict[sys_type] == 'flat_two_side':
                    barlow_i = barlow_i / 3  # 1 sigma estimate for a uniform distribution with default at center
                barlow += np.max(barlow_i) if barlow_i.size > 0 else 0
            barlow = np.sqrt(barlow)

        df_entry = group_df_def.reset_index().to_dict(orient='records')[0]
        df_entry.update({sys_col: barlow})
        del df_entry['index']
        df_def_sys.append(df_entry)

    return pd.DataFrame(df_def_sys)


def add_sys_info(df, sys_info_dict, name_col='name'):
    sys_name_dict = {sys_type: sys_info[name_col] for sys_type, sys_info in sys_info_dict.items()}
    # df[['sys_type', 'sys_val']] = df[name_col].apply(lambda x: pd.Series(split_string(x)))
    df[['sys_type', 'sys_val']] = df[name_col].apply(lambda x: pd.Series(split_sys_type_val(x)))
    df['sys_name'] = df['sys_type'].map(sys_name_dict)

    def get_sys_val(row):
        if row[name_col] == 'default':
            return None
        if row['sys_type'] == 'm2r' and row['sys_val'] == '10':
            return 1.0
        if row['sys_type'] == 'Efficiency':
            val = round(float(f'0.{row["sys_val"]}') * 10 ** sys_info_dict[row['sys_type']]['decimal'], 3)
            return int(100 - val)
        if '_rand' in row['sys_type']:
            return int(row['sys_val'])
        if sys_info_dict[row['sys_type']]['decimal'] is None:
            return row['sys_val']
        # if row['sys_type']
        return round(float(f'0.{row["sys_val"]}') * 10 ** sys_info_dict[row['sys_type']]['decimal'], 3)

    def get_sys_val_str(row):
        val = row["sys_val"]
        if row[name_col] == 'default':
            return None
        elif row['sys_type'] == 'nsprx':
            val *= 2
        elif row['sys_type'] == 'vz':
            if val == 'high-7':
                val = '[-50, -11.7]' if row['energy'] == 7 else '[-30, -7]'
            elif val == 'low7':
                val = '[11.7, 50]' if row['energy'] == 7 else '[7, 30]'
            elif val == 'low-5_vzhigh5':
                val = '[-8.3, 8.3]' if row['energy'] == 7 else '[-5, 5]'
        elif row['sys_type'] == 'sysrefshift':
            if int(val) == 1:
                val = '+1'
        elif row['sys_type'] == 'dcxyqa':
            if val == 'loose':
                val = r'$\sigma$ + 1'
            if val == 'tight':
                val = r'$\sigma$ - 1'
            if val == '2loose':
                val = r'$\sigma$ + 2'
            if val == '2tight':
                val = r'$\sigma$ - 2'
        elif row['sys_type'] == 'pileupqa':
            if val == 'loose':
                val = r'$\sigma$ = 4'
            if val == 'tight':
                val = r'$\sigma$ = 2'
            if val == '2loose':
                val = r'$\sigma$ = 5'
            if val == '2tight':
                val = r'$\sigma$ = 1.5'

        return f'{sys_info_dict[row["sys_type"]]["title"]} = {val}{sys_info_dict[row["sys_type"]]["val_unit"]}'

    df['sys_val'] = df.apply(get_sys_val, axis=1)
    df['sys_val_str'] = df.apply(get_sys_val_str, axis=1)

    return df


def sys_info_dict_to_var_names(sys_info_dict):
    """
    Given sys_info_dict, return dictionary of all systematics data set names to be combined for systematics.
    :param sys_info_dict:
    :return: Dictionary of systematic data set names to be combined. {key: [data_set_names]}
    """
    sys_vars_names = {}
    for key in sys_info_dict:
        if sys_info_dict[key]['sys_vars'] is not None:
            sys_vars_names.update({key: []})
            for sys_var_val in sys_info_dict[key]['sys_vars']:
                if sys_info_dict[key]['decimal'] is None:
                    sys_vars_names[key].append(f'{key}{sys_var_val}')
                else:
                    if key == 'Efficiency':
                        sys_var_val = 100 - sys_var_val
                    str_val = sys_var_val / 10 ** sys_info_dict[key]['decimal']
                    # str_val = str(round(str_val, sys_info_dict[key]['decimal'] + 1)).replace('0.', '')
                    str_val = str(str_val).replace('0.', '')
                    sys_vars_names[key].append(f'{key}{str_val}')
    # print(sys_vars_names)
    # print([y for x in sys_vars_names.values() for y in x])
    return sys_vars_names


def sys_info_dict_to_priors(sys_info_dict):
    """
    Read priors out of sys_info_dict and return dictionary of priors.
    :param sys_info_dict: Dictionary with full systematics info
    :return: Dict matching sys type to prior. {sys_type: prior}
    """
    sys_priors = {}
    for key in sys_info_dict:
        if sys_info_dict[key]['sys_vars'] is not None and sys_info_dict[key]['prior'] is not None:
            sys_priors.update({key: sys_info_dict[key]['prior']})
    return sys_priors


def split_sys_type_val(sys_name):
    if any([x in sys_name for x in ['vzlow', 'vzhigh']]):
        return 'vz', sys_name[2:]
    for sys_type in ['dcxyqa', 'pileupqa', 'sysrefshift']:
        if sys_type in sys_name:
            return sys_type, sys_name.replace(sys_type, '')
    # if 'dcxyqa' in sys_name:
    #     return 'dcxyqa', sys_name.replace('dcxyqa', '')
    # if 'pileupqa' in sys_name:
    #     return 'pileupqa', sys_name.replace('pileupqa', '')
    # if 'sysrefshift' in sys_name:
    #     pass
    else:
        return split_string(sys_name)


def split_string(string):
    for i in range(len(string), 0, -1):
        if not string[i - 1].isdigit():
            return string[:i], string[i:]
    return string, ''


def sort_on_sys_info_dict(df, sys_info_dict):
    sys_type_map = dict(zip(sys_info_dict.keys(), range(len(sys_info_dict))))
    df['sys_type_index'] = df['sys_type'].map(sys_type_map)

    # print(df.columns)
    # df['sys_val_index'] = df.apply(lambda x: sys_info_dict[x['sys_type']]['sys_vars'].index(x['sys_val']), axis=1)

    def get_sys_val_index(row):
        sys_vars = sys_info_dict[row['sys_type']]['sys_var_order']
        if sys_vars is None:
            return 1
        val_index = len(sys_vars) + 1
        if row['sys_val'] in sys_vars:
            val_index = sys_vars.index(row['sys_val'])

        return val_index

    df['sys_val_index'] = df.apply(get_sys_val_index, axis=1)

    return df


def get_set_dir(set_name, def_dir, base_dir):
    # print(set_name)
    if set_name == 'default':
        new_dir = def_dir
    else:
        if any([x in set_name for x in ['sysrefshift', 'dcxyqa', 'pileupqa', 'vzlow', 'vzhigh']]):
            def_dir = def_dir.replace('_seed', '')
        if 'all_rand_' in set_name:
            set_num = int(set_name.replace('all_rand_', ''))
            new_dir = '_'.join(def_dir.split('_')[:-1]) + f'_{set_num}/'
            new_dir = new_dir.replace('_seed', '')
        elif 'mix_rand_' in set_name:
            set_num = int(set_name.replace('mix_rand_', ''))
            new_dir = '_'.join(def_dir.split('_')[:-1]) + f'_{set_num}/'
            new_dir = new_dir.replace('_seed', '_mixnoseed')
        elif 'Efficiency' in set_name:
            new_dir = def_dir.replace('_epbins1', f'_{set_name}_epbins1')
        elif 'sysrefshift' in set_name:
            new_dir = def_dir.replace('_epbins1', f'_epbins1_{set_name}')
        elif any([x in set_name for x in ['sysrefshift', 'vzlow', 'vzhigh', 'pileupqa', 'dcxyqa']]):
            new_dir = def_dir.replace('_epbins1', f'_epbins1_{set_name}')
        else:
            sys_type, sys_val = split_string(set_name)
            new_dir = '_'.join([set_name if sys_type in dir_key else dir_key for dir_key in def_dir.split('_')])
    # print(f'{set_name} {base_dir}{new_dir}')
    return f'{base_dir}{new_dir}'


def plot_vs_sys(df, df_def_name, def_val, df_sys_set_names, sys_info_dict, group_cols=None,
                val_col='val', err_col='err'):
    if group_cols is None:
        group_cols = ['divs', 'energy', 'cent', 'data_type', 'total_protons']
    df_filtered = df[df['name'].isin(list(df_sys_set_names) + [df_def_name])]
    df_filtered = add_sys_info(df_filtered, sys_info_dict)
    df_filtered.loc[df_filtered['name'] == 'default', 'sys_val'] = def_val
    data_types = df_filtered['data_type'].unique()

    df_set = df_filtered.groupby(group_cols)

    for group_name, group_df in df_set:
        if type(group_name) is not list:
            group_name = [group_name]
        if df_def_name not in group_df['name'].values:
            continue  # No default value so no systematic
        fig, axs = plt.subplots(nrows=len(data_types), sharex=True, dpi=144, figsize=(6.66, 5))
        axs[-1].set_xlabel(split_string(df_sys_set_names[0])[0])
        print(group_cols, group_name)
        fig.suptitle(', '.join([f'{col}={val}' for col, val in zip(group_cols, group_name)]))
        for index, data_type in enumerate(group_df['data_type'].unique()):
            axs[index].grid()
            axs[index].set_ylabel(data_type)
            df_dt = group_df[group_df['data_type'] == data_type]
            for energy in df_dt['energy'].unique():
                df_e = df_dt[df_dt['energy'] == energy]
                df_e = df_e.sort_values(by='sys_val')
                axs[index].errorbar(df_e['sys_val'], df_e[val_col], df_e[err_col], marker='o', label=f'{energy} GeV')
        axs[0].legend()
        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)


def plot_sys(df, df_def_name, df_sys_set_names, sys_info_dict, sys_prior_dict=None, group_cols=None,
             val_col='val', err_col='err', name_col='name', plot_barlow_decomp=False, plot_bars=True, y_label=None,
             pdf_out_path=None, indiv_pdf_path=None):
    if group_cols is None:
        group_cols = ['divs', 'energy', 'cent', 'data_type', 'total_protons']

    def_color = 'black'
    def_title = 'default'
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    title_maps = {'divs': lambda div: f'{div}°',
                  'energy': lambda energy: f'{energy}GeV',
                  'cent': lambda cent: f'{cent_map[cent]}'}

    df_filtered = df[df[name_col].isin(list(df_sys_set_names) + [df_def_name])]
    df_filtered[['sys_type', 'sys_val']] = df_filtered[name_col].apply(lambda x: pd.Series(split_sys_type_val(x)))
    # df_filtered.loc[:, ['sys_type', 'sys_val']] = df_filtered[name_col].apply(lambda x: pd.Series(split_sys_type_val(x)))
    # Commented out line above doesn't work, don't let warnings make you try it again

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sys_types = [sys_type for sys_type in df_filtered['sys_type'].unique() if sys_type != df_def_name]
    color_dict = {sys_type: color for sys_type, color in zip(sys_types, default_colors)}
    for sys_type, color in zip(sys_types, default_colors):
        print(sys_type, color)
    df_filtered['color'] = df_filtered['sys_type'].map(color_dict)
    print(df_filtered.loc[:, ['sys_type', 'color']].head(200))

    if pdf_out_path is not None:
        pdf_pages = PdfPages(pdf_out_path)

    df_set = df_filtered.groupby(group_cols)

    for group_name, group_df in df_set:
        if df_def_name not in group_df[name_col].values:
            continue  # No default value so no systematic
        fig = plt.figure(figsize=(13.33, 6.16), dpi=144)
        ts = [title_maps[col](val) if col in title_maps else f'{col}={val}' for col, val in zip(group_cols, group_name)]
        title = ', '.join(ts)
        if plot_bars:
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            ax_bar = plt.subplot(gs[0])
            ax_errorbar = plt.subplot(gs[1], sharex=ax_bar)
            ax_bar.set_title(title)
            if y_label is not None:
                ax_bar.set_ylabel('Systematic Uncertainty')
        else:
            ax_errorbar = plt.subplot()
            ax_errorbar.set_title(title)
        ax_errorbar.grid()
        if y_label is not None:
            ax_errorbar.set_ylabel(y_label)

        if plot_barlow_decomp:
            fig_barlow_decomp, ax_barlow_decomp = plt.subplots(figsize=(8, 6), dpi=144)
            ax_barlow_decomp.set_title(', '.join([f'{col}={val}' for col, val in zip(group_cols, group_name)]))
            ax_barlow_decomp.grid()

        group_df_def = group_df[group_df[name_col] == df_def_name]
        assert len(group_df_def) == 1
        def_val, def_err = group_df_def[val_col].values[0], group_df_def[err_col].values[0]

        if len(group_df) <= 1:
            pass  # No systematic values so sys = 0
        else:
            group_df_sys = group_df[group_df[name_col] != df_def_name].copy()
            group_df_sys = add_sys_info(group_df_sys, sys_info_dict, name_col)

            sys_val = group_df_sys[val_col].values
            sys_err = group_df_sys[err_col].values

            barlow_i = calc_sys(def_val, def_err, sys_val, sys_err, 'indiv')
            # Where prior is 'flat_one_side' or 'flat_two_side', divide by sqrt(12) or sqrt(3) respectively.
            # Otherwise, assume gaussian and divide by 1
            # Use sys_prior_dict with group_df_sys['sys_type'] to get series of priors
            priors = group_df_sys['sys_type'].map(sys_prior_dict)
            prior_divisors = priors.map(lambda x: np.sqrt(12) if x == 'flat_one_side' else np.sqrt(3) if x == 'flat_two_side' else 1)
            barlow_i = barlow_i / prior_divisors
            group_df_sys['barlow'] = barlow_i
            group_df_sys['barlow_sum'] = barlow_i

            group_df_sys['include_sys'] = group_df_sys.apply(
                lambda row: False if sys_info_dict[row['sys_type']]['sys_vars'] is None else
                row['sys_val'] in sys_info_dict[row['sys_type']]['sys_vars'], axis=1)

            # print(group_df_sys)

            max_indices = []  # For each systematic find index of variation to count in systematic total
            for sys_type in group_df_sys['sys_type'].unique():
                # print(sys_type, sys_info_dict[sys_type]['sys_vars'],
                #       group_df_sys[(group_df_sys['sys_type'] == sys_type)]['sys_val'])
                sys_mask = (group_df_sys['sys_type'] == sys_type) & (group_df_sys['include_sys'])  # Only 'sys_vars'
                sys_barlows = group_df_sys.loc[sys_mask, 'barlow_sum']
                if len(sys_barlows) > 0:
                    max_indices.append(group_df_sys.loc[sys_mask, 'barlow_sum'].idxmax())
            # print(f'max_indices: {max_indices}')
            # print(group_df_sys.index.isin(max_indices))
            group_df_sys.loc[~group_df_sys.index.isin(max_indices), 'barlow_sum'] = 0

            barlow = np.sqrt(np.sum(group_df_sys['barlow_sum'].values ** 2))

            sys_types = group_df_sys['sys_type'].values
            if plot_barlow_decomp:
                ax_barlow_decomp.axhline(0, color='black')
                ax_barlow_decomp.axhline(def_err ** 2, ls='--', label=r'$\sigma_d^2$')
                ax_barlow_decomp.scatter(sys_types, (def_val - sys_val) ** 2, marker='_', s=500, label=r'$(d-v)^2$')
                ax_barlow_decomp.scatter(sys_types, sys_err ** 2, marker='_', s=500, label=r'$\sigma_v^2$')
                ax_barlow_decomp.scatter(sys_types, np.abs(def_err ** 2 - sys_err ** 2), s=500, marker='_',
                                         label=r'$|\sigma_d^2 - \sigma_v^2|$')
                ax_barlow_decomp.scatter(sys_types, barlow_i, marker='_', s=500,
                                         label=r'$(d-v)^2 - |\sigma_d^2 - \sigma_v^2|$')
                ax_barlow_decomp.legend()

            ax_errorbar.axhline(def_val, color=def_color)
            # print(df_def_name, def_val, def_err)
            ax_errorbar.errorbar([def_title], [def_val], [def_err], color=def_color, ls='none', marker='o',
                                 label=def_title)
            ax_errorbar.errorbar([def_title], [def_val], [barlow], color=def_color, linewidth=5, alpha=0.5,
                                 ls='none', marker=None)

            group_df_sys = sort_on_sys_info_dict(group_df_sys, sys_info_dict)
            group_df_sys = group_df_sys.sort_values(by=['sys_type_index', 'sys_val_index'])
            for sys_type in group_df_sys['sys_type'].unique():
                sys_type_df = group_df_sys[group_df_sys['sys_type'] == sys_type]
                # sys_type_df = sys_type_df.sort_values('sys_val')
                # print(f'Sys_type_df:')
                # print(sys_type_df)
                # print(f'Sys_type_df colors:')
                # print(sys_type_df['color'])
                # print(sys_type_df['color'].values)
                sys_type_sum_df = sys_type_df[sys_type_df['include_sys']]
                ax_errorbar.errorbar(sys_type_df['sys_val_str'], sys_type_df[val_col], sys_type_df[err_col],
                                     color=sys_type_df['color'].values[0], ls='none', marker='o', zorder=9,
                                     label=sys_info_dict[sys_type][name_col])
                ax_errorbar.scatter(sys_type_sum_df['sys_val_str'], sys_type_sum_df[val_col], color='black', zorder=10,
                                    marker='o', s=4)
                ax_errorbar.scatter(sys_type_sum_df['sys_val_str'], sys_type_sum_df[val_col], color='yellow', zorder=11,
                                    marker='o', s=1)
            if plot_bars:
                bars = ax_bar.bar([def_title] + list(group_df_sys['sys_val_str']),
                                  [barlow] + list(group_df_sys['barlow_sum'].values),
                                  color=[def_color] + list(group_df_sys['color']))
                ax_bar.plot([bars[0].get_x(), bars[0].get_x() + bars[0].get_width()], [def_err, def_err], color='red')
                # ax_bar.set_xticklabels([])
            ax_errorbar.set_xlim(right=ax_errorbar.get_xlim()[-1] + ax_errorbar.get_xlim()[-1] * 0.15)
            ax_errorbar.tick_params(axis='x', rotation=45)
            for label in ax_errorbar.get_xticklabels():
                label.set_ha('right')
            ax_errorbar.legend()
            fig.subplots_adjust(hspace=0.0)  # Adjust the vertical spacing between subplots
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.0)  # Adjust the vertical spacing between subplots
            if plot_barlow_decomp:
                fig_barlow_decomp.tight_layout()

            if pdf_out_path is not None or indiv_pdf_path is not None:
                if pdf_out_path is not None:
                    pdf_pages.savefig(fig)
                if indiv_pdf_path is not None:
                    fig_name = indiv_pdf_path + '_'.join([f'{col}{val}' for col, val in zip(group_cols, group_name)])
                    fig.savefig(fig_name + '.pdf', format='pdf')
                plt.close(fig)
                if plot_barlow_decomp:
                    pdf_pages.savefig(fig_barlow_decomp)
                    plt.close(fig_barlow_decomp)

    if pdf_out_path is not None:
        pdf_pages.close()


def plot_sys_table(df, df_def_name, df_sys_set_names, sys_info_dict, sys_prior_dict=None, group_cols=None,
                   val_col='val', err_col='err', name_col='name', indiv_pdf_path=None):
    if group_cols is None:
        group_cols = ['divs', 'energy', 'cent', 'data_type', 'total_protons']

    cents = pd.unique(df['cent'])
    energies = pd.unique(df['energy'])
    if 'divs' in df:
        divs = pd.unique(df['divs'])
    else:
        divs = [None]

    def_title = 'default'
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: 7.7, 11: 11.5, 19: 19.6, 27: 27, 39: 39, 62: 62.4}
    title_maps = {'divs': lambda div: f'{div}°',
                  'energy': lambda energy: f'{energy}GeV',
                  'cent': lambda cent: f'{cent_map[cent]}'}

    df_filtered = df[df[name_col].isin(list(df_sys_set_names) + [df_def_name])]
    df_filtered[['sys_type', 'sys_val']] = df_filtered[name_col].apply(lambda x: pd.Series(split_sys_type_val(x)))

    sys_types = [sys_type for sys_type in pd.unique(df_filtered['sys_type']) if sys_type != df_def_name]
    sys_types = [sys_type for sys_type in sys_info_dict.keys() if sys_type in sys_types]  # Sort correctly
    sys_type_i_map = {sys_type: i for i, sys_type in enumerate([df_def_name] + sys_types)}
    print(sys_type_i_map)
    sys_names = [def_title] + [sys_info_dict[sys_type][name_col] for sys_type in sys_types]

    for div in divs:
        fig, axes = plt.subplots(1, len(cents), figsize=(14, 5), dpi=144, sharey='all')
        vmin = 0
        vmax = 100
        if div is not None:
            fig.suptitle(f"{title_maps['divs'](div)} Partitions")
            df_div = df_filtered[df_filtered['divs'] == div]
        else:
            df_div = df_filtered

        # Iterate through centralities and create a heatmap for each
        for i, cent in enumerate(sorted(cents, reverse=True)):
            ax = axes[i]
            ax.set_title(f'{title_maps["cent"](cent)} Centrality')
            df_cent = df_div[df_div['cent'] == cent]

            hm_array = np.full((len(sys_names), len(energies)), float('nan'))
            for energy_i, energy in enumerate(energies):
                df_e = df_cent[df_cent['energy'] == energy]
                df_def = df_e[df_e[name_col] == df_def_name]
                if len(df_def) != 1:
                    print('Bad df_def: \n', df_def)
                assert len(df_def) == 1
                def_val, def_err = df_def[val_col].values[0], df_def[err_col].values[0]

                if len(df_e) <= 1:
                    pass  # No systematic values so sys = 0
                else:
                    df_sys = df_e[df_e[name_col] != df_def_name].copy()
                    df_sys = add_sys_info(df_sys, sys_info_dict, name_col)

                    sys_val = df_sys[val_col].values
                    sys_err = df_sys[err_col].values

                    barlow_i = calc_sys(def_val, def_err, sys_val, sys_err, 'indiv')
                    # Where prior is 'flat_one_side' or 'flat_two_side', divide by sqrt(12) or sqrt(3) respectively.
                    # Otherwise, assume gaussian and divide by 1
                    # Use sys_prior_dict with group_df_sys['sys_type'] to get series of priors
                    priors = df_sys['sys_type'].map(sys_prior_dict)
                    prior_divisors = priors.map(
                        lambda x: np.sqrt(12) if x == 'flat_one_side' else np.sqrt(3) if x == 'flat_two_side' else 1)
                    barlow_i = barlow_i / prior_divisors
                    df_sys['barlow'] = barlow_i
                    df_sys['barlow_sum'] = barlow_i

                    df_sys['include_sys'] = df_sys.apply(
                        lambda row: False if sys_info_dict[row['sys_type']]['sys_vars'] is None else
                        row['sys_val'] in sys_info_dict[row['sys_type']]['sys_vars'], axis=1)

                    max_indices = []  # For each systematic find index of variation to count in systematic total
                    for sys_type in df_sys['sys_type'].unique():
                        sys_mask = (df_sys['sys_type'] == sys_type) & (df_sys['include_sys'])  # Only 'sys_vars'
                        sys_barlows = df_sys.loc[sys_mask, 'barlow_sum']
                        if len(sys_barlows) > 0:
                            max_indices.append(df_sys.loc[sys_mask, 'barlow_sum'].idxmax())
                    df_sys.loc[~df_sys.index.isin(max_indices), 'barlow_sum'] = 0

                    barlow = np.sqrt(np.sum(df_sys['barlow_sum'].values ** 2))

                    df_sys = sort_on_sys_info_dict(df_sys, sys_info_dict)
                    df_sys = df_sys.sort_values(by=['sys_type_index', 'sys_val_index'])
                    hm_array[0, energy_i] = abs(barlow / def_val)
                    for sys_type in sys_types:  # Only one value per sys_type is summed in the barlow systematics
                        barlow_i = df_sys[df_sys['sys_type'] == sys_type]['barlow_sum'].max()
                        hm_array[sys_type_i_map[sys_type], energy_i] = (barlow_i / barlow) ** 2
            sns.heatmap(hm_array * 100, cmap='YlOrRd', annot=True, fmt='.0f',
                        xticklabels=[energy_map[e] for e in energies],
                        yticklabels=sys_names, ax=ax, vmin=vmin, vmax=vmax, cbar=False, mask=~np.isfinite(hm_array))
            for t in ax.texts:
                t.set_text(t.get_text() + "%")
            ax.tick_params(left=False, right=False, top=False, bottom=False)
            ax.axhline(y=1, color='white', linewidth=10)
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        fig.tight_layout()

        if indiv_pdf_path is not None:
            pdf_name = f'{indiv_pdf_path.replace(".pdf", "")}_div{div}.pdf' if div is not None else indiv_pdf_path
            fig.savefig(pdf_name, format='pdf')
            plt.close(fig)


def dvar_vs_protons(df, div, cent, energies, data_types, data_sets_plt, y_ranges=None, plot=False, avg=False,
                    hist=False, data_sets_colors=None, data_sets_labels=None, star_prelim_loc=None, alpha=0.6,
                    marker_map=None, errbar_alpha=0.2, legend_pos='lower center', ylabel=None, xlim=None,
                    kin_info_loc=(0.26, 0.91), title=None, data_sets_bands=None, legend_order=None, leg=True,
                    ax_in=None, ylim=None, print_data=False):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    data = []
    for data_type in data_types:
        for data_set in data_sets_plt:
            for energy in energies:
                df_pre = df
                if 'data_type' in df_pre:
                    df_pre = df_pre[df_pre['data_type'] == data_type]
                if 'cent' in df_pre:
                    df_pre = df_pre[df_pre['cent'] == cent]
                df_set = df_pre[(df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                if len(df_set) == 0:
                    continue
                if data_sets_labels is None:
                    if energy == 'sim':
                        lab = f'{data_set}_{data_type}'
                    else:
                        lab = f'{data_set}_{data_type}_{energy}GeV'
                else:
                    if type(data_sets_labels[data_set]) == dict:
                        lab = data_sets_labels[data_set][data_type]
                    else:
                        lab = data_sets_labels[data_set]
                if len(data_types) > 1 and data_sets_labels is not None and type(data_sets_labels[data_set]) != dict:
                    if len(data_sets_plt) == 1:
                        if data_type == 'raw':
                            lab = 'Single Event'
                        elif data_type == 'mix':
                            lab = 'Mixed Event'
                        elif data_type == 'diff':
                            lab = 'Single Event - Mixed Event'
                        elif data_type == 'v2_sub':
                            lab = 'Single Event - Mixed Event - v2 Correction'
                    else:
                        lab += f' {data_type.capitalize()}'
                df_set = df_set.sort_values(by=['total_protons'])
                amp, spread = df_set['amp'].iloc[0], df_set['spread'].iloc[0]
                data.append((df_set, lab, data_set, data_type, amp, spread, energy))

    if plot:
        if ax_in is None:
            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(f'dvar vs Total Protons {energy}GeV {div}°')
        else:
            ax = ax_in
        if title is None:
            ax.set_title(f'{energy} GeV, {cent_map[cent]} Centrality, {div}° Partitions, 72 Samples/Event')
        else:
            ax.set_title(title)
        ax.set_xlabel(r'Total Protons in Event ($N$)')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif len(data_types) == 1:
            if data_types[0] == 'raw':
                ax.set_ylabel(rf'Single Event $\Delta \sigma^2$')
            elif data_types[0] == 'mix':
                ax.set_ylabel(rf'Mixed Event $\Delta \sigma^2$')
            elif data_types[0] == 'diff':
                ax.set_ylabel(r'$\Delta \sigma^2_{Single} - \Delta \sigma^2_{Mixed}$')
            else:
                ax.set_ylabel(rf'$\Delta \sigma^2$')
        else:
            ax.set_ylabel(rf'$\Delta \sigma^2$')
        ax.axhline(0, ls='-', color='gray', zorder=0)
        if y_ranges:
            ax.set_ylim(y_ranges)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))

    avgs = []
    for i, (df, lab, data_set, data_type, amp, spread, energy) in enumerate(data):
        zo = len(data) - i + 4
        if plot:
            if data_sets_colors is None:
                if len(data_types) > 1:
                    if data_type == 'raw':
                        c = 'blue'
                    elif data_type == 'mix':
                        c = 'green'
                    elif data_type in ['diff', 'sub']:
                        c = 'red'
                    elif data_type == 'v2_sub':
                        c = 'black'
                else:
                    c = next(color)
            else:
                if type(data_sets_colors[data_set]) == dict:
                    c = data_sets_colors[data_set][data_type]
                else:
                    c = data_sets_colors[data_set]
            if data_sets_bands is not None and data_set in data_sets_bands:
                ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'], label=lab, color=c,
                                alpha=0.4)
            else:
                if star_prelim_loc is not None and y_ranges is not None:
                    df_plt = df[(df['val'] < y_ranges[1]) & (df['val'] > y_ranges[0])]
                else:
                    df_plt = df
                marker = marker_map[data_set][data_type] if marker_map is not None else 'o'
                ax.errorbar(df_plt['total_protons'], df_plt['val'], df_plt['err'], label=lab,
                            marker=marker, ls='', color=c, alpha=alpha, zorder=zo)
                if print_data:
                    print(f'\n{data_set} {data_type} {energy}GeV {div}°')
                    print(f'\'n\': [{", ".join([str(x) for x in df_plt["total_protons"].values])}],')
                    print(f'\'val\': [{", ".join([str(x) for x in df_plt["val"].values])}],')
                    print(f'\'err\': [{", ".join([str(x) for x in df_plt["err"].values])}],')
                if 'sys' in df:
                    ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='', elinewidth=4,
                                color=c, alpha=errbar_alpha, zorder=zo)
                    if print_data:
                        print(f'\'sys\': [{", ".join([str(x) for x in df["sys"].values])}]')

        if avg and len(df) > 1:
            weight_avg = np.average(df['val'], weights=1 / df['err'] ** 2)
            weight_avg_err = np.sqrt(1 / np.sum(1 / df['err'] ** 2))
            # def const(x, a):
            #     return a
            # 
            # popt, pcov = cf(const, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
            # weight_avg, weight_avg_err = popt[0], np.sqrt(pcov[0][0])
            avgs.append({'name': data_set, 'energy': energy, 'divs': div, 'cent': cent,
                         'data_type': data_type, 'amp': amp, 'spread': spread,
                         'avg': weight_avg, 'avg_err': weight_avg_err, 'avg_meas': Measure(weight_avg, weight_avg_err)})
            if plot:
                ax.axhline(weight_avg, ls='--', color=c)
                ax.axhspan(weight_avg + weight_avg_err, weight_avg - weight_avg_err, alpha=0.4, color=c)
                if hist:
                    sigs = (df['val'] - weight_avg) / df['err']  # Should probably add weight_avg_err in quad_func
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.set_title(f'{lab}')
                    ax_hist.set_xlabel('Standard Deviations from Average')
                    sns.histplot(sigs, stat='density', kde=True)
                    x_norm = np.linspace(min(sigs), max(sigs), 1000)
                    ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                    ax_hist.legend()
                    fig_hist.tight_layout()

    if plot:
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if kin_info_loc is not None:
            eta_line = r'|y| < 0.5'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}', ha='left', va='top',
                    transform=ax.transAxes)
        if star_prelim_loc is not None:
            ax.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', va='top',
                    transform=ax.transAxes)
            # ax.legend(loc=legend_pos, framealpha=1.0).set_zorder(10)
        # else:
        #     ax.legend(loc=legend_pos).set_zorder(10)
        if leg:
            if legend_order is not None:
                handles, labels = ax.get_legend_handles_labels()
                handles_dict = dict(zip(labels, handles))
                ordered_handles = [handles_dict[label] for label in legend_order]
                ax.legend(loc=legend_pos, handles=ordered_handles, labels=legend_order).set_zorder(10)
            else:
                ax.legend(loc=legend_pos).set_zorder(10)
        if ax_in is None:
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, right=0.995, bottom=0.083, left=0.154)

    return pd.DataFrame(avgs)


def stat_vs_protons_divs(df, stat, divs, cent, energies, data_types, data_sets_plt, y_ranges=None, plot=False,
                         fit=False, hist=False, data_sets_colors=None, data_sets_labels=None):
    div_data = []
    for div in divs:
        data = []
        for data_type in data_types:
            for data_set in data_sets_plt:
                for energy in energies:
                    df_pre = df
                    if 'data_type' in df_pre:
                        df_pre = df_pre[df_pre['data_type'] == data_type]
                    if 'cent' in df_pre:
                        df_pre = df_pre[df_pre['cent'] == cent]
                    if 'stat' in df_pre:
                        df_pre = df_pre[df_pre['stat'] == stat]
                    df_set = df_pre[
                        (df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                    if len(df_set) == 0:
                        continue
                    if data_sets_labels is None:
                        if energy == 'sim':
                            lab = f'{data_set}_{data_type}'
                        else:
                            lab = f'{data_set}_{data_type}_{energy}GeV'
                    else:
                        lab = data_sets_labels[data_set]
                    df_set = df_set.sort_values(by=['total_protons'])
                    data.append((df_set, lab, data_set, df_set['amp'].iloc[0], df_set['spread'].iloc[0]))
        div_data.append(data)

    if plot:
        fig, ax_divs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        ax_divs = ax_divs.flat
        fig.suptitle(f'{energy}GeV')
        # ax.set_title(f'{stat} {div}° Divisions')
        for ax in ax_divs[-3:]:
            ax.set_xlabel('Total Protons in Event')
        for i, ax in enumerate(ax_divs):
            ax.axhline(1, ls='--', color='black')
            if i == 6:
                ax.set_ylabel(f'Raw / Mix {stat}')
            # if y_ranges:
            #     ax.set_ylim(y_ranges[stat])
            if stat == 'standard deviation':
                ax.set_ylim(0.921, 1.019)
            elif stat == 'skewness':
                ax.set_ylim(0.71, 1.09)
            elif stat == 'non-excess kurtosis':
                ax.set_ylim(0.92, 1.06)
            ax.set_xlim(0, 65)

    fits = []
    for data, ax, div in zip(div_data, ax_divs[:len(divs)], divs):
        if plot:
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for i, (df, lab, data_set, amp, spread) in enumerate(data):
            zo = len(data) - i + 4
            if plot:
                if data_sets_colors is None:
                    c = next(color)
                else:
                    c = data_sets_colors[data_set]
                ax.text(5, .94, f'{div}°')
                if 'sim_' in data_set:
                    ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'],
                                    label=lab, color=c, alpha=0.4)
                else:
                    ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab,
                                marker='o', ls='', color=c, alpha=0.7, zorder=zo)
                    if 'sys' in df:
                        ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='',
                                    elinewidth=3, color=c, alpha=0.4, zorder=zo)

            if fit and len(df) > 1:
                # popt, pcov = cf(line, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
                popt, pcov = cf(line_yint1, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
                popt = np.append(popt, 1)
                pcov = np.array(((pcov[0][0], 0), (0, 0)))
                fits.append({'name': data_set, 'divs': div, 'amp': amp, 'spread': spread, 'slope': popt[0],
                             'slope_err': np.sqrt(np.diag(pcov))[0]})
                if plot:
                    ax.plot(df['total_protons'], line(df['total_protons'], *popt), ls='--', color=c)
                    if hist:
                        sigs = (df['val'] - line(df['total_protons'], *popt)) / df['err']
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.set_title(f'{lab}')
                        ax_hist.set_xlabel('Standard Deviations from Linear Fit')
                        sns.histplot(sigs, stat='density', kde=True)
                        x_norm = np.linspace(min(sigs), max(sigs), 1000)
                        ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                        ax_hist.legend()
                        fig_hist.tight_layout()

    if plot:
        data, ax, div = div_data[-1], ax_divs[-1], divs[-1]
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for df, lab, data_set, amp, spread in data:
            if data_sets_colors is None:
                c = next(color)
            else:
                c = data_sets_colors[data_set]
            if 'sim_' in data_set:
                ax.fill_between([], [], [], label=lab, color=c, alpha=0.4)
            else:
                ax.errorbar([], [], [], label=lab, marker='o', ls='', color=c, alpha=0.7)
                if 'sys' in df:
                    ax.errorbar([], [], [], marker='', ls='', elinewidth=3, color=c, alpha=0.4)
        ax_divs[-1].legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title(f'binom_slices_{energies[0]}GeV_{stat}')

    return pd.DataFrame(fits)


def stat_vs_protons_energies(df, stat, divs, cent, energies, data_types, data_sets_plt, y_ranges=None, plot=False,
                             fit=False, plot_fit=False, hist=False, data_sets_colors=None, data_sets_labels=None,
                             star_prelim=False):
    energy_data = []
    for energy in energies:
        data = []
        for data_type in data_types:
            for data_set in data_sets_plt:
                for div in divs:
                    df_pre = df
                    if 'data_type' in df_pre:
                        df_pre = df_pre[df_pre['data_type'] == data_type]
                    if 'cent' in df_pre:
                        df_pre = df_pre[df_pre['cent'] == cent]
                    if 'stat' in df_pre:
                        df_pre = df_pre[df_pre['stat'] == stat]
                    df_set = df_pre[
                        (df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                    if len(df_set) == 0:
                        continue
                    if data_sets_labels is None:
                        if energy == 'sim':
                            lab = f'{data_set}_{data_type}'
                        else:
                            lab = f'{data_set}_{data_type}_{energy}GeV'
                    else:
                        lab = data_sets_labels[data_set]
                    df_set = df_set.sort_values(by=['total_protons'])
                    data.append((df_set, lab, data_set, df_set['amp'].iloc[0], df_set['spread'].iloc[0]))
        energy_data.append(data)

    if plot or plot_fit:
        fig, ax_energies = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        ax_energies = ax_energies.flat
        fig.suptitle(f'0-5% Centrality, {div}° Partitions, 72 Samples/Event')
        # ax.set_title(f'{stat} {div}° Divisions')
        for ax in ax_energies[-3:]:
            ax.set_xlabel('Total Protons in Event')
        for i, ax in enumerate(ax_energies):
            ax.axhline(1, ls='-', color='gray')
            if i in [0, 3]:
                ax.set_ylabel(f'Raw / Mix {stat.title()}')
            # if y_ranges:
            #     ax.set_ylim(y_ranges[stat])
            if stat == 'standard deviation':
                ax.set_ylim(0.921, 1.019)
            elif stat == 'skewness':
                ax.set_ylim(0.71, 1.09)
            elif stat == 'non-excess kurtosis':
                ax.set_ylim(0.92, 1.06)
            ax.set_xlim(0, 69.9)

    fits = []
    for data, ax, energy in zip(energy_data, ax_energies[:len(energies)], energies):
        if plot or plot_fit:
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for i, (df, lab, data_set, amp, spread) in enumerate(data):
            zo = len(data) - i + 4
            if plot or plot_fit:
                if data_sets_colors is None:
                    c = next(color)
                else:
                    c = data_sets_colors[data_set]
                ax.text(12, .925, f'{energy} GeV', size='x-large')
            if plot:
                if 'sim_' in data_set:
                    ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'],
                                    label=lab, color=c, alpha=0.4)
                else:
                    ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab,
                                marker='o', ls='', color=c, alpha=0.7, zorder=zo)
                    if 'sys' in df:
                        ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='',
                                    elinewidth=3, color=c, alpha=0.4, zorder=zo)

            if fit and len(df) > 1:
                # popt, pcov = cf(line, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
                popt, pcov = cf(line_yint1, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
                popt = np.append(popt, 1)
                pcov = np.array(((pcov[0][0], 0), (0, 0)))
                fits.append({'name': data_set, 'divs': div, 'amp': amp, 'spread': spread, 'slope': popt[0],
                             'slope_err': np.sqrt(np.diag(pcov))[0]})
                if plot_fit:
                    if plot:
                        ax.plot(df['total_protons'], line(df['total_protons'], *popt), ls='--', color=c)
                    else:
                        ax.plot(df['total_protons'], line(df['total_protons'], *popt), ls='--', color=c, label=lab)
                    if hist:
                        sigs = (df['val'] - line(df['total_protons'], *popt)) / df['err']
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.set_title(f'{lab}')
                        ax_hist.set_xlabel('Standard Deviations from Linear Fit')
                        sns.histplot(sigs, stat='density', kde=True)
                        x_norm = np.linspace(min(sigs), max(sigs), 1000)
                        ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                        ax_hist.legend()
                        fig_hist.tight_layout()

    if plot or plot_fit:
        # data, ax, energy = energy_data[-1], ax_energies[-1], energies[-1]
        # color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        # for df, lab, data_set, amp, spread in data:
        #     if data_sets_colors is None:
        #         c = next(color)
        #     else:
        #         c = data_sets_colors[data_set]
        #     if 'sim_' in data_set:
        #         ax.fill_between([], [], [], label=lab, color=c, alpha=0.4)
        #     else:
        #         ax.errorbar([], [], [], label=lab, marker='o', ls='', color=c, alpha=0.7)
        #         if 'sys' in df:
        #             ax.errorbar([], [], [], marker='', ls='', elinewidth=3, color=c, alpha=0.4)
        ax_energies[-3].legend(loc='lower right', framealpha=1.0).set_zorder(10)
        if star_prelim:
            ax_energies[4].text(46, 0.93, 'STAR \nPreliminary', fontsize='large')
            eta_line = r'|$\eta$| < 1'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax_energies[4].text(46, 0.95, f'Au+Au\n{eta_line}\n{pt_line}')
            ax_energies[2].text(15, 1.012, 'statistical uncertainty only')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title(f'binom_slices_{divs[0]}_{stat}')

    return pd.DataFrame(fits)


def dvar_vs_protons_energies(df, divs, cent, energies, data_types, data_sets_plt, y_ranges=None, plot=False,
                             avg=False, plot_avg=False, hist=False, data_sets_colors=None, data_sets_labels=None,
                             star_prelim_loc=None, marker_map=None, alpha=1.0, errbar_alpha=0.2, avgs_df=None,
                             ylabel=None, kin_loc=None, no_hydro_label=False, data_sets_bands=None,
                             legend_order=None, title=None, ylim=None, legend_panel=-1, fig_splt_adjust=None,
                             plot_letters=False, print_data=False):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    energy_data = []
    for energy in energies:
        data = []
        for data_type in data_types:
            for data_set in data_sets_plt:
                for div in divs:
                    df_pre = df
                    if 'data_type' in df_pre:
                        df_pre = df_pre[df_pre['data_type'] == data_type]
                    if 'cent' in df_pre:
                        df_pre = df_pre[df_pre['cent'] == cent]
                    df_set = df_pre[
                        (df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                    if len(df_set) == 0:
                        continue
                    if data_sets_labels is None:
                        if energy == 'sim':
                            lab = f'{data_set}_{data_type}'
                        else:
                            lab = f'{data_set}_{data_type}_{energy}GeV'
                    else:
                        lab = data_sets_labels[data_set]
                    if len(data_types) > 1:
                        lab += f' {data_type.capitalize()}'
                    df_set = df_set.sort_values(by=['total_protons'])
                    data.append(
                        (df_set, lab, data_set, data_type, div, df_set['amp'].iloc[0], df_set['spread'].iloc[0]))
        energy_data.append(data)

    if plot or plot_avg:
        # fig, ax_energies = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        fig, ax_energies = plt.subplots(2, 3, sharex='all', figsize=(13.33, 5.1), dpi=144)
        ax_energies = ax_energies.flat
        if title is None:
            fig.suptitle(f'{cent_map[cent]} Centrality, {div}° Partitions, 72 Samples/Event')
        else:
            fig.suptitle(title)
        for ax in ax_energies[-3:]:
            ax.set_xlabel(r'Total Protons in Event ($N$)', fontsize='x-large')
        for i, ax in enumerate(ax_energies):
            ax.axhline(0, ls='-', color='gray', zorder=0)
            if i in [0, 3]:
                if ylabel is not None:
                    ax.set_ylabel(ylabel, fontsize='x-large')
                elif len(data_types) == 1:
                    if data_types[0] == 'raw':
                        ax.set_ylabel(rf'Single Event $\Delta \sigma^2$')
                    elif data_types[0] == 'mix':
                        ax.set_ylabel(rf'Mixed Event $\Delta \sigma^2$')
                    elif data_types[0] == 'diff':
                        ax.set_ylabel(r'$\Delta \sigma^2_{Single} - \Delta \sigma^2_{Mixed}$')
                    else:
                        ax.set_ylabel(rf'$\Delta \sigma^2$')
                else:
                    ax.set_ylabel(rf'$\Delta \sigma^2$')
            if i not in [0, 3]:
                ax.set_yticks([])
            if y_ranges:
                ax.set_ylim(y_ranges)
            ax.set_xlim(0, 69.9)
    else:
        ax_energies = [None] * len(energies)

    avgs = []
    for energy_i, (data, ax, energy) in enumerate(zip(energy_data, ax_energies[:len(energies)], energies)):
        if plot or plot_avg:
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
            ax.text(0.95, 0.97, f'{energy_map[energy]} GeV', size='x-large', ha='right', va='top',
                    transform=ax.transAxes)
            if plot_letters:
                ax.text(0.4, 0.97, f'({chr(97 + energy_i)})', size='x-large', ha='center', va='top',
                        transform=ax.transAxes)
        for i, (df, lab, data_set, data_type, div, amp, spread) in enumerate(data):
            zo = len(data) - i + 4
            if plot or plot_avg:
                if data_sets_colors is None:
                    c = next(color)
                else:
                    c = data_sets_colors[data_set]
            if plot:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'],
                                    label=lab, color=c, alpha=0.4)
                else:
                    if star_prelim_loc is not None and y_ranges is not None:
                        df_plt = df[(df['val'] < y_ranges[1]) & (df['val'] > y_ranges[0])]
                    else:
                        df_plt = df
                    marker = marker_map[data_set][data_type] if marker_map is not None else 'o'
                    ax.errorbar(df_plt['total_protons'], df_plt['val'], df_plt['err'], label=lab,
                                marker=marker, ls='', color=c, alpha=alpha, zorder=zo)
                    if 'sys' in df:
                        ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='',
                                    elinewidth=3, color=c, alpha=errbar_alpha, zorder=zo)

            if avgs_df is not None and plot_avg:
                avg_df = avgs_df[(avgs_df['name'] == data_set) & (avgs_df['energy'] == energy) &
                                 (avgs_df['divs'] == div) & (avgs_df['cent'] == cent) &
                                 (avgs_df['data_type'] == data_type)]
                if len(avg_df) != 1:
                    print(f'Degenerate! {avgs_df}')
                else:
                    avg, avg_err, avg_sys = avg_df['avg'].iloc[0], avg_df['avg_err'].iloc[0], avg_df['sys'].iloc[0]
                    ax.axhline(avg, ls='--', color=c)
                    ax.axhspan(avg + avg_err, avg - avg_err, alpha=0.4, color=c)
                    ax.axhspan(avg + avg_sys, avg - avg_sys, alpha=0.4, color=c)
            if avg and len(df) > 1:
                weight_avg = np.average(df['val'], weights=1 / df['err'] ** 2)
                weight_avg_err = np.sqrt(1 / np.sum(1 / df['err'] ** 2))
                avgs.append({'name': data_set, 'energy': energy, 'divs': div, 'amp': amp, 'spread': spread,
                             'avg': weight_avg, 'avg_err': weight_avg_err, 'cent': cent,
                             'avg_meas': Measure(weight_avg, weight_avg_err)})

                if plot_avg and avgs_df is None:
                    if plot:
                        ax.axhline(weight_avg, ls='--', color=c)
                    else:
                        ax.axhline(weight_avg, ls='--', color=c, label=lab)
                    ax.axhspan(weight_avg + weight_avg_err, weight_avg - weight_avg_err, alpha=0.4, color=c)
                    if hist:
                        sigs = (df['val'] - weight_avg) / df['err']  # Should probably add weight_avg_err in quad_func
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.set_title(f'{lab}')
                        ax_hist.set_xlabel('Standard Deviations from Average', fontsize='large')
                        sns.histplot(sigs, stat='density', kde=True)
                        x_norm = np.linspace(min(sigs), max(sigs), 1000)
                        ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                        ax_hist.legend()
                        fig_hist.tight_layout()
            if print_data:
                print(f'\n{data_set} {data_type} {energy}GeV {div}°')
                print(f'\'n\': [{", ".join([str(x) for x in df["total_protons"].values])}],')
                print(f'\'val\': [{", ".join([str(x) for x in df["val"].values])}],')
                print(f'\'err\': [{", ".join([str(x) for x in df["err"].values])}],')
                if 'sys' in df:
                    print(f'\'sys\': [{", ".join([str(x) for x in df["sys"].values])}]')
                if avgs_df is not None:
                    print(f'\'avg\': {avg}, \'avg_err\': {avg_err}, \'avg_sys\': {avg_sys}')

    if plot or plot_avg:
        if len(data_sets_plt) * len(data_types) > 1:
            if legend_order is not None:
                handles, labels = ax.get_legend_handles_labels()
                handles_dict = dict(zip(labels, handles))
                ordered_handles = [handles_dict[label] for label in legend_order]
                ax_energies[legend_panel].legend(loc='lower right', framealpha=1.0, handles=ordered_handles,
                                                 labels=legend_order).set_zorder(10)
            else:
                ax_energies[legend_panel].legend(loc='lower right', framealpha=1.0).set_zorder(10)
        eta_line = r'|y| < 0.5'
        pt_line = r'0.4 < $p_T$ < 2.0 GeV'
        pt_line = f'{pt_line}\n{cent_map[cent]} Centrality\n{div}° Partitions'
        samples_line = None  # '72 Samples/Event'
        if kin_loc is None:
            kin_loc = (0.5, 0.65)
        kin_text = f'Au+Au\n{eta_line}\n{pt_line}'
        kin_text = kin_text + f'\n{samples_line}' if samples_line is not None else kin_text
        ax_energies[4].text(*kin_loc, kin_text, ha='left', va='top', transform=ax_energies[4].transAxes)
        if star_prelim_loc is not None:
            ax_energies[star_prelim_loc[0]].text(*star_prelim_loc[1:], 'STAR Preliminary', fontsize='large', ha='left',
                                                 va='top', transform=ax_energies[star_prelim_loc[0]].transAxes)
        if no_hydro_label:
            ax_energies[1].text(0.76, 0.1, f'No MUSIC data\nat 11.5 GeV', ha='center', va='center', fontsize='small',
                                transform=ax_energies[1].transAxes)
        if ylim is not None:
            ax_energies[0].set_ylim(*ylim)
        fig.tight_layout()
        if fig_splt_adjust is None:
            fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.083, top=0.995, right=0.995, bottom=0.085)
        else:
            fig.subplots_adjust(**fig_splt_adjust)
        fig.canvas.manager.set_window_title(f'binom_slices_{divs[0]}')

    return pd.DataFrame(avgs)


def stat_vs_protons_cents(df, stat, divs, cents, energy, data_types, data_sets_plt, y_ranges=None, plot=False,
                          fit=False, plot_fit=False, hist=False, data_sets_colors=None, data_sets_labels=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    cent_data = []
    for cent in cents:
        data = []
        for data_type in data_types:
            for data_set in data_sets_plt:
                for div in divs:
                    df_pre = df
                    if 'data_type' in df_pre:
                        df_pre = df_pre[df_pre['data_type'] == data_type]
                    if 'cent' in df_pre:
                        df_pre = df_pre[df_pre['cent'] == cent]
                    if 'stat' in df_pre:
                        df_pre = df_pre[df_pre['stat'] == stat]
                    df_set = df_pre[
                        (df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                    if len(df_set) == 0:
                        continue
                    if data_sets_labels is None:
                        if energy == 'sim':
                            lab = f'{data_set}_{data_type}'
                        else:
                            lab = f'{data_set}_{data_type}_{energy}GeV'
                    else:
                        lab = data_sets_labels[data_set]
                    df_set = df_set.sort_values(by=['total_protons'])
                    data.append((df_set, div, lab, data_set, df_set['amp'].iloc[0], df_set['spread'].iloc[0]))
        cent_data.append(data)

    if plot or plot_fit:
        fig, ax_cents = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        ax_cents = ax_cents.flat
        fig.suptitle(f'{energy}GeV, {div}° Partitions, 72 Samples/Event')
        # ax.set_title(f'{stat} {div}° Divisions')
        for ax in ax_cents[-3:]:
            ax.set_xlabel('Total Protons in Event')
        ax_cents[3].set_ylabel(f'Raw / Mix {stat.title()}')
        for i, ax in enumerate(ax_cents):
            ax.axhline(1, ls='-', color='gray')
            # if y_ranges:
            #     ax.set_ylim(y_ranges[stat])
            if stat == 'standard deviation':
                ax.set_ylim(0.921, 1.019)
            elif stat == 'skewness':
                ax.set_ylim(0.71, 1.09)
            elif stat == 'non-excess kurtosis':
                ax.set_ylim(0.92, 1.06)
            ax.set_xlim(0, 65)

    fits = []
    for data, ax_index, cent in zip(cent_data, range(len(cents)), cents):
        if plot or plot_fit:
            ax = ax_cents[ax_index]
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for i, (df, div, lab, data_set, amp, spread) in enumerate(data):
            zo = len(data) - i + 4
            if plot or plot_fit:
                if data_sets_colors is None:
                    c = next(color)
                else:
                    c = data_sets_colors[data_set]
                ax.text(62, .925, f'{cent_map[cent]}', size='x-large', horizontalalignment='right')
            if plot:
                if 'sim_' in data_set:
                    ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'],
                                    label=lab, color=c, alpha=0.4)
                else:
                    ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab,
                                marker='o', ls='', color=c, alpha=0.7, zorder=zo)
                    if 'sys' in df:
                        ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='',
                                    elinewidth=3, color=c, alpha=0.4, zorder=zo)

            if fit and len(df) > 1:
                popt, pcov = cf(line_yint1, df['total_protons'], df['val'], sigma=df['err'], absolute_sigma=True)
                fits.append({'name': data_set, 'energy': energy, 'divs': div, 'amp': amp, 'spread': spread,
                             'slope': popt[0], 'slope_err': np.sqrt(pcov[0])[0], 'cent': cent})
                if plot_fit:
                    line_vals = line_yint1(df['total_protons'], *popt)
                    line_high = line_yint1(df['total_protons'], popt[0] + np.sqrt(pcov[0]))
                    line_low = line_yint1(df['total_protons'], popt[0] - np.sqrt(pcov[0]))
                    if plot:
                        ax.fill_between(df['total_protons'], line_high, line_low, color=c, alpha=0.3)
                        ax.plot(df['total_protons'], line_vals, ls='--', color=c)
                    else:
                        ax.fill_between(df['total_protons'], line_high, line_low, color=c, alpha=0.3)
                        ax.plot(df['total_protons'], line_yint1(df['total_protons'], *popt), ls='--', color=c,
                                label=lab)
                    if hist:
                        sigs = (df['val'] - line_yint1(df['total_protons'], *popt)) / df['err']
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.set_title(f'{lab}')
                        ax_hist.set_xlabel('Standard Deviations from Linear Fit')
                        sns.histplot(sigs, stat='density', kde=True)
                        x_norm = np.linspace(min(sigs), max(sigs), 1000)
                        ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                        ax_hist.legend()
                        fig_hist.tight_layout()

    if plot or plot_fit:
        for i, (df, div, lab, data_set, amp, spread) in enumerate(data):  # Just to get legend in last axis
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
            if data_sets_colors is None:
                c = next(color)
            else:
                c = data_sets_colors[data_set]
            if 'sim_' in data_set:
                ax_cents[-1].fill_between([], [], [], label=lab, color=c, alpha=0.4)
            else:
                ax_cents[-1].errorbar([], [], [], label=lab, marker='o', ls='', color=c, alpha=0.7)
                if 'sys' in df:
                    ax_cents[-1].errorbar([], [], [], marker='', ls='', elinewidth=3, color=c, alpha=0.4)
        ax_cents[-1].legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title(f'binom_slices_cents_{divs[0]}_{stat}')

    return pd.DataFrame(fits)


def dvar_vs_protons_cents(df, divs, cents, energy, data_types, data_sets_plt, y_ranges=None, plot=False,
                          avg=False, plot_avg=False, data_sets_colors=None, data_sets_labels=None, hist=False):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    cent_data = []
    for cent in cents:
        data = []
        for data_type in data_types:
            for data_set in data_sets_plt:
                for div in divs:
                    df_pre = df
                    if 'data_type' in df_pre:
                        df_pre = df_pre[df_pre['data_type'] == data_type]
                    if 'cent' in df_pre:
                        df_pre = df_pre[df_pre['cent'] == cent]
                    df_set = df_pre[
                        (df_pre['name'] == data_set) & (df_pre['divs'] == div) & (df_pre['energy'] == energy)]
                    if len(df_set) == 0:
                        continue
                    if data_sets_labels is None:
                        if energy == 'sim':
                            lab = f'{data_set}_{data_type}'
                        else:
                            lab = f'{data_set}_{data_type}_{energy}GeV'
                    else:
                        lab = data_sets_labels[data_set]
                    df_set = df_set.sort_values(by=['total_protons'])
                    data.append((df_set, div, lab, data_set, df_set['amp'].iloc[0], df_set['spread'].iloc[0],
                                 data_type))
        cent_data.append(data)

    if plot or plot_avg:
        fig, ax_cents = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        ax_cents = ax_cents.flat
        fig.suptitle(f'{energy}GeV, {div}° Partitions, 72 Samples/Event')
        # ax.set_title(f'{stat} {div}° Divisions')
        for ax in ax_cents[-3:]:
            ax.set_xlabel('Total Protons in Event')
        ax_cents[3].set_ylabel(r'$\Delta\sigma^2$')
        for i, ax in enumerate(ax_cents):
            ax.axhline(0, ls='-', color='gray')
            if y_ranges:
                ax.set_ylim(y_ranges)
            ax.set_xlim(0, 65)

    avgs = []
    for data, ax_index, cent in zip(cent_data, range(len(cents)), cents):
        if plot or plot_avg:
            ax = ax_cents[ax_index]
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for i, (df, div, lab, data_set, amp, spread, data_type) in enumerate(data):
            zo = len(data) - i + 4
            if plot or plot_avg:
                if data_sets_colors is None:
                    c = next(color)
                else:
                    c = data_sets_colors[data_set]
                ax.text(0.5, .92, f'{cent_map[cent]}', size='x-large', ha='center', va='top', transform=ax.transAxes)
            if plot:
                if 'sim_' in data_set:
                    ax.fill_between(df['total_protons'], df['val'] - df['err'], df['val'] + df['err'],
                                    label=lab, color=c, alpha=0.4)
                else:
                    ax.errorbar(df['total_protons'], df['val'], df['err'], label=lab,
                                marker='o', ls='', color=c, alpha=0.7, zorder=zo)
                    if 'sys' in df:
                        ax.errorbar(df['total_protons'], df['val'], df['sys'], marker='', ls='',
                                    elinewidth=3, color=c, alpha=0.4, zorder=zo)

            if avg and len(df) > 1:
                weight_avg = np.average(df['val'], weights=1 / df['err'] ** 2)
                weight_avg_err = np.sqrt(1 / np.sum(1 / df['err'] ** 2))
                avgs.append({'name': data_set, 'energy': energy, 'divs': div, 'amp': amp, 'spread': spread,
                             'data_type': data_type, 'avg': weight_avg, 'avg_err': weight_avg_err,
                             'avg_meas': Measure(weight_avg, weight_avg_err), 'cent': cent})
                if plot_avg:
                    ax.axhline(weight_avg, ls='--', color=c)
                    ax.axhspan(weight_avg + weight_avg_err, weight_avg - weight_avg_err, alpha=0.4, color=c)
                    if hist:
                        sigs = (df['val'] - weight_avg) / df['err']  # Should probably add weight_avg_err in quad_func
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.set_title(f'{lab}')
                        ax_hist.set_xlabel('Standard Deviations from Linear Fit')
                        sns.histplot(sigs, stat='density', kde=True)
                        x_norm = np.linspace(min(sigs), max(sigs), 1000)
                        ax_hist.plot(x_norm, norm(0, 1).pdf(x_norm), color='red', label='Standard Normal')
                        ax_hist.legend()
                        fig_hist.tight_layout()

    if plot or plot_avg:
        for i, (df, div, lab, data_set, amp, spread) in enumerate(data):  # Just to get legend in last axis
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
            if data_sets_colors is None:
                c = next(color)
            else:
                c = data_sets_colors[data_set]
            if 'sim_' in data_set:
                ax_cents[-1].fill_between([], [], [], label=lab, color=c, alpha=0.4)
            else:
                ax_cents[-1].errorbar([], [], [], label=lab, marker='o', ls='', color=c, alpha=0.7)
                if 'sys' in df:
                    ax_cents[-1].errorbar([], [], [], marker='', ls='', elinewidth=3, color=c, alpha=0.4)
        ax_cents[-1].legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title(f'dsigma_cents_{divs[0]}')

    return pd.DataFrame(avgs)


def plot_protons_fits_divs(df, data_sets_plt, fit=False, data_sets_colors=None, data_sets_labels=None, exclude_divs=[],
                           verbose=True, plt_energies=True, title=None, alpha=1):
    energies = pd.unique(df['energy'])
    if plt_energies:
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f'Slope vs Width All Energies')
        ax.axhline(0, ls='-', color='black')
        fig_panels, ax_panels = plt.subplots(2, 3, sharex=True, sharey=True, dpi=144, figsize=(13.33, 6.16))
        fig_panels.canvas.manager.set_window_title(f'Slope vs Width Energy Panels')
        ax_panels = dict(zip(energies, ax_panels.flat))
    energy_fig_axs = {energy: plt.subplots() for energy in energies}
    markers = ['o', 's', 'P', 'D', '*', '^', 'p']
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(energies) * len(data_sets_plt))))
    fit_pars = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='divs')
        for energy_marker, energy in enumerate(energies):
            df_energy = df_set[df_set['energy'] == energy]
            df_energy.sort_values(by='divs')
            if data_sets_labels is None:
                lab_energy = data_set
            else:
                lab_energy = data_sets_labels[data_set]
            if len(energies) > 1:
                lab = f'{lab_energy}_{energy}GeV'
            else:
                lab = lab_energy
            if data_sets_colors is None:
                color = next(colors)
            else:
                color = data_sets_colors[data_set]
            energy_fig, energy_ax = energy_fig_axs[energy]
            energy_ax.errorbar(df_energy['divs'], df_energy['slope'], yerr=df_energy['slope_err'], ls='none',
                               marker='o', label=lab_energy, color=color, alpha=alpha)
            if plt_energies:
                ax.errorbar(df_energy['divs'], df_energy['slope'], yerr=df_energy['slope_err'], ls='none',
                            marker=markers[energy_marker], label=lab, color=color, alpha=alpha)
                ax_panels[energy].errorbar(df_energy['divs'], df_energy['slope'], yerr=df_energy['slope_err'],
                                           ls='none',
                                           marker='o', label=lab_energy, color=color, alpha=alpha)
            if fit and df_energy.size > 1:
                try:
                    df_energy = df_energy[~df_energy.divs.isin(exclude_divs)]
                    popt, pcov = cf(quad_180, df_energy['divs'], df_energy['slope'], sigma=df_energy['slope_err'],
                                    absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))
                    fit_pars.append({'data_set': data_set, 'energy': energy, 'curvature': popt[0], 'color': color,
                                     'curve_baseline': popt[1], 'curve_err': perr[0], 'curve_base_err': perr[1],
                                     'spread': df_energy['spread'].iloc[0], 'amp': df_energy['amp'].iloc[0],
                                     'ap': df_energy['ap'].iloc[0], 'am': df_energy['am'].iloc[0],
                                     'sp': df_energy['sp'].iloc[0], 'sm': df_energy['sm'].iloc[0]})
                    x = np.linspace(0, 360, 100)
                    energy_ax.plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                    if plt_energies:
                        ax.plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                        ax_panels[energy].plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                    # p0 = [*popt, -popt[0], popt[1]]
                    # popt3, pcov3 = cf(double_quad_180_zparam, df_energy['divs'], df_energy['slope'], p0=p0,
                    #                   sigma=df_energy['slope_err'], absolute_sigma=True)
                    # perr3 = np.sqrt(np.diag(pcov))
                    # ax.plot(x, double_quad_180_zparam(x, *popt3), ls='--', color=color, alpha=0.65)
                    # energy_ax.plot(x, double_quad_180_zparam(x, *popt3), ls='--', color=color, alpha=0.65)
                    if popt[0] * popt[1] < 0:
                        popt2, pcov2 = cf(quad_180_zparam, df_energy['divs'], df_energy['slope'],
                                          sigma=df_energy['slope_err'],
                                          absolute_sigma=True)
                        perr2 = np.sqrt(np.diag(pcov2))
                        fit_pars[-1].update({'zero_mag': popt2[0], 'zero_mag_err': perr2[0], 'baseline': popt2[1],
                                             'base_err': perr2[1]})
                        if verbose:
                            print(
                                f'{data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\nz-c fit: {popt2}\n'
                                f'z-c covariance: {pcov2}')
                            print()
                    else:
                        if verbose:
                            print(f'No Zeros! {data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n')
                            print()
                except RuntimeError as e:
                    print(f'Fitting Error, skipping data_set {data_set}, {e}')

    if title is None:
        title = ''
        if len(data_sets_plt) == 1:
            title += f'{data_set}'
        if len(energies) == 1:
            if title != '':
                title += ' '
            title += f'{energies[0]}GeV'

    if plt_energies:
        if title != '' and title is not None:
            ax.set_title(title)
        ax.set_ylabel('Slope of Raw/Mix SD vs Total Protons per Event')
        ax.set_xlabel('Azimuthal Partition Width')
        ax.legend()

    for energy_i, (energy, (energy_fig, energy_ax)) in enumerate(energy_fig_axs.items()):
        if title is None:
            energy_ax.set_title(f'{energy} GeV')
        else:
            energy_ax.set_title(title)
        energy_ax.set_xlabel('Azimuthal Partition Width (w)')
        energy_ax.set_ylabel('Slope of k2 vs Total Protons per Event')
        energy_ax.axhline(0, color='black', zorder=0)
        energy_ax.legend()
        energy_fig.tight_layout()
        if title is None:
            energy_fig.canvas.manager.set_window_title(f'Slope vs Width {energy}GeV')
        else:
            energy_fig.canvas.manager.set_window_title(f'Slope vs Width {title}')

        if plt_energies:
            ax_panels[energy].axhline(0, color='black', zorder=0)
            ax_panels[energy].text(180, 0.00025, f'{energy} GeV', ha='center', fontsize='large')
            if energy_i >= 3:
                ax_panels[energy].set_xlabel('Azimuthal Partition Width')
            if energy_i in [0, 3]:
                ax_panels[energy].set_ylabel('Slope of Raw/Mix SD vs Protons/Event')
            if energy_i == 1:
                ax_panels[energy].legend(loc='upper center', bbox_to_anchor=(0.5, 0.85), framealpha=1.0)

    if plt_energies:
        fig.tight_layout()
        fig_panels.tight_layout()
        fig_panels.subplots_adjust(wspace=0.0, hspace=0.0)

    return pd.DataFrame(fit_pars)


def plot_dvar_avgs_divs(df, data_sets_plt, fit=False, data_sets_colors=None, data_sets_labels=None,
                        exclude_divs=[], verbose=False, plot_energy_panels=True, title=None, alpha=1, ylab=None,
                        plot=True, errbar_alpha=0.2, plot_indiv=True, plot_energies_fig=False, ylim=None, xlim=None,
                        leg_panel=0, kin_loc=(0.02, 0.02), star_prelim_loc=None, no_hydro_label=False, xlab=None,
                        data_sets_bands=None, legend_order=None, leg_frameon=False, data_sets_markers=None,
                        data_sets_fills=None, data_sets_ls=None, fig_splt_adj=None, panel_letters=False, leg=True,
                        print_data=False):
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    if xlab is None:
        xlab = r'Azimuthal Partition Width ($w$°)'
    energies = pd.unique(df['energy'])
    if plot:
        if plot_energies_fig:
            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(f'dsigma^2 vs Width All Energies')
            ax.axhline(0, ls='-', color='black')
        if plot_energy_panels:
            # fig_panels, ax_panels = plt.subplots(2, 3, sharex='all', sharey='all', dpi=144, figsize=(13.33, 6.16))
            fig_panels, ax_panels = plt.subplots(2, 3, sharex='all', dpi=144, figsize=(13.33, 5))
            fig_panels.canvas.manager.set_window_title(f'dsigma^2 vs Width Energy Panels')
            ax_panels = dict(zip(energies, ax_panels.flat))
        if plot_indiv:
            energy_fig_axs = {energy: plt.subplots() for energy in energies}
        markers = ['o', 's', 'P', 'D', '*', '^', 'p']
    if data_sets_colors is None:
        num_colorless = len(energies) * len(data_sets_plt) * len(pd.unique(df['cent']))
    else:
        data_sets_covered = len(set(data_sets_colors).intersection(set(data_sets_plt)))
        num_colorless = len(energies) * (len(data_sets_plt) - data_sets_covered) * len(pd.unique(df['cent']))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_colorless)))
    fit_pars = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='divs')
        for energy_marker, energy in enumerate(energies):
            df_energy = df_set[df_set['energy'] == energy]
            cents = pd.unique(df_energy['cent'])
            for cent in cents:
                df_cent = df_energy[df_energy['cent'] == cent]
                df_cent = df_cent[~df_cent.divs.isin(exclude_divs)]
                df_cent.sort_values(by='divs')
                if data_sets_colors is not None and data_set in data_sets_colors:
                    color = data_sets_colors[data_set]
                else:
                    color = next(colors)
                if plot:
                    if data_sets_labels is not None and data_set in data_sets_labels:
                        lab_energy = data_sets_labels[data_set]
                    else:
                        lab_energy = data_set
                    if len(energies) > 1:
                        lab = f'{lab_energy}_{energy}GeV'
                    else:
                        lab = lab_energy
                    if len(cents) > 1:
                        lab += f'_cent{cent}'

                    if data_sets_markers is not None and data_set in data_sets_markers:
                        if type(data_sets_markers[data_set]) is dict and 'diff' in data_sets_markers[data_set]:
                            marker = data_sets_markers[data_set]['diff']
                        else:
                            marker = data_sets_markers[data_set]
                    else:
                        marker = 'o'

                    if data_sets_fills is not None and data_set in data_sets_fills:
                        if type(data_sets_fills[data_set]) is dict and 'diff' in data_sets_fills[data_set]:
                            fill = data_sets_fills[data_set]['diff']
                        else:
                            fill = data_sets_fills[data_set]
                    else:
                        fill = 'full'

                    if plot_indiv:
                        energy_fig, energy_ax = energy_fig_axs[energy]
                        if data_sets_bands is not None and data_set in data_sets_bands:
                            energy_ax.fill_between(df_cent['divs'], df_cent['avg'] - df_cent['avg_err'],
                                                   df_cent['avg'] + df_cent['avg_err'], label=lab_energy,
                                                   color=color, alpha=alpha)
                        else:
                            energy_ax.errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['avg_err'], ls='none',
                                               marker=marker, label=lab_energy, color=color, alpha=alpha,
                                               fillstyle=fill)
                            if 'sys' in df_cent.columns:
                                energy_ax.errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['sys'], ls='',
                                                   elinewidth=4, marker='', color=color, alpha=errbar_alpha)
                    if plot_energies_fig:
                        if data_sets_bands is not None and data_set in data_sets_bands:
                            ax.fill_between(df_cent['divs'], df_cent['avg'] - df_cent['avg_err'],
                                            df_cent['avg'] + df_cent['avg_err'], label=lab, color=color, alpha=alpha)
                        else:
                            ax.errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['avg_err'], ls='none',
                                        marker=markers[energy_marker], label=lab, color=color, alpha=alpha,
                                        fillstyle=fill)
                            if 'sys' in df_cent.columns:
                                ax.errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['sys'], ls='', elinewidth=4,
                                            marker='', color=color, alpha=errbar_alpha)
                    if plot_energy_panels:
                        if data_sets_bands is not None and data_set in data_sets_bands:
                            ax_panels[energy].fill_between(df_cent['divs'], df_cent['avg'] - df_cent['avg_err'],
                                                           df_cent['avg'] + df_cent['avg_err'], label=lab_energy,
                                                           color=color, alpha=alpha)
                        else:
                            ax_panels[energy].errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['avg_err'],
                                                       ls='none', marker=marker, label=lab_energy, color=color,
                                                       alpha=alpha, fillstyle=fill)
                            if 'sys' in df_cent.columns:
                                ax_panels[energy].errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['sys'],
                                                           ls='', marker='', color=color, alpha=errbar_alpha,
                                                           elinewidth=4)
                    if print_data:
                        print(f'\n{data_set}, {energy}GeV, {cent} Centrality')
                        print(f'\'w\': [{", ".join(map(str, df_cent["divs"]))}]')
                        print(f'\'val\': [{", ".join(map(str, df_cent["avg"]))}]')
                        print(f'\'err\': [{", ".join(map(str, df_cent["avg_err"]))}]')
                        if 'sys' in df_cent.columns:
                            print(f'\'sys\': [{", ".join(map(str, df_cent["sys"]))}]')

                if fit and df_cent.size > 1:
                    try:
                        df_cent = df_cent[~df_cent.divs.isin(exclude_divs)]
                        popt, pcov = cf(quad_180, df_cent['divs'], df_cent['avg'], sigma=df_cent['avg_err'],
                                        absolute_sigma=True)
                        perr = np.sqrt(np.diag(pcov))
                        fit_pars.append({'data_set': data_set, 'energy': energy, 'cent': cent, 'color': color,
                                         'curvature': popt[0], 'curve_baseline': popt[1], 'curve_err': perr[0],
                                         'curve_base_err': perr[1],
                                         'spread': df_cent['spread'].iloc[0], 'amp': df_cent['amp'].iloc[0]})
                        if print_data:
                            print(f'\'baseline\' = {popt[1]}, \'baseline_err\' = {perr[1]}')
                            print(f'\'curvature\' = {popt[0]}, \'curvature_err\' = {perr[0]}')
                        if plot:
                            if data_sets_ls is not None and data_set in data_sets_ls:
                                ls = data_sets_ls[data_set]
                            else:
                                ls = '-'
                            x = np.linspace(0, 360, 100)
                            if plot_indiv:
                                energy_ax.plot(x, quad_180(x, *popt), ls=ls, color=color, alpha=0.65)
                            if plot_energies_fig:
                                ax.plot(x, quad_180(x, *popt), ls=ls, color=color, alpha=0.65)
                            if plot_energy_panels:
                                ax_panels[energy].plot(x, quad_180(x, *popt), ls=ls, color=color, alpha=0.65)
                        if popt[0] * popt[1] < 0:
                            popt2, pcov2 = cf(quad_180_zparam, df_cent['divs'], df_cent['avg'],
                                              sigma=df_cent['avg_err'], absolute_sigma=True)
                            perr2 = np.sqrt(np.diag(pcov2))
                            fit_pars[-1].update({'zero_mag': popt2[0], 'zero_mag_err': perr2[0], 'baseline': popt2[1],
                                                 'base_err': perr2[1]})
                            if verbose:
                                print(f'{data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n'
                                      f'z-c fit: {popt2}\nz-c covariance: {pcov2}')
                                print()
                        else:
                            if verbose:
                                print(f'No Zeros! {data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n')
                                print()
                    except RuntimeError as e:
                        print(f'Fitting Error, skipping data_set {data_set}, {e}')

    if plot:
        if title is None:
            title = ''
            if len(data_sets_plt) == 1:
                title += f'{data_set}'
            if len(energies) == 1:
                if title != '':
                    title += ' '
                title += f'{energies[0]}GeV'

        if ylab is None:
            ylab = r'$\langle\Delta\sigma^2\rangle_{single} - \langle\Delta\sigma^2\rangle_{mix}$'

        if plot_energies_fig:
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            if title != '' and title is not None:
                ax.set_title(title)
            ax.set_ylabel(ylab, fontsize='x-large')
            ax.set_xlabel(xlab, fontsize='x-large')
            if leg:
                if legend_order is not None:
                    handles, labels = ax.get_legend_handles_labels()
                    handles_dict = dict(zip(labels, handles))
                    ordered_handles = [handles_dict[label] for label in legend_order]
                    ax.legend(handles=ordered_handles, labels=legend_order)
                else:
                    ax.legend()

        if plot_indiv:
            for energy_i, (energy, (energy_fig, energy_ax)) in enumerate(energy_fig_axs.items()):
                if ylim is not None:
                    energy_ax.set_ylim(ylim)
                if xlim is not None:
                    energy_ax.set_xlim(xlim)
                if title is None:
                    energy_ax.set_title(f'{energy} GeV')
                else:
                    energy_ax.set_title(title)
                energy_ax.set_xlabel(xlab, fontsize='x-large')
                energy_ax.set_ylabel(ylab, fontsize='x-large')
                energy_ax.axhline(0, color='black', zorder=0)
                if leg:
                    if legend_order is not None:
                        handles, labels = energy_ax.get_legend_handles_labels()
                        handles_dict = dict(zip(labels, handles))
                        ordered_handles = [handles_dict[label] for label in legend_order]
                        energy_ax.legend(handles=ordered_handles, labels=legend_order)
                    else:
                        energy_ax.legend()
                energy_fig.tight_layout()
                if fig_splt_adj is not None:
                    energy_fig.subplots_adjust(**fig_splt_adj)
                if title is None:
                    energy_fig.canvas.manager.set_window_title(f'dsigma^2 vs Width {energy}GeV')
                else:
                    energy_fig.canvas.manager.set_window_title(f'dsigma^2 vs Width {title}')

        if plot_energy_panels:
            if title != '':
                fig_panels.suptitle(title)
            for energy_i, energy in enumerate(energies):
                if ylim is not None:
                    ax_panels[energy].set_ylim(ylim)
                if xlim is not None:
                    ax_panels[energy].set_xlim(xlim)
                ax_panels[energy].axhline(0, color='gray', alpha=0.8, zorder=0)
                # ax_panels[energy].text(0.5, 0.9, f'{energy_map[energy]} GeV', size='x-large', ha='center', va='top',
                #                        transform=ax_panels[energy].transAxes)
                ax_panels[energy].text(0.97, 0.05, f'{energy_map[energy]} GeV', size='x-large', ha='right', va='bottom',
                                       transform=ax_panels[energy].transAxes)
                if panel_letters:
                    ax_panels[energy].text(0.01, 0.5, f'({chr(97 + energy_i)})', size='x-large', ha='left', va='center',
                                           transform=ax_panels[energy].transAxes)
                if energy_i >= 3:
                    ax_panels[energy].set_xlabel(xlab, fontsize='x-large')
                if energy_i in [0, 3]:
                    ax_panels[energy].set_ylabel(ylab, fontsize='x-large')
                if energy_i not in [0, 3]:
                    ax_panels[energy].set_yticks([])
                if energy_i == leg_panel:
                    # ax_panels[energy].legend(loc='lower left', bbox_to_anchor=(0.5, 0.85), framealpha=1.0)
                    if leg:
                        if legend_order is not None:
                            handles, labels = ax_panels[energy].get_legend_handles_labels()
                            handles_dict = dict(zip(labels, handles))
                            ordered_handles = [handles_dict[label] for label in legend_order]
                            leg = ax_panels[energy].legend(handles=ordered_handles, labels=legend_order, loc='lower left',
                                                           frameon=leg_frameon)
                        else:
                            leg = ax_panels[energy].legend(loc='lower left', frameon=leg_frameon)
                        leg.zorder = 0

                if kin_loc is not None and energy_i == 1:
                    eta_line = r'|y| < 0.5'
                    pt_line = r'0.4 < $p_T$ < 2.0 GeV'
                    pt_line = f'{pt_line}\n0-5% Centrality'
                    samples_line = None  # '72 Samples/Event'
                    kin_text = f'Au+Au\n{eta_line}\n{pt_line}'
                    kin_text = f'{kin_text}\n{samples_line}' if samples_line is not None else kin_text
                    ax_panels[energy].text(*kin_loc, kin_text, ha='left', va='bottom', 
                                           transform=ax_panels[energy].transAxes)
                if star_prelim_loc is not None and energy_i == star_prelim_loc[0]:
                    ax_panels[energy].text(*star_prelim_loc[1:], 'STAR Preliminary', fontsize='large', ha='center',
                                           va='top', transform=ax_panels[energy].transAxes)
                if no_hydro_label and energy_i == 1:
                    ax_panels[energy].text(0.85, 0.2, f'No MUSIC data\nat 11.5 GeV', ha='center', va='bottom',
                                           fontsize='medium', transform=ax_panels[energy].transAxes)

        if plot_energies_fig:
            fig.tight_layout()
        if plot_energy_panels:
            fig_panels.tight_layout()
            if fig_splt_adj is not None:
                fig_panels.subplots_adjust(**fig_splt_adj)
            else:
                fig_panels.subplots_adjust(wspace=0.0, hspace=0.0, left=0.077, top=0.995, right=0.995, bottom=0.088)

    return pd.DataFrame(fit_pars)


def plot_dvar_avgs_vs_divs_cents(df, data_sets_plt, fit=False, data_sets_colors=None, data_sets_labels=None,
                                 exclude_divs=[], verbose=False, title=None, alpha=1, ylab=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    cents = pd.unique(df['cent'])
    fig_panels, ax_panels_all = plt.subplots(3, 3, sharex=True, sharey=True, dpi=144, figsize=(13.33, 6.16))
    fig_panels.canvas.manager.set_window_title(f'dsigma^2 vs Width Centrality Panels')
    ax_panels = dict(zip(cents, ax_panels_all.flat))
    if data_sets_colors is None:
        num_colorless = len(cents) * len(data_sets_plt) * len(pd.unique(df['energy']))
    else:
        num_colorless = len(cents) * (len(data_sets_plt) - len(data_sets_colors)) * len(pd.unique(df['energy']))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_colorless)))
    fit_pars = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='divs')
        for cent_marker, cent in enumerate(cents):
            df_cent = df_set[df_set['cent'] == cent]
            energies = pd.unique(df_cent['cent'])
            for energy in energies:
                df_energy = df_cent[df_cent['energy'] == energy]
                df_energy.sort_values(by='divs')
                if data_sets_colors is not None and data_set in data_sets_colors:
                    color = data_sets_colors[data_set]
                else:
                    color = next(colors)
                if data_sets_labels is not None and data_set in data_sets_labels:
                    lab_cent = data_sets_labels[data_set]
                else:
                    lab_cent = data_set
                if len(cents) > 1:
                    lab = f'{lab_cent}_{cent_map[cent]}'
                else:
                    lab = lab_cent
                if len(cents) > 1:
                    lab += f'_cent{cent}'
                ax_panels[energy].errorbar(df_cent['divs'], df_cent['avg'], yerr=df_cent['avg_err'],
                                           ls='none',
                                           marker='o', label=lab_cent, color=color, alpha=alpha)
                if fit and df_cent.size > 1:
                    try:
                        df_cent = df_cent[~df_cent.divs.isin(exclude_divs)]
                        popt, pcov = cf(quad_180, df_cent['divs'], df_cent['avg'], sigma=df_cent['avg_err'],
                                        absolute_sigma=True)
                        perr = np.sqrt(np.diag(pcov))
                        fit_pars.append({'data_set': data_set, 'energy': energy, 'cent': cent, 'color': color,
                                         'curvature': popt[0], 'curve_baseline': popt[1], 'curve_err': perr[0],
                                         'curve_base_err': perr[1],
                                         'spread': df_cent['spread'].iloc[0], 'amp': df_cent['amp'].iloc[0]})
                        x = np.linspace(0, 360, 100)
                        ax_panels[energy].plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                        if popt[0] * popt[1] < 0:
                            popt2, pcov2 = cf(quad_180_zparam, df_cent['divs'], df_cent['avg'],
                                              sigma=df_cent['avg_err'], absolute_sigma=True)
                            perr2 = np.sqrt(np.diag(pcov2))
                            fit_pars[-1].update({'zero_mag': popt2[0], 'zero_mag_err': perr2[0], 'baseline': popt2[1],
                                                 'base_err': perr2[1]})
                            if verbose:
                                print(f'{data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n'
                                      f'z-c fit: {popt2}\nz-c covariance: {pcov2}')
                                print()
                        else:
                            if verbose:
                                print(f'No Zeros! {data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n')
                                print()
                    except RuntimeError as e:
                        print(f'Fitting Error, skipping data_set {data_set}, {e}')

    if title is None:
        title = ''
        if len(data_sets_plt) == 1:
            title += f'{data_set}'
        if len(energies) == 1:
            if title != '':
                title += ' '
            title += f'{energies[0]}GeV'

    if ylab is None:
        ylab = r'$\langle\Delta\sigma^2\rangle_{single} - \langle\Delta\sigma^2\rangle_{mix}$'

    for cent_i, cent in enumerate(cents):
        ax_panels[cent].axhline(0, color='gray', alpha=0.8, zorder=0)
        ax_panels[cent].text(0.5, 0.9, f'{cent_map[cent]}', size='x-large', ha='center', va='top',
                             transform=ax_panels[cent].transAxes)
        if cent_i >= 6:
            ax_panels[cent].set_xlabel('Azimuthal Partition Width')
        if cent_i in [0, 3, 6]:
            ax_panels[cent].set_ylabel(ylab)
    ax_panels_all.flat[-1].set_xlabel('Azimuthal Partition Width')
    ax_panels[8].legend()

    fig_panels.tight_layout()
    fig_panels.subplots_adjust(wspace=0.0, hspace=0.0, left=0.075, right=0.995, top=0.985, bottom=0.075)

    return pd.DataFrame(fit_pars)


def plot_protons_fits_divs_flow(df, data_sets_plt, data_sets_colors=None):
    fig, ax = plt.subplots(dpi=144)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(data_sets_plt))))
    ax.axhline(0, color='black')
    x_divs = np.linspace(0, 360, 1000)
    fit_pars = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='divs')
        if data_sets_colors is None:
            color = next(colors)
        else:
            color = data_sets_colors[data_set]
        v2 = None
        for element in data_set.split('_'):
            if 'v2' in element:
                v2 = float('0.' + element.strip('v2'))
                break
        print(data_set, v2)
        if 'anticlflow_' in data_set:
            # v2 = float('0.' + data_set.split('_')[-3][2:])
            lab = f'anticl + v2={v2:.2f}'
        elif 'flow_' in data_set:
            # v2 = float('0.' + data_set.split('_')[-1][2:])
            lab = f'v2={v2:.2f}'
            ax.plot(x_divs, v2_divs(x_divs / 180 * np.pi, v2), color=color)
        else:
            lab = data_set

        ax.errorbar(df_set['divs'], df_set['slope'], yerr=df_set['slope_err'], ls='none',
                    marker='o', label=lab, color=color)

        if 'anticlflow_' in data_set:
            divs = np.array(df_set['divs'])
            slopes = np.array(df_set['slope'])
            y_sub = slopes - v2_divs(divs / 180 * np.pi, v2)
            ax.errorbar(divs, y_sub, yerr=df_set['slope_err'], ls='none',
                        marker='x', label=f'{lab} - v2(div)', color=color)

    ax.set_title('V2 vs Partition Width')
    ax.set_ylabel('Slope of Raw/Mix SD vs Total Protons per Event')
    ax.set_xlabel('Azimuthal Partition Width')
    ax.legend()

    fig.canvas.manager.set_window_title('V2 vs Partition Width')
    fig.tight_layout()

    return pd.DataFrame(fit_pars)


def plot_dsigma_fits_divs_flow(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None):
    fig, ax = plt.subplots(dpi=144)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(data_sets_plt))))
    ax.axhline(0, color='black')
    x_divs = np.linspace(0, 360, 1000)
    fit_pars = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='divs')
        if data_sets_colors is None:
            color = next(colors)
        else:
            color = data_sets_colors[data_set]
        v2 = 0
        for element in data_set.split('_'):
            if 'v2' in element:
                v2 = float('0.' + element.strip('v2'))
                break
        print(data_set, v2)
        if 'anticlflow_' in data_set:
            # v2 = float('0.' + data_set.split('_')[-3][2:])
            lab = f'anticl + v2={v2:.2f}'
        elif 'flow_' in data_set and abs(v2) > 0:
            # v2 = float('0.' + data_set.split('_')[-1][2:])
            lab = f'v2={v2:.2f}'
            ax.plot(x_divs, v2_divs(x_divs / 180 * np.pi, v2), color=color)
        else:
            lab = data_set
        if data_sets_labels is not None:
            lab = data_sets_labels[data_set]

        ax.errorbar(df_set['divs'], df_set['avg'], yerr=df_set['avg_err'], ls='none',
                    marker='o', label=lab, color=color)

        if 'anticlflow_' in data_set:
            divs = np.array(df_set['divs'])
            avgs = np.array(df_set['avg'])
            y_sub = avgs - v2_divs(divs / 180 * np.pi, v2)
            ax.errorbar(divs, y_sub, yerr=df_set['avg_err'], ls='none',
                        marker='x', label=f'{lab} - v2(div)', color=color)

    ax.set_title('V2 vs Partition Width')
    ax.set_ylabel(r'$\Delta\sigma^2$')
    ax.set_xlabel('Azimuthal Partition Width')
    ax.legend()

    fig.canvas.manager.set_window_title('V2 vs Partition Width')
    fig.tight_layout()

    return pd.DataFrame(fit_pars)


def plot_protons_fits_divs_cents(df, data_sets_plt, plot=False, fit=False, data_sets_colors=None, data_sets_labels=None,
                                 exclude_divs=[]):
    if plot:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        fig.suptitle(f'Slope vs Partition Width 72 Samples/Event')
        axs = axs.flatten()
        fig.canvas.manager.set_window_title(f'Slope vs Width All Energies')
        markers = ['o', 's', 'P', 'D', '*', '^', 'p']
    energies = pd.unique(df['energy'])
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(energies) * len(data_sets_plt))))
    fit_pars = []
    for ax_index, cent in enumerate(pd.unique(df['cent'])):
        if plot:
            ax = axs[ax_index]
            ax.axhline(0, ls='-', color='black')
            ax.text(125, 0.002, f'Centrality {cent}', size='x-large')
        df_cent = df[df['cent'] == cent]
        for data_set in data_sets_plt:
            df_set = df_cent[df_cent['name'] == data_set]
            for energy_marker, energy in enumerate(energies):
                df_energy = df_set[df_set['energy'] == energy]
                # print(f'{data_set} {energy}GeV Cent {cent}:\n {df_energy}\n')
                df_energy.sort_values(by='divs')
                if data_sets_colors is None:
                    color = next(colors)
                else:
                    color = data_sets_colors[data_set]
                if plot:
                    if data_sets_labels is None:
                        lab_energy = data_set
                    else:
                        lab_energy = data_sets_labels[data_set]
                    if len(energies) > 1:
                        lab = f'{lab_energy}_{energy}GeV'
                    else:
                        lab = lab_energy
                    ax.errorbar(df_energy['divs'], df_energy['slope'], yerr=df_energy['slope_err'], ls='none',
                                marker=markers[energy_marker], label=lab, color=color)
                if fit and df_energy.size > 1:
                    df_energy = df_energy[~df_energy.divs.isin(exclude_divs)]
                    popt, pcov = cf(quad_180, df_energy['divs'], df_energy['slope'], sigma=df_energy['slope_err'],
                                    absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))
                    fit_pars.append({'data_set': data_set, 'energy': energy, 'curvature': popt[0], 'color': color,
                                     'curve_base': popt[1], 'curve_err': perr[0], 'curve_base_err': perr[1],
                                     'cent': cent, 'spread': df_energy['spread'].iloc[0],
                                     'amp': df_energy['amp'].iloc[0]})
                    if plot:
                        x = np.linspace(0, 360, 100)
                        ax.plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                    if popt[0] * popt[1] < 0:
                        popt2, pcov2 = cf(quad_180_zparam, df_energy['divs'], df_energy['slope'],
                                          sigma=df_energy['slope_err'],
                                          absolute_sigma=True)
                        perr2 = np.sqrt(np.diag(pcov2))
                        fit_pars[-1].update({'zero_mag': popt2[0], 'zero_mag_err': perr2[0], 'baseline': popt2[1],
                                             'base_err': perr2[1]})
                        print(f'{data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\nz-c fit: {popt2}\n'
                              f'z-c covariance: {pcov2}')
                        print()
                    else:
                        print(f'No Zeros! {data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n')
                        print()

    if plot:
        for i in [0, 3, 6]:
            axs[i].set_ylabel('Slope')
        for i in [6, 7, 8]:
            axs[i].set_xlabel('Azimuthal Partition Width')
        ax.legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title('Slopes vs Partition Width Centralities')

    return pd.DataFrame(fit_pars)


def plot_dvar_avgs_divs_cents(df, data_sets_plt, plot=False, fit=False, data_sets_colors=None, data_sets_labels=None,
                              exclude_divs=[], verbose=False):
    energies = pd.unique(df['energy'])
    if plot:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(13.33, 6.16), dpi=144)
        suptitle = r'$\langle\Delta\sigma^2\rangle$ vs Partition Width 72 Samples/Event'
        suptitle = f'{energies[0]}GeV {suptitle}' if len(energies) == 1 else suptitle
        fig.suptitle(suptitle)
        axs = axs.flatten()
        fig.canvas.manager.set_window_title(f'DSigma vs Width All Energies')
        markers = ['o', 's', 'P', 'D', '*', '^', 'p']
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(energies) * len(data_sets_plt))))
    fit_pars = []
    for ax_index, cent in enumerate(pd.unique(df['cent'])):
        if plot:
            ax = axs[ax_index]
            ax.axhline(0, ls='-', color='black')
            ax.text(0.02, 0.08, f'Centrality {cent}', size='x-large', ha='left', va='bottom', transform=ax.transAxes)
        df_cent = df[df['cent'] == cent]
        for data_set in data_sets_plt:
            df_set = df_cent[df_cent['name'] == data_set]
            for energy_marker, energy in enumerate(energies):
                df_energy = df_set[df_set['energy'] == energy]
                # print(f'{data_set} {energy}GeV Cent {cent}:\n {df_energy}\n')
                df_energy.sort_values(by='divs')
                if data_sets_colors is None:
                    color = next(colors)
                else:
                    color = data_sets_colors[data_set]
                if plot:
                    if data_sets_labels is None:
                        lab_energy = data_set
                    else:
                        lab_energy = data_sets_labels[data_set]
                    if len(energies) > 1:
                        lab = f'{lab_energy}_{energy}GeV'
                    else:
                        lab = lab_energy
                    ax.errorbar(df_energy['divs'], df_energy['avg'], yerr=df_energy['avg_err'], ls='none',
                                marker=markers[energy_marker], label=lab, color=color)
                if fit and df_energy.size > 1:
                    df_energy = df_energy[~df_energy.divs.isin(exclude_divs)]
                    popt, pcov = cf(quad_180, df_energy['divs'], df_energy['avg'], sigma=df_energy['avg_err'],
                                    absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))
                    fit_pars.append({'data_set': data_set, 'energy': energy, 'curvature': popt[0], 'color': color,
                                     'curve_base': popt[1], 'curve_err': perr[0], 'curve_base_err': perr[1],
                                     'cent': cent, 'spread': df_energy['spread'].iloc[0],
                                     'amp': df_energy['amp'].iloc[0]})
                    if plot:
                        x = np.linspace(0, 360, 100)
                        ax.plot(x, quad_180(x, *popt), ls='-', color=color, alpha=0.65)
                    if popt[0] * popt[1] < 0:
                        popt2, pcov2 = cf(quad_180_zparam, df_energy['divs'], df_energy['avg'],
                                          sigma=df_energy['avg_err'],
                                          absolute_sigma=True)
                        perr2 = np.sqrt(np.diag(pcov2))
                        fit_pars[-1].update({'zero_mag': popt2[0], 'zero_mag_err': perr2[0], 'baseline': popt2[1],
                                             'base_err': perr2[1]})
                        if verbose:
                            print(
                                f'{data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\nz-c fit: {popt2}\n'
                                f'z-c covariance: {pcov2}')
                            print()
                    elif verbose:
                        print(f'No Zeros! {data_set}, {energy}GeV\na-c fit: {popt}\na-c covariance: {pcov}\n')
                        print()

    if plot:
        for i in [0, 3, 6]:
            axs[i].set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
        for i in [6, 7, 8]:
            axs[i].set_xlabel('Azimuthal Partition Width')
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.canvas.manager.set_window_title('Dsigmas vs Partition Width Centralities')

    return pd.DataFrame(fit_pars)


def plot_vs_div_width_comp(df, title=''):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f'Closest Sims vs Partition Width {title}')
    ax.set_title(title)
    ax.set_xlabel('Azimuthal Partition Width')
    ax.set_ylabel('Raw SD / Mix SD vs Total Protons Slope')
    ax.axhline(0, color='black')

    fig2, ax2 = plt.subplots()
    fig2.canvas.manager.set_window_title(f'Closest Sims Zero Base {title}')
    ax2.set_title(title)
    ax2.set_xlabel('Baseline')
    ax2.set_ylabel('Zeros')
    ax2.set_ylim(110, 230)
    ax2.set_xlim(-0.013, 0.0005)
    ax2.grid()

    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(pd.unique(df['name'])))))
    for data_set in pd.unique(df['name']):
        color = next(colors)
        df_set = df[df['name'] == data_set]

        ax.errorbar(df_set['divs'], df_set['slope'], df_set['slope_err'], ls='none', marker='o', alpha=0.6, color=color,
                    label=data_set)
        popt, pcov = cf(quad_180_zparam, df_set['divs'], df_set['slope'], sigma=df_set['slope_err'],
                        absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        x_plt = np.linspace(0, 360, 1000)
        ax.plot(x_plt, quad_180_zparam(x_plt, *popt), color=color)
        ax2.errorbar(popt[1], popt[0], perr[0], perr[1], color=color, marker='o', ls='none', label=data_set)

    ax.legend()
    ax2.legend()
    fig.tight_layout()
    fig2.tight_layout()


def plot_slope_div_fits(df_fits, data_sets_colors=None, data_sets_labels=None, ref_line=0, data_sets=None):
    fig_base_energy, ax_base_energy = plt.subplots()
    ax_base_energy.set_xlabel('Energy (GeV)')
    ax_base_energy.set_ylabel('Baseline')
    ax_base_energy.axhline(ref_line, color='black')
    fig_base_energy.canvas.manager.set_window_title('Slope Baseline vs Energy')

    fig_zeros_energy, ax_zeros_energy = plt.subplots()
    ax_zeros_energy.set_xlabel('Energy (GeV)')
    ax_zeros_energy.set_ylabel('Zeros')
    ax_zeros_energy.axhline(ref_line, color='black')
    fig_zeros_energy.canvas.manager.set_window_title('Slope Zeros vs Energy')

    fig_basez_zeros, ax_basez_zeros = plt.subplots()
    ax_basez_zeros.set_xlabel('Zeros')
    ax_basez_zeros.set_ylabel('Baseline')
    ax_basez_zeros.axvline(0, color='black')
    ax_basez_zeros.axhline(0, color='black')
    fig_basez_zeros.canvas.manager.set_window_title('Slope Baseline vs Zeros')

    colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'P', 'p', '2']

    if data_sets is None:
        data_sets = pd.unique(df_fits['data_set'])
    for data_set_index, data_set in enumerate(data_sets):
        if data_sets_labels is None:
            lab = data_set
        else:
            lab = data_sets_labels[data_set]
        if data_sets_colors is None:
            color = colors[data_set_index % len(colors)]
        else:
            color = data_sets_colors[data_set]

        df_data_set = df_fits[df_fits['data_set'] == data_set]

        ax_base_energy.errorbar(df_data_set['energy'], df_data_set['baseline'], yerr=df_data_set['base_err'], ls='none',
                                marker='o', label=lab, color=color, alpha=0.8)

        ax_zeros_energy.errorbar(df_data_set['energy'], df_data_set['zero_mag'], yerr=df_data_set['zero_mag_err'],
                                 ls='none', marker='o', label=lab, color=color, alpha=0.8)

        energies = pd.unique(df_data_set['energy'])
        for energy_index, energy in enumerate(energies):
            df_energy = df_data_set[df_data_set['energy'] == energy]
            ax_basez_zeros.errorbar(df_energy['zero_mag'], df_energy['baseline'], xerr=df_energy['zero_mag_err'],
                                    yerr=df_energy['base_err'], ls='none', color=colors[energy_index % len(colors)],
                                    marker=markers[data_set_index % len(markers)], label=f'{lab} {energy}GeV',
                                    alpha=0.8)

    ax_base_energy.legend()
    ax_zeros_energy.legend()
    ax_basez_zeros.legend(loc='center left', bbox_to_anchor=(0.05, 0.5))

    ax_base_energy.grid()
    ax_zeros_energy.grid()
    ax_basez_zeros.grid()

    fig_base_energy.tight_layout()
    fig_zeros_energy.tight_layout()
    fig_basez_zeros.tight_layout()


def plot_slope_div_fits_simpars(df_fits):
    fig_base_amp, ax_base_amp = plt.subplots()
    ax_base_amp.set_xlabel(r'Simulation Amplitude ($A$)')
    ax_base_amp.set_ylabel(r'$b$')
    ax_base_amp.axhline(0, color='black')
    ax_base_amp.axvline(0, color='black')
    fig_base_amp.canvas.manager.set_window_title('Quad Fit Baseline vs Amplitude')

    fig_amp_base, ax_amp_base = plt.subplots()
    ax_amp_base.set_xlabel(r'$b$')
    ax_amp_base.set_ylabel(r'Simulation Amplitude ($A$)')
    ax_amp_base.axvline(0, color='black')
    ax_amp_base.axhline(0, color='black')
    fig_amp_base.canvas.manager.set_window_title('Quad Fit Amplitude vs Baseline')

    fig_base_spread, ax_base_spread = plt.subplots()
    ax_base_spread.set_xlabel(r'Simulation Range ($\sigma$)')
    ax_base_spread.set_ylabel(r'$b$')
    ax_base_spread.axhline(0, color='black')
    fig_base_spread.canvas.manager.set_window_title('Quad Fit Baseline vs Spread')

    fig_zeros_amp, ax_zeros_amp = plt.subplots()
    ax_zeros_amp.set_xlabel(r'Simulation Amplitude ($A$)')
    ax_zeros_amp.set_ylabel(r'$z$')
    ax_zeros_amp.axhline(0, color='black')
    fig_zeros_amp.canvas.manager.set_window_title('Quad Fit Zeros vs Amplitude')

    fig_zeros_spread, ax_zeros_spread = plt.subplots()
    ax_zeros_spread.set_xlabel(r'Simulation Range ($\sigma$)')
    ax_zeros_spread.set_ylabel(r'$z$')
    # ax_zeros_spread.grid()
    # ax_zeros_spread.axhline(0, color='black')
    fig_zeros_spread.canvas.manager.set_window_title('Quad Fit Zeros vs Spread')

    fig_base_zeros_gamp, ax_base_zeros_gamp = plt.subplots()
    ax_base_zeros_gamp.set_ylabel(r'$z$')
    ax_base_zeros_gamp.set_xlabel(r'$b$')
    ax_base_zeros_gamp.axvline(0, color='black')
    # ax_base_zeros_gamp.axhline(0, color='black')
    fig_base_zeros_gamp.canvas.manager.set_window_title('Quad Fit Baseline vs Zeros Amp Sets')

    fig_base_zeros_gspread, ax_base_zeros_gspread = plt.subplots()
    ax_base_zeros_gspread.set_ylabel(r'$z$')
    ax_base_zeros_gspread.set_xlabel(r'$b$')
    ax_base_zeros_gspread.axvline(0, color='black')
    # ax_base_zeros_gspread.axhline(0, color='black')
    fig_base_zeros_gspread.canvas.manager.set_window_title('Quad Fit Baseline vs Zeros Spread Sets')

    amps = pd.unique(df_fits['amp'])
    spreads = pd.unique(df_fits['spread'])
    print(f'Amps: {amps}\nSpreads: {spreads}')

    for amp in amps:
        df_amp = df_fits[df_fits['amp'] == amp]
        ax_base_spread.errorbar(df_amp['spread'], df_amp['baseline'], yerr=df_amp['base_err'], ls='none', marker='o',
                                label=f'A={amp}')
        ax_zeros_spread.errorbar(df_amp['spread'], df_amp['zero_mag'], yerr=df_amp['zero_mag_err'], ls='none',
                                 marker='o', label=f'A={amp}')
        ax_base_zeros_gamp.errorbar(df_amp['baseline'], df_amp['zero_mag'], xerr=df_amp['base_err'],
                                    yerr=df_amp['zero_mag_err'], ls='none', marker='o', label=f'A={amp}')

    sigma_fits = []
    cl_type_data = {'_clmul_': {'name': 'Attractive', 'line': '--'}, '_aclmul_': {'name': 'Repulsive', 'line': ':'}}
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(spreads))))
    for spread in spreads:
        color = next(colors)
        df_spread = df_fits[df_fits['spread'] == spread]
        for cl_type, cl_data in cl_type_data.items():
            df_set = df_spread[df_spread['data_set'].str.contains(cl_type)]
            if df_set.size == 0 or any(np.isnan(df_set['baseline'])) or any(np.isinf(df_set['baseline'])) or \
                    any(np.isnan(df_set['base_err'])) or any(np.isinf(df_set['base_err'])):
                continue
            if cl_type == '_aclmul_':
                lab = f'σ={round(spread, 2)}'
            else:
                lab = None
            ax_base_amp.errorbar(df_set['amp'], df_set['baseline'], yerr=df_set['base_err'], ls='none',
                                 marker='o', label=lab, color=color)
            popt, pcov = cf(line_yint0, df_set['amp'], df_set['baseline'], sigma=df_set['base_err'],
                            absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            x_vals = np.array([0, max(df_set['amp'])])
            ax_base_amp.plot(x_vals, line_yint0(x_vals, *popt), ls=cl_data['line'], color=color)
            ax_base_amp.fill_between(x_vals, line_yint0(x_vals, popt[0] - perr[0]),
                                     line_yint0(x_vals, popt[0] + perr[0]), color=color, alpha=0.3)

            ax_amp_base.errorbar(df_set['baseline'], df_set['amp'], xerr=df_set['base_err'], ls='none',
                                 marker='o', label=lab, color=color)
            inv_slope = 1 / Measure(popt[0], perr[0])
            x_vals = np.array([0, max(df_set['baseline']) * 1.1]) if cl_type == '_clmul_' \
                else np.array([min(df_set['baseline']) * 1.1, 0])
            ax_amp_base.plot(x_vals, line_yint0(x_vals, inv_slope.val), ls=cl_data['line'], color=color)
            ax_amp_base.fill_between(x_vals, line_yint0(x_vals, inv_slope.val - inv_slope.err),
                                     line_yint0(x_vals, inv_slope.val + inv_slope.err), color=color, alpha=0.3)

            avg_zero = np.mean(
                [Measure(val, err) for val, err in zip(df_set['zero_mag'], df_set['zero_mag_err'])])
            ax_zeros_amp.errorbar(df_set['amp'], df_set['zero_mag'], yerr=df_set['zero_mag_err'], ls='none',
                                  marker='o', label=lab, color=color)
            ax_zeros_amp.axhspan(avg_zero.val - avg_zero.err, avg_zero.val + avg_zero.err, alpha=0.3, color=color)
            ax_zeros_amp.axhline(avg_zero.val, ls=cl_data['line'], color=color)
            ax_base_zeros_gspread.errorbar(df_set['baseline'], df_set['zero_mag'], xerr=df_set['base_err'],
                                           yerr=df_set['zero_mag_err'], ls='none', marker='o', label=lab,
                                           color=color)
            sigma_fits.append({'clust_type': cl_data['name'], 'spread': spread, 'zero_mean_val': avg_zero.val,
                               'zero_mean_err': avg_zero.err, 'base_slope_val': popt[0], 'base_slope_err': perr[0],
                               'amp_slope_val': inv_slope.val, 'amp_slope_err': inv_slope.err})

    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(spreads))))  # Go through again to plot horizontal lines
    for spread in spreads:
        color = next(colors)
        df_spread = df_fits[df_fits['spread'] == spread]
        for cl_type, cl_data in cl_type_data.items():
            df_set = df_spread[df_spread['data_set'].str.contains(cl_type)]
            if df_set.size == 0:
                continue
            avg_zero = sum([Measure(val, err) for val, err in zip(df_set['zero_mag'], df_set['zero_mag_err'])])
            avg_zero /= df_set['zero_mag'].size
            ax_base_zeros_lims = ax_base_zeros_gspread.get_xlim()
            ax_base_zeros_mid = (0 - ax_base_zeros_lims[0]) / np.diff(ax_base_zeros_lims)[0]
            if cl_type == '_aclmul_':
                ax_base_zeros_gspread.axhspan(avg_zero.val - avg_zero.err, avg_zero.val + avg_zero.err, alpha=0.3,
                                              xmax=ax_base_zeros_mid, color=color)
                ax_base_zeros_gspread.axhline(avg_zero.val, ls=cl_data['line'], color=color, xmax=ax_base_zeros_mid)
            else:
                ax_base_zeros_gspread.axhspan(avg_zero.val - avg_zero.err, avg_zero.val + avg_zero.err, alpha=0.3,
                                              xmin=ax_base_zeros_mid, color=color)
                ax_base_zeros_gspread.axhline(avg_zero.val, ls=cl_data['line'], color=color, xmin=ax_base_zeros_mid)

    ax_base_amp.legend()
    ax_amp_base.legend()
    ax_base_spread.legend()
    ax_zeros_amp.legend()
    ax_zeros_spread.legend()
    ax_base_zeros_gamp.legend()
    ax_base_zeros_gspread.legend()

    fig_base_amp.tight_layout()
    fig_amp_base.tight_layout()
    fig_base_spread.tight_layout()
    fig_zeros_amp.tight_layout()
    fig_zeros_spread.tight_layout()
    fig_base_zeros_gamp.tight_layout()
    fig_base_zeros_gspread.tight_layout()

    return pd.DataFrame(sigma_fits)


def plot_z_vs_spread(df_fits, amps=None, spreads=None, amps_colors=None, amps_markers=None, amps_x_shifts=None,
                     alpha=1):
    fig_zeros_spread, ax_zeros_spread = plt.subplots(figsize=(8, 4), dpi=144)
    ax_zeros_spread.set_xlabel(r'Simulation Range ($\Sigma$)', fontsize=14)
    ax_zeros_spread.set_ylabel(r'$z$', fontsize=14)
    fig_zeros_spread.canvas.manager.set_window_title('Quad Fit Zeros vs Spread')

    if amps is None:
        amps = pd.unique(df_fits['amp'])
    if spreads is None:
        spreads = pd.unique(df_fits['spread'])

    print(f'Amps: {amps}\nSpreads: {spreads}')

    fit_zs, fit_z_errs, fit_spreads = [], [], []
    colors_amps = iter(plt.cm.rainbow(np.linspace(0, 1, len(amps))))
    for amp in amps:
        df_amp = df_fits[df_fits['amp'] == amp]
        df_cl_amp = df_amp[df_amp['data_set'].str.contains('_clmul_')]
        df_acl_amp = df_amp[df_amp['data_set'].str.contains('_aclmul_')]
        fit_zs.extend(df_cl_amp['zero_mag'])
        fit_z_errs.extend(df_cl_amp['zero_mag_err'])
        fit_spreads.extend(df_cl_amp['spread'])
        if amps_colors is None or amp not in amps_colors:
            color = next(colors_amps)
        else:
            color = amps_colors[amp]
        if amps_markers is None or amp not in amps_markers:
            marker = 'o'
        else:
            marker = amps_markers[amp]
        if amps_x_shifts is None or amp not in amps_x_shifts:
            x_shift = 0
        else:
            x_shift = amps_x_shifts[amp]
        ax_zeros_spread.errorbar(df_cl_amp['spread'] + x_shift, df_cl_amp['zero_mag'],
                                 yerr=df_cl_amp['zero_mag_err'], ls='none', alpha=alpha,
                                 marker=marker, label=rf'$A=+{amp}$', color=color, fillstyle='none')
        ax_zeros_spread.errorbar(df_acl_amp['spread'] + x_shift, df_acl_amp['zero_mag'],
                                 yerr=df_acl_amp['zero_mag_err'], ls='none', alpha=alpha,
                                 marker=marker, label=rf'$A=-{amp}$', color=color, fillstyle='full')

    popt, pcov = cf(inv_x_const, fit_spreads, fit_zs, sigma=fit_z_errs, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    spread_range = max(fit_spreads) - min(fit_spreads)
    x_vals = np.linspace(min(fit_spreads) - spread_range * 0.05, max(fit_spreads) + spread_range * 0.05, 1000)
    ax_zeros_spread.plot(x_vals, inv_x_const(x_vals, *popt), ls='--', color='salmon',
                         label=rf'${round(popt[0])}/x+{round(popt[1])}$')
    print(popt)

    ax_zeros_spread.legend()
    fig_zeros_spread.tight_layout()
    fig_zeros_spread.subplots_adjust(left=0.075, top=0.99, bottom=0.12, right=0.995)


def plot_b_vs_amp(df_fits, spreads=None, alpha=1, ylim=None):
    fig_base_amp, ax_base_amp = plt.subplots(figsize=(8, 4), dpi=144)
    ax_base_amp.set_xlabel(r'Simulation Amplitude Magnitude ($\left| A \right|$)', fontsize=14)
    ax_base_amp.set_ylabel(r'$b$', fontsize=14)
    ax_base_amp.axhline(0, color='black')
    ax_base_amp.axvline(0, color='black')
    fig_base_amp.canvas.manager.set_window_title('Quad Fit Baseline vs Amplitude')

    fig_base_div_spread_amp, ax_base_div_spread_amp = plt.subplots(figsize=(8, 4), dpi=144)
    ax_base_div_spread_amp.set_xlabel(r'Simulation Amplitude Magnitude ($\left| A \right|$)', fontsize=14)
    ax_base_div_spread_amp.set_ylabel(r'$b/\Sigma$', fontsize=14)
    ax_base_div_spread_amp.axhline(0, color='black')
    ax_base_div_spread_amp.axvline(0, color='black')
    fig_base_div_spread_amp.canvas.manager.set_window_title('Quad Fit Baseline/Spread vs Amplitude')

    if spreads is None:
        spreads = pd.unique(df_fits['spread'])
    cl_type_data = {'_clmul_': {'name': 'Attractive', 'line': '--', 'fill': 'none'},
                    '_aclmul_': {'name': 'Repulsive', 'line': ':', 'fill': 'full'}}
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(spreads))))
    for spread in spreads:
        color = next(colors)
        df_spread = df_fits[df_fits['spread'] == spread]
        for cl_type, cl_data in cl_type_data.items():
            df_set = df_spread[df_spread['data_set'].str.contains(cl_type)]
            if df_set.size == 0 or any(np.isnan(df_set['baseline'])) or any(np.isinf(df_set['baseline'])) or \
                    any(np.isnan(df_set['base_err'])) or any(np.isinf(df_set['base_err'])):
                continue
            if cl_type == '_aclmul_':
                lab = f'Σ={round(spread, 2)}'
            else:
                lab = None
            ax_base_amp.errorbar(df_set['amp'], df_set['baseline'], yerr=df_set['base_err'], ls='none',
                                 marker='o', label=lab, color=color, fillstyle=cl_data['fill'], alpha=alpha)
            ax_base_div_spread_amp.errorbar(df_set['amp'], df_set['baseline'] / df_set['spread'], ls='none',
                                            yerr=df_set['base_err'], marker='o', label=lab, color=color,
                                            fillstyle=cl_data['fill'], alpha=alpha)
            # popt, pcov = cf(line_yint0, df_set['amp'], df_set['baseline'], sigma=df_set['base_err'],
            #                 absolute_sigma=True)
            # perr = np.sqrt(np.diag(pcov))
            x_vals = np.array([0, max(df_set['amp'])])
            popt, pcov = cf(line, df_set['amp'], df_set['baseline'], sigma=df_set['base_err'],
                            absolute_sigma=True)
            ax_base_amp.plot(x_vals, line(x_vals, *popt), ls=cl_data['line'], color=color)
            popt, pcov = cf(line, df_set['amp'], df_set['baseline'] / df_set['spread'], sigma=df_set['base_err'],
                            absolute_sigma=True)
            ax_base_div_spread_amp.plot(x_vals, line(x_vals, *popt), ls=cl_data['line'], color=color)
            # ax_base_amp.plot(x_vals, line_yint0(x_vals, *popt), ls=cl_data['line'], color=color)
            # ax_base_amp.fill_between(x_vals, line_yint0(x_vals, popt[0] - perr[0]),
            #                          line_yint0(x_vals, popt[0] + perr[0]), color=color, alpha=0.3)
    if ylim is not None:
        ax_base_amp.set_ylim(ylim)
    ax_base_amp.legend()
    ax_base_div_spread_amp.legend()
    fig_base_amp.tight_layout()
    fig_base_div_spread_amp.tight_layout()
    fig_base_amp.subplots_adjust(left=0.121, top=0.99, bottom=0.125, right=0.995)
    fig_base_div_spread_amp.subplots_adjust(left=0.115, top=0.99, bottom=0.11, right=0.995)


def plot_b_vs_amp_sig_dep(df_fits, spreads=None, alpha=1, ylim=None):
    fig_base_spread, ax_base_spread = plt.subplots(figsize=(8, 4), dpi=144)
    ax_base_spread.set_xlabel(r'Simulation Spread ($\sigma$)')
    ax_base_spread.set_ylabel(r'$b$')
    ax_base_spread.axhline(0, color='black')
    fig_base_spread.canvas.manager.set_window_title('Quad Fit Baseline vs Spread Amplitude Sets')

    fig_base_div_spread, ax_base_div_spread = plt.subplots(figsize=(8, 4), dpi=144)
    ax_base_div_spread.set_xlabel(r'Simulation Spread ($\sigma$)')
    ax_base_div_spread.set_ylabel(r'$b/\sigma$')
    ax_base_div_spread.axhline(0, color='black')
    fig_base_div_spread.canvas.manager.set_window_title('Quad Fit Baseline/Spread vs Spread Amplitude Sets')

    amps = pd.unique(df_fits['amp'])
    cl_type_data = {'_clmul_': {'name': 'Attractive', 'line': '--', 'fill': 'none'},
                    '_aclmul_': {'name': 'Repulsive', 'line': ':', 'fill': 'full'}}
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(amps))))

    def b_spread_fit(x, a, b, c):
        return a + b * x + c * x**2

    fits = {'_clmul_': {'a_1': [], 'a_2': [], 'a_3': []}, '_aclmul_': {'a_1': [], 'a_2': [], 'a_3': []}}
    for amp in amps:
        color = next(colors)
        df_amp = df_fits[df_fits['amp'] == amp]
        for cl_type, cl_data in cl_type_data.items():
            df_set = df_amp[df_amp['data_set'].str.contains(cl_type)]
            if df_set.size == 0 or any(np.isnan(df_set['baseline'])) or any(np.isinf(df_set['baseline'])) or \
                    any(np.isnan(df_set['base_err'])) or any(np.isinf(df_set['base_err'])):
                continue
            if cl_type == '_aclmul_':
                lab = rf'$A={round(amp, 3)}$'
            else:
                lab = None
            ax_base_spread.errorbar(df_set['spread'], df_set['baseline'], yerr=df_set['base_err'], ls='none',
                                 marker='o', label=lab, color=color, fillstyle=cl_data['fill'], alpha=alpha)
            ax_base_div_spread.errorbar(df_set['spread'], df_set['baseline'] / df_set['spread'],
                                        yerr=df_set['base_err'], ls='none', marker='o', label=lab, color=color,
                                        fillstyle=cl_data['fill'], alpha=alpha)
            popt, pcov = cf(b_spread_fit, df_set['spread'], df_set['baseline'], sigma=df_set['base_err'],
                            absolute_sigma=True)
            x_fit_plot = np.linspace(0, 1.3, 1000)
            ax_base_spread.plot(x_fit_plot, b_spread_fit(x_fit_plot, *popt), color=color, alpha=0.4, ls='--')
            fits[cl_type]['a_1'].append(popt[0])
            fits[cl_type]['a_2'].append(popt[1])
            fits[cl_type]['a_3'].append(popt[2])
    ax_base_spread.text(0, -0.0004, r'$b = a_1 + a_2 * \sigma + a_3 * \sigma^2$')
    if ylim is not None:
        ax_base_spread.set_ylim(ylim)
    ax_base_spread.legend()
    ax_base_div_spread.legend()
    fig_base_spread.tight_layout()
    fig_base_div_spread.tight_layout()
    fig_base_spread.subplots_adjust(left=0.115, top=0.99, bottom=0.11, right=0.995)
    fig_base_div_spread.subplots_adjust(left=0.115, top=0.99, bottom=0.11, right=0.995)

    fig_fits, axs_fits = plt.subplots(nrows=3, figsize=(8, 6), dpi=144, sharex='all')
    for par, ax in zip(['a_1', 'a_2', 'a_3'], axs_fits):
        ax.set_ylabel(rf'${par}$')
        ax.axhline(0, color='gray')
        for cl_type, cl_data in cl_type_data.items():
            ax.plot(amps, fits[cl_type][par], marker='o', alpha=0.8, fillstyle=cl_data['fill'], label=cl_data['name'])
    axs_fits[0].text(0.005, -4.7e-5, r'$b = a_1 + a_2 * \sigma + a_3 * \sigma^2$')
    axs_fits[0].legend()
    axs_fits[-1].set_xlabel(r'$\left| A \right|$')
    fig_fits.tight_layout()


def plot_sigma_fits_interp(sigma_fits):
    fig_spread_zero, ax_spread_zero = plt.subplots()
    ax_spread_zero.set_xlabel('Mean Zero')
    ax_spread_zero.set_ylabel('Spread')
    ax_spread_zero.axhline(0, color='black')
    fig_spread_zero.canvas.manager.set_window_title('Spread vs Mean Zero')

    fig_spread_base_slope, ax_spread_base_slope = plt.subplots()
    ax_spread_base_slope.set_xlabel('Spread')
    ax_spread_base_slope.set_ylabel('Base Slope')
    ax_spread_base_slope.axhline(0, color='black')
    fig_spread_base_slope.canvas.manager.set_window_title('Base Slope vs Spread')

    fig_spread_amp_slope, ax_spread_amp_slope = plt.subplots()
    ax_spread_amp_slope.set_xlabel('Spread')
    ax_spread_amp_slope.set_ylabel('Amp Slope')
    ax_spread_amp_slope.axhline(0, color='black')
    fig_spread_amp_slope.canvas.manager.set_window_title('Amp Slope vs Spread')

    cl_types = pd.unique(sigma_fits['clust_type'])
    interpolations = {cl_type: {} for cl_type in cl_types}
    for cl_type in cl_types:
        print(cl_type)
        df_cl_type = sigma_fits[sigma_fits['clust_type'] == cl_type]
        f_spread_zero = interp1d(df_cl_type['zero_mean_val'], df_cl_type['spread'], kind='cubic')
        x_spread_zero = np.linspace(min(df_cl_type['zero_mean_val']), max(df_cl_type['zero_mean_val']), 1000)
        ax_spread_zero.errorbar(df_cl_type['zero_mean_val'], df_cl_type['spread'], xerr=df_cl_type['zero_mean_err'],
                                ls='none', label=cl_type, marker='o')
        ax_spread_zero.plot(x_spread_zero, f_spread_zero(x_spread_zero))

        f_spread_base = interp1d(df_cl_type['spread'], df_cl_type['base_slope_val'], kind='cubic')
        x_spread_base = np.linspace(min(df_cl_type['spread']), max(df_cl_type['spread']), 1000)
        ax_spread_base_slope.errorbar(df_cl_type['spread'], df_cl_type['base_slope_val'], ls='none', label=cl_type,
                                      marker='o', yerr=df_cl_type['base_slope_err'])
        ax_spread_base_slope.plot(x_spread_base, f_spread_base(x_spread_base))

        f_spread_amp = interp1d(df_cl_type['spread'], df_cl_type['amp_slope_val'], kind='cubic')
        ax_spread_amp_slope.errorbar(df_cl_type['spread'], df_cl_type['amp_slope_val'], ls='none', label=cl_type,
                                     marker='o', yerr=df_cl_type['amp_slope_err'])
        ax_spread_amp_slope.plot(x_spread_base, f_spread_amp(x_spread_base))

        interpolations[cl_type] = {'spread_zero': f_spread_zero, 'amp_slope_spread': f_spread_amp}

    ax_spread_zero.legend()
    ax_spread_base_slope.legend()
    ax_spread_amp_slope.legend()

    fig_spread_zero.tight_layout()
    fig_spread_base_slope.tight_layout()
    fig_spread_amp_slope.tight_layout()

    return interpolations


def plot_protons_fits_sim(df):
    fig_amp, ax_amp = plt.subplots()
    ax_amp.set_ylabel('slope')
    ax_amp.set_xlabel('amp')

    fig_spread, ax_spread = plt.subplots()
    ax_spread.set_ylabel('slope')
    ax_spread.set_xlabel('spread')

    spreads = np.unique(df['spread'])
    for spread in spreads:
        df_spread = df[df['spread'] == spread]
        ax_amp.errorbar(df_spread['amp'], df_spread['slope'], yerr=df_spread['slope_err'], ls='none', marker='o',
                        label=spread)

    amps = np.unique(df['amp'])
    for amp in amps:
        df_amp = df[df['amp'] == amp]
        ax_spread.errorbar(df_amp['spread'], df_amp['slope'], yerr=df_amp['slope_err'], ls='none', marker='o',
                           label=amp)

    ax_amp.legend()
    ax_spread.legend()

    fig_sa, ax_sa = plt.subplots()
    ax_sa.set_xlabel('amp')
    ax_sa.set_ylabel('spread')
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=min(df['slope']), vmax=0)
    for data_set in np.unique(df['name']):
        df_set = df[df['name'] == data_set]
        ax_sa.scatter(df_set['amp'], df_set['spread'], marker='o', color=cmap(norm(df_set['slope'])))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig_sa.colorbar(sm)

    fig_3d = plt.figure()
    ax_3d = plt.axes(projection='3d')
    df_sim = df[df['name'].str.contains('sim_')]
    ax_3d.plot_trisurf(df_sim['amp'], df_sim['spread'], df_sim['slope'], cmap='viridis', edgecolor='none')

    f = interp2d(df_sim['amp'], df_sim['spread'], df_sim['slope'], kind='cubic')
    x = np.linspace(df_sim['amp'].min(), df_sim['amp'].max(), 100)
    y = np.linspace(df_sim['spread'].min(), df_sim['spread'].max(), 100)
    xx, yy = np.meshgrid(x, y)
    fig_3d_interp = plt.figure()
    ax_3d_interp = plt.axes(projection='3d')
    z = f(x, y)
    print(z)
    ax_3d_interp.plot_surface(xx, yy, z, cmap='viridis', edgecolor='none')
    ax_3d_interp.set_xlabel('amp')
    ax_3d_interp.set_ylabel('spread')
    ax_3d_interp.set_zlabel('slope')
    ax_3d_interp.axhline(0.002)


def plot_protons_fits_vs_energy(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None):
    fig_slope, ax_slope = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax_slope.axhline(0, color='gray')
    fig_slope.canvas.manager.set_window_title(f'Slopes vs Energy')
    fig_int, ax_int = plt.subplots()
    ax_int.axhline(1, color='gray')
    fig_int.canvas.manager.set_window_title(f'Intercepts vs Energy')
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='energy')
        if data_sets_labels is None:
            lab = data_set
        else:
            lab = data_sets_labels[data_set]
        if data_sets_colors is None:
            ax_slope.errorbar(df_set['energy'], df_set['slope'], yerr=df_set['slope_err'], ls='none', marker='o',
                              label=lab)
            ax_int.errorbar(df_set['energy'], df_set['int'], yerr=df_set['int_err'], ls='none', marker='o', label=lab)
        else:
            ax_slope.errorbar(df_set['energy'], df_set['slope'], yerr=df_set['slope_err'], ls='none', marker='o',
                              color=data_sets_colors[data_set], label=lab)
            ax_int.errorbar(df_set['energy'], df_set['int'], yerr=df_set['int_err'], ls='none', marker='o',
                            color=data_sets_colors[data_set], label=lab)
    ax_slope.set_ylabel('Slope of Raw/Mix SD vs Total Protons per Event')
    ax_slope.set_xlabel('Energy (GeV)')
    ax_slope.grid()
    ax_int.set_ylabel('Intercept')
    ax_int.set_xlabel('Energy (GeV)')
    ax_int.grid()
    if title:
        ax_slope.set_title(title)
        ax_int.set_title(title)
    legend_slope = ax_slope.legend()
    # legend_slope.get_frame().set_alpha(0)
    legend_int = ax_int.legend()
    # legend_int.get_frame().set_alpha(0)
    fig_slope.tight_layout()
    fig_int.tight_layout()


def plot_protons_avgs_vs_energy(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None, alpha=1,
                                kin_info_loc=(0.12, 0.68), star_prelim_loc=None, marker_map=None, leg_loc=None,
                                data_sets_bands=None, legend_order=None, ylim=None, fig_splt_adj=None):
    fig_avg, ax_avg = plt.subplots()
    ax_avg.axhline(0, color='black', zorder=0)
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Averages vs Energy')
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='energy')
        if data_sets_labels is None:
            lab = data_set
        else:
            lab = data_sets_labels[data_set]
        if marker_map is None:
            marker = 'o'
        else:
            marker = marker_map[data_set]['diff']
        if data_sets_colors is None:
            if data_sets_bands is not None and data_set in data_sets_bands:
                ax_avg.fill_between(df_set['energy'], df_set['avg'] - df_set['avg_err'],
                                    df_set['avg'] + df_set['avg_err'], label=lab, alpha=alpha)
            else:
                ax_avg.errorbar(df_set['energy'], df_set['avg'], yerr=df_set['avg_err'], ls='none', marker=marker,
                                label=lab, alpha=alpha, markersize=9)
                if 'sys' in df:
                    ax_avg.errorbar(df_set['energy'], df_set['avg'], yerr=df_set['sys'], ls='', marker='', elinewidth=4,
                                    alpha=0.3)
        else:
            if data_sets_bands is not None and data_set in data_sets_bands:
                ax_avg.fill_between(df_set['energy'], df_set['avg'] - df_set['avg_err'],
                                    df_set['avg'] + df_set['avg_err'], color=data_sets_colors[data_set],
                                    label=lab, alpha=alpha)
            else:
                ax_avg.errorbar(df_set['energy'], df_set['avg'], yerr=df_set['avg_err'], ls='none', marker=marker,
                                color=data_sets_colors[data_set], label=lab, alpha=alpha, markersize=9)
                if 'sys' in df:
                    ax_avg.errorbar(df_set['energy'], df_set['avg'], yerr=df_set['sys'], ls='', marker='', elinewidth=4,
                                    color=data_sets_colors[data_set], alpha=0.3)
    ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle$', fontsize='x-large')
    ax_avg.set_xlabel('Energy (GeV)', fontsize='x-large')
    ax_avg.set_axisbelow(True)
    # ax_avg.grid()
    if title:
        ax_avg.set_title(title)
    if kin_info_loc is not None:
        eta_line = r'|y| < 0.5'
        pt_line = r'0.4 < $p_T$ < 2.0 GeV'
        pt_line = f'{pt_line}\n0-5% Centrality\n120° Partitions'
        samples_line = None  # '72 Samples/Event'
        kin_text = f'Au+Au\n{eta_line}\n{pt_line}'
        kin_text = f'{kin_text}\n{samples_line}' if samples_line is not None else kin_text
        ax_avg.text(*kin_info_loc, kin_text, ha='left', va='bottom', transform=ax_avg.transAxes,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
    if star_prelim_loc is not None:
        ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
    if legend_order is not None:
        handles, labels = ax_avg.get_legend_handles_labels()
        handles_dict = dict(zip(labels, handles))
        ordered_handles = [handles_dict[label] for label in legend_order]
        if leg_loc is None:
            ax_avg.legend(handles=ordered_handles, labels=legend_order)
        else:
            ax_avg.legend(loc=leg_loc, handles=ordered_handles, labels=legend_order)
    else:
        if leg_loc is None:
            ax_avg.legend()
        else:
            ax_avg.legend(loc=leg_loc)
    if ylim is not None:
        ax_avg.set_ylim(*ylim)
    fig_avg.tight_layout()
    if fig_splt_adj is not None:
        fig_avg.subplots_adjust(**fig_splt_adj)
    else:
        fig_avg.subplots_adjust(top=0.995, right=0.995, bottom=0.087, left=0.153)


def plot_protons_fits_vs_cent(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                              fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-'):
    fig_slope, ax_slope = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax_slope.axhline(0, color='black')
    fig_slope.canvas.manager.set_window_title(f'Slopes vs Centrality')
    energies = pd.unique(df['energy'])
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(1, 0.4, len(energies))])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy.sort_values(by='cent')
            if data_sets_labels is None:
                lab = data_set
            else:
                lab = data_sets_labels[data_set]
            if len(energies) > 1:
                lab += f' {energy}GeV'

            if cent_ref is None:
                x = df_energy['cent']
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            ls = 'none' if fit else ls
            if colors is None and color is None:
                ax_slope.errorbar(x, df_energy['slope'], xerr=x_err, yerr=df_energy['slope_err'], marker='o', ls=ls,
                                  label=lab, alpha=0.6)
            else:
                ax_slope.errorbar(x, df_energy['slope'], xerr=x_err, yerr=df_energy['slope_err'], marker='o', ls=ls,
                                  color=color, label=lab, alpha=0.6)
            if fit:
                p0 = [-0.02, 0.0001]
                x_fit = np.linspace(min(x), max(x), 1000)
                odr_model = odr.Model(inv_sqrtx_odr)
                fit_index_highs = range(4, len(x))
                colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                for i in fit_index_highs:
                    color_fit = next(colors_fit)
                    sx = None if x_err is None else x_err[:i]
                    odr_data = odr.RealData(x[:i], df_energy['slope'][:i], sx=sx, sy=df_energy['slope_err'][:i])
                    inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                    odr_out = inv_sqrt_odr.run()
                    ax_slope.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                    ax_slope.axhline(odr_out.beta[1], color=color_fit, ls='--')
                    ax_slope.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                                     color=color_fit, alpha=0.4)
                    print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

                # popt, pcov = cf(inv_sqrtx, x, df_energy['slope'], sigma=df_energy['slope_err'], absolute_sigma=True)
                # perr = np.sqrt(np.diag(pcov))
                # x_fit = np.linspace(min(x), max(x), 1000)
                # ax_slope.plot(x_fit, inv_sqrtx(x_fit, *popt), alpha=0.6, color=color)
                # ax_slope.axhline(popt[1], color=color, ls='--')
                # ax_slope.axhspan(popt[1] - perr[1], popt[1] + perr[1], color=color, alpha=0.4)
                # print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(popt, perr)]}')
    ax_slope.set_ylabel('Slope of Raw/Mix SD vs Total Protons per Event')
    if ref_type is None:
        ax_slope.set_xlabel('Centrality')
    else:
        ax_slope.set_xlabel('Reference Multiplicity')
    ax_slope.grid()
    if title:
        ax_slope.set_title(title)
    legend_slope = ax_slope.legend()
    # legend_slope.get_frame().set_alpha(0)
    fig_slope.tight_layout()
    fig_slope.subplots_adjust(top=0.995, right=0.995, bottom=0.085, left=0.14)


def plot_protons_avgs_vs_cent(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                              fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                              alpha=0.6, errbar_alpha=0.2, band_alpha=0.2, kin_info_loc=(0.18, 0.1),
                              star_prelim_loc=None, data_sets_energies_colors=None, marker_map=None,
                              data_sets_bands=None, xerr=True, xlim=None, figsize=(6.66, 5), fig_splt_adj=None,
                              print_data=False, ylim=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    fig_avg, ax_avg = plt.subplots(figsize=figsize, dpi=144)
    ax_avg.axhline(0, color='black')
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Avg vs Centrality')
    energies = pd.unique(df['energy'])
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(0.1, 0.9, len(energies))])
        elif data_sets_energies_colors is not None:
            colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None

        for energy in pd.unique(df_set['energy']):
            if marker_map is None:
                marker = 'o'
            elif data_set in marker_map and 'diff' in marker_map[data_set]:
                marker = marker_map[data_set]['diff']
            elif data_set in marker_map and energy in marker_map[data_set]:
                marker = marker_map[data_set][energy]
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if len(data_sets_plt) == 1 and len(energies) > 1:
                lab = ''
            elif data_sets_labels is None:
                lab = data_set
            else:
                lab = data_sets_labels[data_set]
            if len(energies) > 1:
                lab += f' {energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                if xerr is False:
                    x_err = None
                else:
                    x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                             df_energy['cent']]
            ls = 'none' if fit else ls
            if colors is None and color is None:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax_avg.fill_between(x, df_energy['avg'] - df_energy['avg_err'],
                                        df_energy['avg'] + df_energy['avg_err'], label=lab, alpha=band_alpha)
                else:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    elinewidth=4, alpha=errbar_alpha)
            else:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax_avg.fill_between(x, df_energy['avg'] - df_energy['avg_err'],
                                        df_energy['avg'] + df_energy['avg_err'], label=lab, alpha=band_alpha,
                                        color=color)
                else:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    color=color, label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    color=color, elinewidth=4, alpha=errbar_alpha)
            if print_data:
                print(f'\n{lab} 120° Partitions:')
                print(f'\'x\': [{", ".join([str(xi) for xi in x])}]')
                print(f'\'val\': [{", ".join([str(val) for val in df_energy["avg"]])}]')
                print(f'\'err\': [{", ".join([str(err) for err in df_energy["avg_err"]])}]')
                if 'sys' in df_energy.columns:
                    print(f'\'sys\': [{", ".join([str(sys) for sys in df_energy["sys"]])}]')
            if fit:
                p0 = [-0.02, 0.0001]
                x_fit = np.linspace(min(x), max(x), 1000)
                odr_model = odr.Model(inv_sqrtx_odr)
                fit_index_highs = range(4, len(x))
                colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                for i in fit_index_highs:
                    color_fit = next(colors_fit)
                    sx = None if x_err is None else x_err[:i]
                    odr_data = odr.RealData(x[:i], df_energy['slope'][:i], sx=sx, sy=df_energy['slope_err'][:i])
                    inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                    odr_out = inv_sqrt_odr.run()
                    ax_avg.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                    ax_avg.axhline(odr_out.beta[1], color=color_fit, ls='--')
                    ax_avg.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                                   color=color_fit, alpha=0.4)
                    print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

    ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle$', fontsize='x-large')
    if ref_type is None:
        ax_avg.set_xlabel('Centrality', fontsize='x-large')
    elif ref_type == 'npart':
        ax_avg.set_xlabel('Number of Participant Nucleons', fontsize='x-large')
    else:
        ax_avg.set_xlabel('Reference Multiplicity', fontsize='x-large')
    # ax_avg.grid()
    if title:
        ax_avg.set_title(title)
    if kin_info_loc is not None:
        eta_line = r'|y| < 0.5'
        pt_line = r'0.4 < $p_T$ < 2.0 GeV'
        pt_line = f'{pt_line}\n120° Partitions'
        samples_line = None  # '72 Samples/Event'
        kin_text = f'Au+Au\n{eta_line}\n{pt_line}'
        kin_text = f'{kin_text}\n{samples_line}' if samples_line is not None else kin_text
        ax_avg.text(*kin_info_loc, kin_text, ha='left', va='bottom', transform=ax_avg.transAxes,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
    if star_prelim_loc is not None:
        ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
    if xlim is not None:
        ax_avg.set_xlim(*xlim)
    if ylim is not None:
        ax_avg.set_ylim(*ylim)
    legend_avg = ax_avg.legend()
    # legend_avg.get_frame().set_alpha(0)
    fig_avg.tight_layout()
    if fig_splt_adj is not None:
        fig_avg.subplots_adjust(**fig_splt_adj)
    else:
        fig_avg.subplots_adjust(top=0.995, right=0.995, bottom=0.085, left=0.14)


def plot_dsig_avg_vs_cent_2panel(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                 fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                 alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                 data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                 legend_order=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
    for ax_avg in ax_avgs:
        ax_avg.axhline(0, color='black')
    ax_avgs = dict(zip(data_sets_plt, ax_avgs))
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Avg vs Centrality')
    energies = pd.unique(df['energy'])
    ampt_xs, ampt_ys, ampt_yerrs = [], [], []
    for data_set in data_sets_plt:
        ax_avg = ax_avgs[data_set]
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(0.1, 0.9, len(energies))])
        elif data_sets_energies_colors is not None:
            colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        if marker_map is None:
            marker = 'o'
        else:
            marker = marker_map[data_set]['diff']
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if data_sets_labels is not None:
                lab = data_sets_labels[data_set]
            else:
                lab = ''
            if len(energies) > 1:
                lab += f'{energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            ls = 'none' if fit else ls
            if 'ampt' in data_set:
                ampt_xs.extend(x)
                ampt_ys.extend(df_energy['avg'])
                ampt_yerrs.extend(df_energy['avg_err'])
            if colors is None and color is None:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax_avg.fill_between(x, df_energy['avg'] - df_energy['avg_err'],
                                        df_energy['avg'] + df_energy['avg_err'], label=lab, alpha=0.4)
                else:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    elinewidth=4, alpha=errbar_alpha)
            else:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax_avg.fill_between(x, df_energy['avg'] - df_energy['avg_err'],
                                        df_energy['avg'] + df_energy['avg_err'], label=lab, alpha=0.4, color=color)
                else:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    color=color, label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax_avg.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    color=color, elinewidth=4, alpha=errbar_alpha)
            if fit:
                p0 = [-0.02, 0.0001]
                x_fit = np.linspace(min(x), max(x), 1000)
                odr_model = odr.Model(inv_sqrtx_odr)
                fit_index_highs = range(4, len(x))
                colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                for i in fit_index_highs:
                    color_fit = next(colors_fit)
                    sx = None if x_err is None else x_err[:i]
                    odr_data = odr.RealData(x[:i], df_energy['slope'][:i], sx=sx, sy=df_energy['slope_err'][:i])
                    inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                    odr_out = inv_sqrt_odr.run()
                    ax_avg.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                    ax_avg.axhline(odr_out.beta[1], color=color_fit, ls='--')
                    ax_avg.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                                   color=color_fit, alpha=0.4)
                    print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

    fit_func = inv_x_plus_sqrt
    p0 = (-0.15, 0, 0)
    # popt, pcov = cf(fit_func, ampt_xs, ampt_ys, sigma=ampt_yerrs, absolute_sigma=True)
    popt, pcov = cf(fit_func, ampt_xs, ampt_ys)
    perr = np.sqrt(np.diag(pcov))
    letters = ['a', 'b', 'c', 'd', 'e', 'f'][:len(popt)]
    print(', '.join([f'{par} = {meas}' for par, meas in zip(letters, [Measure(v, e) for v, e in zip(popt, perr)])]))
    xs_plot = np.linspace(min(ampt_xs), max(ampt_xs), 1000)

    for set_i, data_set in enumerate(data_sets_plt):
        ax_avg = ax_avgs[data_set]
        # ax_avg.fill_between(xs_plot, fit_func(xs_plot, *(popt - perr)), fit_func(xs_plot, *(popt + perr)),
        #                     color='black')
        ax_avg.plot(xs_plot, fit_func(xs_plot, *popt), color='black', ls='--', label='AMPT Fit')
        # ax_avg.plot(xs_plot, fit_func(xs_plot, *p0), color='black', ls=':')
        # if 'ampt' in data_set:
        #     ax_avg.text(0.3, 0.2, f'AMPT', ha='left', va='bottom', transform=ax_avg.transAxes, fontsize=15,
        #                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if set_i == 0:
            ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
        if set_i == 1:
            ax_avg.tick_params(axis='y', which='both', left=False, right=False)
        if ref_type is None:
            ax_avg.set_xlabel('Centrality')
        elif ref_type == 'npart':
            ax_avg.set_xlabel('Number of Participant Nucleons')
        else:
            ax_avg.set_xlabel('Reference Multiplicity')
        if xlim is not None:
            ax_avg.set_xlim(xlim)
        ax_avg.set_axisbelow(True)
        # ax_avg.grid()
        if kin_info_loc is not None and set_i == 0:
            eta_line = r'|y| < 0.5'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions', ha='left', va='bottom',
                        transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if star_prelim_loc is not None and data_set == 'bes_def':
            ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
        if legend_order is not None:
            handles, labels = ax_avg.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            ordered_handles = [handles_dict[label] for label in legend_order]
            legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order)
        else:
            legend_avg = ax_avg.legend()
        # legend_avg = ax_avg.legend()
        legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
        legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
        # legend_avg.get_frame().set_alpha(0)
    if title and title != '':
        fig_avg.suptitle(title)
    fig_avg.tight_layout()
    fig_avg.subplots_adjust(top=0.94, right=0.993, bottom=0.088, left=0.095, wspace=0)


def plot_dsig_avg_vs_cent_2panel3(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                  fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                  alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                  data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                  legend_order=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
    fig_star, ax_star = plt.subplots(figsize=(7, 5.1), dpi=144)
    for ax_avg in ax_avgs:
        ax_avg.axhline(0, color='black')
    ax_star.axhline(0, color='black')
    ax_avgs = dict(zip(data_sets_plt, ax_avgs))
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Avg vs Centrality')
    energies = pd.unique(df['energy'])
    ampt_xs, ampt_ys, ampt_yerrs = [], [], []
    for data_set in data_sets_plt:
        ax_avg = ax_avgs[data_set]
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(0.1, 0.9, len(energies))])
        elif data_sets_energies_colors is not None:
            colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        if marker_map is None:
            marker = 'o'
        else:
            marker = marker_map[data_set]['diff']
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if data_sets_labels is not None:
                lab = data_sets_labels[data_set]
            else:
                lab = ''
            if len(energies) > 1:
                lab += f'{energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            ls = 'none' if fit else ls
            x = np.array(x)
            if 'ampt' in data_set:
                ampt_xs.extend(x)
                ampt_ys.extend(df_energy['avg'])
                ampt_yerrs.extend(df_energy['avg_err'])
            axs = [ax_avg]
            if 'bes' in data_set:
                axs.append(ax_star)
            for ax in axs:
                if colors is None and color is None:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, (df_energy['avg'] - df_energy['avg_err']),
                                        (df_energy['avg'] + df_energy['avg_err']), label=lab, alpha=0.4)
                    else:
                        ax.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    label=lab, alpha=alpha, size=9)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    elinewidth=4, alpha=errbar_alpha)
                else:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, df_energy['avg'] - df_energy['avg_err'],
                                        df_energy['avg'] + df_energy['avg_err'], label=lab, alpha=0.4, color=color)
                    else:
                        ax.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    color=color, label=lab, alpha=alpha, markersize=9)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    color=color, elinewidth=4, alpha=errbar_alpha)
                if fit:
                    p0 = [-0.02, 0.0001]
                    x_fit = np.linspace(min(x), max(x), 1000)
                    odr_model = odr.Model(inv_sqrtx_odr)
                    fit_index_highs = range(4, len(x))
                    colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                    for i in fit_index_highs:
                        color_fit = next(colors_fit)
                        sx = None if x_err is None else x_err[:i]
                        odr_data = odr.RealData(x[:i], df_energy['slope'][:i], sx=sx, sy=df_energy['slope_err'][:i])
                        inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                        odr_out = inv_sqrt_odr.run()
                        ax.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                        ax.axhline(odr_out.beta[1], color=color_fit, ls='--')
                        ax.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                                   color=color_fit, alpha=0.4)
                        print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

    fit_func = inv_x_plus_sqrt
    p0 = (-0.15, 0, 0)
    # popt, pcov = cf(fit_func, ampt_xs, ampt_ys, sigma=ampt_yerrs, absolute_sigma=True)
    popt, pcov = cf(fit_func, ampt_xs, ampt_ys)
    perr = np.sqrt(np.diag(pcov))
    letters = ['a', 'b', 'c', 'd', 'e', 'f'][:len(popt)]
    print(', '.join([f'{par} = {meas}' for par, meas in zip(letters, [Measure(v, e) for v, e in zip(popt, perr)])]))
    xs_plot = np.linspace(min(ampt_xs), max(ampt_xs), 1000)

    for set_i, data_set in enumerate(data_sets_plt):
        ax_avg = ax_avgs[data_set]
        # ax_avg.set_ylim(-0.0055, 0.0005)
        # ax_avg.fill_between(xs_plot, fit_func(xs_plot, *(popt - perr)), fit_func(xs_plot, *(popt + perr)),
        #                     color='black')
        ax_avg.plot(xs_plot, fit_func(xs_plot, *popt), color='black', ls='--', label='AMPT Fit')
        # ax_avg.plot(xs_plot, fit_func(xs_plot, *p0), color='black', ls=':')
        # if 'ampt' in data_set:
        #     ax_avg.text(0.3, 0.2, f'AMPT', ha='left', va='bottom', transform=ax_avg.transAxes, fontsize=15,
        #                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if set_i == 0:
            ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
        if set_i == 1:
            ax_avg.tick_params(axis='y', which='both', left=False, right=False)
        if ref_type is None:
            ax_avg.set_xlabel('Centrality')
        elif ref_type == 'npart':
            ax_avg.set_xlabel('Number of Participant Nucleons')
        else:
            ax_avg.set_xlabel('Reference Multiplicity')
        if xlim is not None:
            ax_avg.set_xlim(xlim)
        ax_avg.set_axisbelow(True)
        # ax_avg.grid()
        if kin_info_loc is not None and set_i == 0:
            eta_line = r'|y| < 0.5'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions', ha='left', va='bottom',
                        transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if star_prelim_loc is not None and data_set == 'bes_def':
            ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
        if legend_order is not None:
            handles, labels = ax_avg.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            ordered_handles = [handles_dict[label] for label in legend_order]
            legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order, loc='lower right')
        else:
            legend_avg = ax_avg.legend()
        # legend_avg = ax_avg.legend()
        legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
        legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
        # legend_avg.get_frame().set_alpha(0)

    ax_star.set_ylim(-0.0055, 0.0009)
    ax_star.plot(xs_plot, fit_func(xs_plot, *popt), color='black', ls='--', label='AMPT Fit')
    ax_star.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
    if ref_type is None:
        ax_star.set_xlabel('Centrality')
    elif ref_type == 'npart':
        ax_star.set_xlabel('Number of Participant Nucleons')
    else:
        ax_star.set_xlabel('Reference Multiplicity')
    if xlim is not None:
        ax_star.set_xlim(xlim)
    ax_star.set_axisbelow(True)
    ax_star.grid()
    if kin_info_loc is not None and set_i == 0:
        eta_line = r'|y| < 0.5'
        pt_line = r'0.4 < $p_T$ < 2.0 GeV'
        ax_star.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions', ha='left', va='bottom',
                     transform=ax_avg.transAxes,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
    if star_prelim_loc is not None:
        ax_star.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_star.transAxes,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
    if legend_order is not None:
        handles, labels = ax_star.get_legend_handles_labels()
        handles_dict = dict(zip(labels, handles))
        ordered_handles = [handles_dict[label] for label in legend_order]
        legend_star = ax_star.legend(handles=ordered_handles, labels=legend_order, loc='lower right')
    else:
        legend_star = ax_star.legend()
    # legend_avg = ax_avg.legend()
    legend_star.set_title('STAR', prop={'size': 'large', 'weight': 'bold'})
    # legend_avg.get_frame().set_alpha(0)

    if title and title != '':
        fig_avg.suptitle(title)
        fig_star.suptitle(title)
    fig_avg.tight_layout()
    fig_star.tight_layout()
    fig_avg.subplots_adjust(top=0.99, right=0.993, bottom=0.1, left=0.095, wspace=0)
    fig_star.subplots_adjust(top=0.99, right=0.993, bottom=0.1, left=0.14)


def plot_dsig_avg_vs_cent_2panel2(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                  fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                  alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                  data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                  legend_order=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
    fig_star, ax_star = plt.subplots(figsize=(7, 5.1), dpi=144)
    for ax_avg in ax_avgs:
        ax_avg.axhline(0, color='black')
    ax_avgs = dict(zip(data_sets_plt, ax_avgs))
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Avg vs Centrality')
    energies = pd.unique(df['energy'])
    ampt_xs, ampt_ys, ampt_yerrs = [], [], []
    for data_set in data_sets_plt:
        ax_avg = ax_avgs[data_set]
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(0.1, 0.9, len(energies))])
        elif data_sets_energies_colors is not None:
            colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        if marker_map is None:
            marker = 'o'
        else:
            marker = marker_map[data_set]['diff']
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if data_sets_labels is not None:
                lab = data_sets_labels[data_set]
            else:
                lab = ''
            if len(energies) > 1:
                lab += f'{energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            ls = 'none' if fit else ls
            x = np.array(x)
            if 'ampt' in data_set:
                ampt_xs.extend(x)
                ampt_ys.extend(df_energy['avg'])
                ampt_yerrs.extend(df_energy['avg_err'])
            axs = [ax_avg]
            if 'bes' in data_set:
                axs.append(ax_star)
            for ax in axs:
                if colors is None and color is None:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, x * (df_energy['avg'] - df_energy['avg_err']),
                                        x * (df_energy['avg'] + df_energy['avg_err']), label=lab, alpha=0.4)
                    else:
                        ax.errorbar(x, x * df_energy['avg'], xerr=x_err, yerr=x * df_energy['avg_err'], marker=marker,
                                    ls=ls,
                                    label=lab, alpha=alpha)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, x * df_energy['avg'], xerr=0, yerr=x * df_energy['sys'], marker='', ls='',
                                    elinewidth=4, alpha=errbar_alpha)
                else:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, x * (df_energy['avg'] - df_energy['avg_err']),
                                        x * (df_energy['avg'] + df_energy['avg_err']), label=lab, alpha=0.4,
                                        color=color)
                    else:
                        ax.errorbar(x, x * df_energy['avg'], xerr=x_err, yerr=x * df_energy['avg_err'], marker=marker,
                                    ls=ls,
                                    color=color, label=lab, alpha=alpha)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, x * df_energy['avg'], xerr=0, yerr=x * df_energy['sys'], marker='', ls='',
                                    color=color, elinewidth=4, alpha=errbar_alpha)
                if fit:
                    p0 = [-0.02, 0.0001]
                    x_fit = np.linspace(min(x), max(x), 1000)
                    odr_model = odr.Model(inv_sqrtx_odr)
                    fit_index_highs = range(4, len(x))
                    colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                    for i in fit_index_highs:
                        color_fit = next(colors_fit)
                        sx = None if x_err is None else x_err[:i]
                        odr_data = odr.RealData(x[:i], df_energy['slope'][:i], sx=sx, sy=df_energy['slope_err'][:i])
                        inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                        odr_out = inv_sqrt_odr.run()
                        ax.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                        ax.axhline(odr_out.beta[1], color=color_fit, ls='--')
                        ax.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                                   color=color_fit, alpha=0.4)
                        print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

    fit_func = inv_x_plus_sqrt
    p0 = (-0.15, 0, 0)
    # popt, pcov = cf(fit_func, ampt_xs, ampt_ys, sigma=ampt_yerrs, absolute_sigma=True)
    popt, pcov = cf(fit_func, ampt_xs, ampt_ys)
    perr = np.sqrt(np.diag(pcov))
    letters = ['a', 'b', 'c', 'd', 'e', 'f'][:len(popt)]
    print(', '.join([f'{par} = {meas}' for par, meas in zip(letters, [Measure(v, e) for v, e in zip(popt, perr)])]))
    xs_plot = np.linspace(min(ampt_xs), max(ampt_xs), 1000)

    for set_i, data_set in enumerate(data_sets_plt):
        ax_avg = ax_avgs[data_set]
        ax_avg.set_ylim(-0.315, 0.095)
        ax_star.set_ylim(-0.315, 0.095)
        # ax_avg.fill_between(xs_plot, fit_func(xs_plot, *(popt - perr)), fit_func(xs_plot, *(popt + perr)),
        #                     color='black')
        ax_avg.plot(xs_plot, xs_plot * fit_func(xs_plot, *popt), color='black', ls='--', label='AMPT Fit')
        ax_star.plot(xs_plot, xs_plot * fit_func(xs_plot, *popt), color='black', ls='--', label='AMPT Fit')
        # ax_avg.plot(xs_plot, fit_func(xs_plot, *p0), color='black', ls=':')
        # if 'ampt' in data_set:
        #     ax_avg.text(0.3, 0.2, f'AMPT', ha='left', va='bottom', transform=ax_avg.transAxes, fontsize=15,
        #                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        ax_star.set_ylabel(r'$\langle\Delta\sigma^2\rangle$' ' x Reference Multiplicity')
        if set_i == 0:
            ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle$' ' x Reference Multiplicity')
        if set_i == 1:
            ax_avg.tick_params(axis='y', which='both', left=False, right=False)
        if ref_type is None:
            ax_avg.set_xlabel('Centrality')
            ax_star.set_xlabel('Centrality')
        elif ref_type == 'npart':
            ax_avg.set_xlabel('Number of Participant Nucleons')
            ax_star.set_xlabel('Number of Participant Nucleons')
        else:
            ax_avg.set_xlabel('Reference Multiplicity')
            ax_star.set_xlabel('Reference Multiplicity')
        if xlim is not None:
            ax_avg.set_xlim(xlim)
            ax_star.set_xlim(xlim)
        ax_avg.set_axisbelow(True)
        ax_star.set_axisbelow(True)
        ax_avg.grid()
        ax_star.grid()
        if kin_info_loc is not None and set_i == 0:
            eta_line = r'|y| < 0.5'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions\n72 Samples/Event', ha='left',
                        va='bottom', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if star_prelim_loc is not None and data_set == 'bes_def':
            ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
            ax_star.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
        if legend_order is not None:
            handles, labels = ax_avg.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            ordered_handles = [handles_dict[label] for label in legend_order]
            legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order, loc='upper right')
            handles, labels = ax_star.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            ordered_handles = [handles_dict[label] for label in legend_order]
            legend_star = ax_star.legend(handles=ordered_handles, labels=legend_order, loc='upper right')
        else:
            legend_avg = ax_avg.legend()
            legend_star = ax_star.legend()
        # legend_avg = ax_avg.legend()
        legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
        legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
        legend_star.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
        # legend_avg.get_frame().set_alpha(0)
    if title and title != '':
        fig_avg.suptitle(title)
        fig_star.suptitle(title)
    fig_avg.tight_layout()
    fig_star.tight_layout()
    fig_avg.subplots_adjust(top=0.995, right=0.995, bottom=0.088, left=0.063, wspace=0)
    fig_star.subplots_adjust(top=0.94, right=0.993, bottom=0.088, left=0.095)


def plot_dsig_avg_vs_cent_2panel62ref(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                      fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                      alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                      data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                      legend_order=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
    fig_62_fits, ax_62_fits = plt.subplots(figsize=(8, 4), dpi=144)
    for ax_avg in ax_avgs:
        ax_avg.axhline(0, color='black')
    ax_62_fits.axhline(0, color='black')
    ax_avgs = dict(zip(data_sets_plt, ax_avgs))
    fig_avg.canvas.manager.set_window_title(f'Dsigma2 Avg - 62 GeV vs Centrality')
    fig_62_fits.canvas.manager.set_window_title(f'62 GeV Fits')
    energies = pd.unique(df['energy'])
    energy_avgs = {data_set: {energy: {} for energy in energies} for data_set in data_sets_plt}
    for data_set in data_sets_plt:
        ax_avg = ax_avgs[data_set]
        df_set = df[df['name'] == data_set]
        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(0.1, 0.9, len(energies))])
        elif data_sets_energies_colors is not None:
            colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        if marker_map is None:
            marker = 'o'
        else:
            marker = marker_map[data_set]['diff']

        # Get 62 GeV fit, cent 2 and above
        df_62 = df_set[(df_set['energy'] == 62) & (df_set['cent'] > -1)]

        if cent_ref is None:
            x = df_62['cent']
            # x = [cent_map[xi] for xi in df_energy['cent']]
            x_err = None
        else:
            cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == 62)]
            x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                 df_62['cent']]
            x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                     df_62['cent']]

        func = inv_pow_x_const
        err = np.sqrt(df_62['avg_err'] ** 2 + df_62['sys'] ** 2)
        popt, pcov = cf(func, x, df_62['avg'], sigma=err, absolute_sigma=True)
        fit_meases = [Measure(var, err) for var, err in zip(popt, np.sqrt(np.diag(pcov)))]

        x_fit_plt = np.linspace(10, 800, 2000)
        if 'bes' in data_set:
            lab = 'STAR'
            color = 'black'
            xy_text = (0.65, 0.3)
        elif 'ampt' in data_set:
            lab = 'AMPT'
            color = 'red'
            xy_text = (0.38, 0.3)
        ax_62_fits.errorbar(x, df_62['avg'], xerr=x_err, yerr=df_62['avg_err'], marker='o', ls='', color=color,
                            label=lab, alpha=alpha)
        if 'sys' in df_62.columns:
            ax_62_fits.errorbar(x, df_62['avg'], xerr=0, yerr=df_62['sys'], marker='', ls='', color=color,
                                elinewidth=4, alpha=errbar_alpha)
        ax_62_fits.plot(x_fit_plt, func(x_fit_plt, *popt), alpha=0.6, color='orange')
        fit_lines = '\n'.join([fr'${letter} = {meas}$' for letter, meas in zip(['a', 'b', 'c'], fit_meases)])
        fit_text = f'{lab}\n{fit_lines}'
        props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
        ax_62_fits.text(*xy_text, fit_text, transform=ax_62_fits.transAxes, verticalalignment='top',
                        bbox=props, horizontalalignment='center')

        # Plot energies minus 62
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if data_sets_labels is not None:
                lab = data_sets_labels[data_set]
            else:
                lab = ''
            if len(energies) > 1:
                lab += f'{energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            ls = 'none' if fit else ls
            x = np.array(x)

            # Calculate difference from 62
            y_og = [Measure(avg, avg_err) for avg, avg_err in zip(df_energy['avg'], df_energy['avg_err'])]
            # y_62 = [func(xi, *fit_meases) for xi in x]
            y_62 = [func(xi, *popt) for xi in x]
            y = np.array(y_og) - np.array(y_62)

            ax = ax_avg
            if colors is None and color is None:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax.fill_between(x, [z.val - z.err for z in y], [z.val + z.err for z in y], label=lab, alpha=0.4)
                else:
                    ax.errorbar(x, [z.val for z in y], xerr=x_err, yerr=[z.err for z in y], marker=marker, ls=ls,
                                label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax.errorbar(x, [z.val for z in y], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                elinewidth=4, alpha=errbar_alpha)
            else:
                if data_sets_bands is not None and data_set in data_sets_bands:
                    ax.fill_between(x, [z.val - z.err for z in y], [z.val + z.err for z in y], label=lab, alpha=0.4,
                                    color=color)
                else:
                    ax.errorbar(x, [z.val for z in y], xerr=x_err, yerr=[z.err for z in y], marker=marker, ls=ls,
                                color=color, label=lab, alpha=alpha)
                if 'sys' in df_energy.columns:
                    ax.errorbar(x, [z.val for z in y], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                color=color, elinewidth=4, alpha=errbar_alpha)

            vals, errs = [z.val for z in y], np.array(
                [np.sqrt(z.err ** 2 + sys ** 2) for z, sys in zip(y, df_energy['sys'])])
            # print(f'{data_set}, {energy}: {", ".join([str(Measure(v, e)) for v, e in zip(vals, errs)])}')
            energy_avgs[data_set][energy].update({'avg': np.average(vals, weights=1 / errs ** 2),
                                                  'avg_err': 1 / np.sqrt(np.sum(1 / errs ** 2))})
            # ax.axhline(np.average(vals, weights=1 / errs**2), color=color)

    ax_62_fits.plot([], [], alpha=0.6, color='orange', label=r'$\frac{a}{M^c}+b$')
    for set_i, data_set in enumerate(data_sets_plt):
        ax_avg = ax_avgs[data_set]
        # ax_avg.set_ylim(-0.315, 0.095)
        ax_62_fits.set_ylim(-0.0042, 0.00045)
        # ax_avg.fill_between(xs_plot, fit_func(xs_plot, *(popt - perr)), fit_func(xs_plot, *(popt + perr)),
        #                     color='black')
        # ax_avg.plot(xs_plot, fit_func(xs_plot, *p0), color='black', ls=':')
        # if 'ampt' in data_set:
        #     ax_avg.text(0.3, 0.2, f'AMPT', ha='left', va='bottom', transform=ax_avg.transAxes, fontsize=15,
        #                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        ax_62_fits.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
        if set_i == 0:
            ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle - \langle\Delta\sigma^2\rangle_{62 \mathrm{\ GeV Fit}}$')
        if set_i == 1:
            ax_avg.tick_params(axis='y', which='both', left=False, right=False)
        if ref_type is None:
            ax_avg.set_xlabel('Centrality')
            ax_62_fits.set_xlabel('Centrality')
        elif ref_type == 'npart':
            ax_avg.set_xlabel('Number of Participant Nucleons')
            ax_62_fits.set_xlabel('Number of Participant Nucleons')
        else:
            ax_avg.set_xlabel('Reference Multiplicity')
            ax_62_fits.set_xlabel('Reference Multiplicity')
        if xlim is not None:
            ax_avg.set_xlim(xlim)
            ax_62_fits.set_xlim(xlim)
        ax_avg.set_axisbelow(True)
        ax_62_fits.set_axisbelow(True)
        ax_avg.grid()
        ax_62_fits.grid()
        if kin_info_loc is not None and set_i == 0:
            eta_line = r'|y| < 0.5'
            pt_line = r'0.4 < $p_T$ < 2.0 GeV'
            ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions\n72 Samples/Event', ha='left',
                        va='bottom', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
        if star_prelim_loc is not None and data_set == 'bes_def':
            ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
        if legend_order is not None:
            handles, labels = ax_avg.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            ordered_handles = [handles_dict[label] for label in legend_order]
            legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order, loc='upper right')
        else:
            legend_avg = ax_avg.legend()
        ax_62_fits.legend(title='62 GeV')
        # legend_avg = ax_avg.legend()
        legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
        legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
        # legend_avg.get_frame().set_alpha(0)
    if title and title != '':
        fig_avg.suptitle(title)
        fig_62_fits.suptitle(title)
    fig_avg.tight_layout()
    fig_62_fits.tight_layout()
    fig_avg.subplots_adjust(top=0.995, right=0.995, bottom=0.088, left=0.077, wspace=0)
    fig_62_fits.subplots_adjust(top=0.995, right=0.995, bottom=0.11, left=0.115)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=144)
    fig.canvas.manager.set_window_title('Centrality Averaged 62 GeV Difference')
    ax.axhline(0, color='black')
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel(
        r'$\langle \langle\Delta\sigma^2\rangle - \langle\Delta\sigma^2\rangle_{62 \mathrm{\ GeV Fit}} \rangle$')
    for data_set in data_sets_plt:
        if 'bes' in data_set:
            lab, color, marker = 'STAR', 'black', 'o'
        else:
            lab, color, marker = 'AMPT', 'red', 's'
        ax.errorbar(energies, [energy_avgs[data_set][energy]['avg'] for energy in energies],
                    [energy_avgs[data_set][energy]['avg_err'] for energy in energies],
                    marker=marker, color=color, label=lab, ls='')
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.995, right=0.995, bottom=0.09, left=0.135)


def plot_dsig_avg_vs_cent_fit(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                      fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                      alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                      data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                      legend_order=None, plot=True):
    print('Starting plot_dsig_avg_vs_cent_fit')
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}
    if plot:
        fig_fits, ax_fits = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
        fig_fit_devs, ax_fit_devs = plt.subplots(figsize=(8, 4), dpi=144)
        fig_fit_params, ax_fit_params = plt.subplots(figsize=(10, 6), dpi=144, nrows=3, sharex='all')
        for ax_avg in ax_fits:
            ax_avg.axhline(0, color='black')
        ax_fits = dict(zip(data_sets_plt, ax_fits))
        fig_fits.canvas.manager.set_window_title(f'Dsigma2 Avg Fits vs Centrality')
        fig_fit_params.canvas.manager.set_window_title(f'Dsigma2 Avg Fit Parameters')
    energies = pd.unique(df['energy'])
    all_fit_params = []
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        if plot:
            ax_avg = ax_fits[data_set]
            if data_sets_energies_cmaps is not None:
                colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                               for i in np.linspace(0.1, 0.9, len(energies))])
            elif data_sets_energies_colors is not None:
                colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
            elif data_sets_colors is not None:
                color, colors = data_sets_colors[data_set], None
            else:
                colors, color = None
            if marker_map is None:
                marker = 'o'
            else:
                marker = marker_map[data_set]['diff']

        # Fit energies and plot
        fit_params = {'data_set': [], 'energy': [], 'a': [], 'a_err': [], 'b': [], 'b_err': [], 'c': [], 'c_err': []}
        for energy in pd.unique(df_set['energy']):
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if plot:
                if colors is not None:
                    color = next(colors)
                if data_sets_labels is not None:
                    lab = data_sets_labels[data_set]
                else:
                    lab = ''
                if len(energies) > 1:
                    lab += f'{energy_map[energy]} GeV'
                ls = 'none' if fit else ls

            if cent_ref is None:
                x = df_energy['cent']
                # x = [cent_map[xi] for xi in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                print(f'data_set: {data_set}, energy: {energy}, ref_type: {ref_type}')
                print(cent_energy)
                for cent in df_energy['cent']:
                    print(f'cent_energy: {cent_energy[cent_energy["cent"] == cent][f"mean_{ref_type}_val"]}')
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            x = np.array(x)

            p0 = (0.0002, -100, 0.5)
            popt, pcov = cf(inv_pow_m, x, df_energy['avg'], p0=p0, sigma=df_energy['avg_err'], absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            # pmeas = [Measure(var, err) for var, err in zip(popt, perr)]
            fit_params['data_set'].append(data_set)
            fit_params['energy'].append(energy)
            fit_params['a'].append(popt[0])
            fit_params['a_err'].append(perr[0])
            fit_params['b'].append(popt[1])
            fit_params['b_err'].append(perr[1])
            fit_params['c'].append(popt[2])
            fit_params['c_err'].append(perr[2])
            x_fit_plt = np.linspace(x.min(), x.max(), 1000)

            if plot:
                ax = ax_avg
                if colors is None and color is None:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, df_energy['avg'] - df_energy['avg_err'], df_energy['avg'] + df_energy['avg_err'],
                                        label=lab, alpha=0.4)
                    else:
                        ax.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    label=lab, alpha=alpha)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    elinewidth=4, alpha=errbar_alpha)
                else:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax.fill_between(x, df_energy['avg'] - df_energy['avg_err'], df_energy['avg'] + df_energy['avg_err'],
                                        label=lab, alpha=0.4, color=color)
                    else:
                        ax.errorbar(x, df_energy['avg'], xerr=x_err, yerr=df_energy['avg_err'], marker=marker, ls=ls,
                                    color=color, label=lab, alpha=alpha)
                    if 'sys' in df_energy.columns:
                        ax.errorbar(x, df_energy['avg'], xerr=0, yerr=df_energy['sys'], marker='', ls='',
                                    color=color, elinewidth=4, alpha=errbar_alpha)
                if fit:
                    ax.plot(x_fit_plt, inv_pow_m(x_fit_plt, *popt), alpha=0.6, color=color)
                    ax.axhline(popt[0], color=color, ls='--')
                    ax.axhspan(popt[0] - perr[0], popt[0] + perr[0], color=color, alpha=0.4)
                    print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(popt, perr)]}')

        if plot:  # Plot fit parameters
            ax_fit_params[0].errorbar(fit_params['energy'], fit_params['a'], yerr=fit_params['a_err'], marker=marker,
                                        ls='', label=data_set)
            ax_fit_params[1].errorbar(fit_params['energy'], fit_params['b'], yerr=fit_params['b_err'], marker=marker,
                                        ls='', label=data_set)
            ax_fit_params[2].errorbar(fit_params['energy'], fit_params['c'], yerr=fit_params['c_err'], marker=marker,
                                        ls='', label=data_set)

        # Convert fit_params from dict of lists to list of dicts and extend all_fit_params
        fit_params = [{key: fit_params[key][i] for key in fit_params} for i in range(len(fit_params['data_set']))]
        all_fit_params.extend(fit_params)

            # vals, errs = [z.val for z in y], np.array(
            #     [np.sqrt(z.err ** 2 + sys ** 2) for z, sys in zip(y, df_energy['sys'])])
            # print(f'{data_set}, {energy}: {", ".join([str(Measure(v, e)) for v, e in zip(vals, errs)])}')
            # energy_avgs[data_set][energy].update({'avg': np.average(vals, weights=1 / errs ** 2),
            #                                       'avg_err': 1 / np.sqrt(np.sum(1 / errs ** 2))})
            # ax.axhline(np.average(vals, weights=1 / errs**2), color=color)

    if plot:
        for set_i, data_set in enumerate(data_sets_plt):
            ax_avg = ax_fits[data_set]
            # ax_avg.set_ylim(-0.315, 0.095)
            # ax_avg.fill_between(xs_plot, fit_func(xs_plot, *(popt - perr)), fit_func(xs_plot, *(popt + perr)),
            #                     color='black')
            # ax_avg.plot(xs_plot, fit_func(xs_plot, *p0), color='black', ls=':')
            # if 'ampt' in data_set:
            #     ax_avg.text(0.3, 0.2, f'AMPT', ha='left', va='bottom', transform=ax_avg.transAxes, fontsize=15,
            #                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
            if set_i == 0:
                ax_avg.set_ylabel(r'$\langle\Delta\sigma^2\rangle - \langle\Delta\sigma^2\rangle_{62 \mathrm{\ GeV Fit}}$')
            if set_i == 1:
                ax_avg.tick_params(axis='y', which='both', left=False, right=False)
            if ref_type is None:
                ax_avg.set_xlabel('Centrality')
            elif ref_type == 'npart':
                ax_avg.set_xlabel('Number of Participant Nucleons')
            else:
                ax_avg.set_xlabel('Reference Multiplicity')
            if xlim is not None:
                ax_avg.set_xlim(xlim)
            ax_avg.set_axisbelow(True)
            ax_avg.grid()
            if kin_info_loc is not None and set_i == 0:
                eta_line = r'|y| < 0.5'
                pt_line = r'0.4 < $p_T$ < 2.0 GeV'
                ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n120° Partitions\n72 Samples/Event', ha='left',
                            va='bottom', transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
            if star_prelim_loc is not None and data_set == 'bes_def':
                ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left', transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), va='top')
            if legend_order is not None:
                handles, labels = ax_avg.get_legend_handles_labels()
                handles_dict = dict(zip(labels, handles))
                ordered_handles = [handles_dict[label] for label in legend_order]
                legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order, loc='upper right')
            else:
                legend_avg = ax_avg.legend()
            # legend_avg = ax_avg.legend()
            legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
            legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
            # legend_avg.get_frame().set_alpha(0)
        if title and title != '':
            fig_fits.suptitle(title)
        fig_fits.tight_layout()
        fig_fits.subplots_adjust(top=0.995, right=0.995, bottom=0.088, left=0.077, wspace=0)

        ax_fit_params[0].set_ylabel('a')
        ax_fit_params[1].set_ylabel('b')
        ax_fit_params[2].set_ylabel('c')
        ax_fit_params[0].legend()
        ax_fit_params[0].axhline(0, color='gray', alpha=0.7, zorder=0)
        ax_fit_params[-1].set_xlabel('Energy (GeV)')
        fig_fit_params.tight_layout()
        fig_fit_params.subplots_adjust(top=0.995, right=0.995, bottom=0.088, left=0.077, hspace=0)

    return pd.DataFrame(all_fit_params)


def plot_div_fits_vs_cent(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                          fit=False, cent_ref=None, ref_type=None, ls='none', data_sets_energies_cmaps=None,
                          fit_boundary=0, data_sets_markers=None):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    ls = 'none' if fit else ls

    fig_base, ax_base = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax_base.axhline(0, color='black')
    fig_base.canvas.manager.set_window_title(f'Baselines vs Centrality')
    fig_zeros, ax_zeros = plt.subplots(figsize=(6.66, 5), dpi=144)
    fig_zeros.canvas.manager.set_window_title(f'Zeros vs Centrality')
    if fit:
        fig_base_res, ax_base_res = plt.subplots(figsize=(6.66, 5), dpi=144)
        ax_base_res.axhline(0, color='black')
        fig_base_res.canvas.manager.set_window_title(f'Baselines vs Centrality Fit Residuals')
    energies = pd.unique(df['energy'])
    bases, refs, cs = [], [], [0]  # Just to set x and y limits for plot
    fit_pars = {}

    ax_base.errorbar([], [], linestyle='None', marker='None', markersize=0, label=f' ')
    ax_zeros.errorbar([], [], linestyle='None', marker='None', markersize=0, label=f' ')
    for energy in pd.unique(df['energy']):
        ax_base.errorbar([], [], linestyle='None', marker='None', markersize=0, label=f'{energy} GeV'.rjust(10))
        ax_zeros.errorbar([], [], linestyle='None', marker='None', markersize=0, label=f'{energy} GeV'.rjust(10))

    for data_set in data_sets_plt:
        df_set = df[df['data_set'] == data_set]
        if data_sets_energies_cmaps is not None and (data_sets_colors is None or len(energies) > 1):
            # cmap = plt.cm.get_cmap(data_sets_energies_cmaps[data_set])
            # print(f'Number of cmap colors for {data_sets_energies_cmaps[data_set]}: {cmap.N}')
            # colors = iter([cmap(i) for i in range(cmap.N)])
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(1, 0.4, len(energies))])
        elif data_sets_colors is not None:
            if type(data_sets_colors[data_set]) is dict:
                color, colors = None, data_sets_colors[data_set]
            else:
                color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None, None
        ax_base.errorbar([], [], linestyle='None', marker='None', markersize=0, label=data_sets_labels[data_set])
        ax_zeros.errorbar([], [], linestyle='None', marker='None', markersize=0, label=data_sets_labels[data_set])
        # print(pd.unique(df_set['energy']))
        fit_pars.update({data_set: {'energy': [], 'const_val': [], 'const_err': [], 'isqrt_val': [], 'isqrt_err': []}})
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                if type(colors) is dict:
                    color = colors[energy]
                else:
                    color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy.sort_values(by='cent')
            if data_sets_labels is None:
                lab = data_set
            else:
                lab = data_sets_labels[data_set]
            if len(energies) > 1:
                lab += f' {energy}GeV'
            # if data_sets_markers is None:
            #     marker = 'o'
            # else:
            #     marker = data_sets_markers[data_set]
            if data_sets_markers is None:
                marker = 'o'
            elif data_set in data_sets_markers and isinstance(data_sets_markers[data_set], str):
                marker = data_sets_markers[data_set]
            elif data_set in data_sets_markers and 'diff' in data_sets_markers[data_set]:
                marker = data_sets_markers[data_set]['diff']
            elif data_set in data_sets_markers and energy in data_sets_markers[data_set]:
                marker = data_sets_markers[data_set][energy]

            if cent_ref is None:
                x = [cent_map[cent_i] for cent_i in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            lab = ' '  # f'{energy} GeV'
            if colors is None and color is None:
                ax_base.errorbar(x, df_energy['baseline'], xerr=x_err, yerr=df_energy['base_err'],
                                 marker=marker, ls=ls, label=lab, alpha=0.6)
                ax_zeros.errobar(x, df_energy['zero_mag'], xerr=x_err, yerr=df_energy['zero_mag_err'],
                                 marker=marker, ls='none', label=lab, alpha=0.6)
                if 'base_sys' in df_energy.columns:
                    ax_base.errorbar(x, df_energy['baseline'], xerr=0, yerr=df_energy['base_sys'],
                                     elinewidth=4, marker='', ls='', alpha=0.3)
                if 'zero_sys' in df_energy.columns:
                    ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=0, yerr=df_energy['zero_sys'],
                                      elinewidth=4, marker='', ls='', alpha=0.3)
            else:
                ax_base.errorbar(x, df_energy['baseline'], xerr=x_err, yerr=df_energy['base_err'],
                                 marker=marker, ls=ls, color=color, label=lab, alpha=0.6)
                ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=x_err, yerr=df_energy['zero_mag_err'],
                                  marker=marker, ls='none', color=color, label=lab, alpha=0.6)
                if 'base_sys' in df_energy.columns:
                    ax_base.errorbar(x, df_energy['baseline'], xerr=0, yerr=df_energy['base_sys'],
                                     elinewidth=4, marker='', ls='', color=color, alpha=0.3)
                if 'zero_sys' in df_energy.columns:
                    ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=0, yerr=df_energy['zero_sys'],
                                      elinewidth=4, marker='', ls='', color=color, alpha=0.3)
            bases.extend(list(df_energy['baseline']))
            refs.extend(list(x))
            if fit:
                # p0 = [-0.02]
                # p0 = [-0.02, 0.01]
                p0 = [-0.02, 0.01, 0, 0]
                # func = inv_sqrtx_odr
                func = inv_sqrtx_poly2_odr
                # func = inv_sqrtx_noconst_odr
                # func_cf = inv_sqrtx

                x_fit_plt = np.linspace(1, 800, 2000)
                odr_model = odr.Model(func)
                # odr_data = odr.RealData(x, df_energy['baseline'], sx=x_err, sy=df_energy['base_err'])
                x_fit, x_fit_err, y_fit, y_fit_err = zip(*[[xi, xi_err, yi, yi_err] for xi, yi, xi_err, yi_err in
                                                           zip(x, df_energy['baseline'], x_err, df_energy['base_err'])
                                                           if xi > fit_boundary])
                # print(x_fit, x_fit_err, y_fit, y_fit_err)
                odr_data = odr.RealData(x_fit, y_fit, sx=x_fit_err, sy=y_fit_err)
                inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                odr_out = inv_sqrt_odr.run()
                # ax_base.axhline(odr_out.beta[1], color=color, ls='-')
                # ax_base.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                #                 color=color, alpha=0.4)
                ax_base.plot(x_fit_plt, poly2_odr(odr_out.beta[1:], x_fit_plt), color=color)
                ax_base.plot(x_fit_plt, func(odr_out.beta, x_fit_plt), alpha=0.6, color=color)
                fit_meases = [Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]
                # print(f'{lab} ODR Fit: {fit_meases}')
                cs.append(odr_out.beta[1])
                fit_pars[data_set]['energy'].append(energy)
                fit_pars[data_set]['const_val'].append(odr_out.beta[1])
                fit_pars[data_set]['const_err'].append(odr_out.sd_beta[1])
                fit_pars[data_set]['isqrt_val'].append(odr_out.beta[0])
                fit_pars[data_set]['isqrt_err'].append(odr_out.sd_beta[0])

                # popt, pcov = cf(func_cf, x_fit, y_fit, sigma=y_fit_err, absolute_sigma=True, p0=p0)
                # ax_base.plot(x_fit_plt, func_cf(x_fit_plt, *popt), alpha=0.6, ls='--', color=color)
                # fit_meases_cf = [Measure(var, err) for var, err in zip(popt, np.sqrt(np.diag(pcov)))]
                # print(f'{lab} CF Fit: {fit_meases_cf}')

                y = [Measure(base, base_err) for base, base_err in zip(df_energy['baseline'], df_energy['base_err'])]
                y_fit = [func(fit_meases, xi) for xi in x]
                y_res = np.array(y) - np.array(y_fit)
                ax_base_res.errorbar(x, [z.val for z in y_res], [z.err for z in y_res], marker='o', ls=ls,
                                     color=color, label=lab, alpha=0.6)

                # p0 = [-0.026, 0.0006, 0.2, 100]
                # x_fit = np.linspace(1, 800, 2000)
                # func = inv_sqrtx_sigmoid_odr
                # odr_model = odr.Model(func)
                # odr_data = odr.RealData(x, df_energy['baseline'], sx=x_err, sy=df_energy['base_err'])
                # inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                # odr_out = inv_sqrt_odr.run()
                # ax_base.axhline(odr_out.beta[1], color=color, ls=':')
                # ax_base.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                #                 color=color, alpha=0.4)
                # ax_base.plot(x_fit, func(p0, x_fit), alpha=0.6, color='gray', ls=':')
                # ax_base.plot(x_fit, func(odr_out.beta, x_fit), alpha=0.6, color=color, ls=':')
                # print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

                # p0 = [-0.02, 0.0001, 1]
                # x_fit = np.linspace(1, 800, 2000)
                # odr_model = odr.Model(inv_invxpow_odr)
                # odr_data = odr.RealData(x, df_energy['baseline'], sx=x_err, sy=df_energy['base_err'])
                # inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                # odr_out = inv_sqrt_odr.run()
                # ax_base.axhline(odr_out.beta[1], color=color, ls=':')
                # ax_base.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                #                 color=color, alpha=0.4)
                # ax_base.plot(x_fit, inv_invxpow_odr(odr_out.beta, x_fit), alpha=0.6, color=color, ls=':')
                # print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')
                # # fit_pars[data_set]['energy'].append(energy)
                # # fit_pars[data_set]['const_val'].append(odr_out.beta[1])
                # # fit_pars[data_set]['const_err'].append(odr_out.sd_beta[1])

                # p0 = [-0.02, 1]
                # x_fit = np.linspace(1, 800, 2000)
                # odr_model = odr.Model(inv_invxpow_noc_odr)
                # odr_data = odr.RealData(x, df_energy['baseline'], sx=x_err, sy=df_energy['base_err'])
                # inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                # odr_out = inv_sqrt_odr.run()
                # ax_base.plot(x_fit, inv_invxpow_noc_odr(odr_out.beta, x_fit), alpha=0.6, color=color, ls=':')
                # print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')

                # p0 = [-0.02, 0.0001]
                # x_fit = np.linspace(1, 800, 2000)
                # odr_model = odr.Model(inv_invx_odr)
                # odr_data = odr.RealData(x, df_energy['baseline'], sx=x_err, sy=df_energy['base_err'])
                # inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                # odr_out = inv_sqrt_odr.run()
                # ax_base.axhline(odr_out.beta[1], color=color, ls='--')
                # ax_base.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                #                 color=color, alpha=0.4)
                # ax_base.plot(x_fit, inv_invx_odr(odr_out.beta, x_fit), alpha=0.6, color=color, ls='--')
                # print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')
                # # fit_pars[data_set]['energy'].append(energy)
                # # fit_pars[data_set]['const_val'].append(odr_out.beta[1])
                # # fit_pars[data_set]['const_err'].append(odr_out.sd_beta[1])
                # # cs.append(odr_out.beta[1])
                # # ax_base.set_ylim(ax_base_ylim)
                # # ax_base.set_xlim(ax_base_xlim)
                # # fit_index_highs = range(4, len(x))
                # # colors_fit = iter(plt.cm.rainbow(np.linspace(0, 1, len(fit_index_highs))))
                # # for i in fit_index_highs:
                # #     color_fit = next(colors_fit)
                # #     odr_data = odr.RealData(x[:i], df_energy['baseline'][:i], sx=x_err[:i],
                # #                             sy=df_energy['base_err'][:i])
                # #     inv_sqrt_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
                # #     odr_out = inv_sqrt_odr.run()
                # #     ax_base.plot(x_fit, inv_sqrtx(x_fit, *odr_out.beta), alpha=0.6, color=color_fit)
                # #     ax_base.axhline(odr_out.beta[1], color=color_fit, ls='--')
                # #     ax_base.axhspan(odr_out.beta[1] - odr_out.sd_beta[1], odr_out.beta[1] + odr_out.sd_beta[1],
                # #                     color=color_fit, alpha=0.4)
                # #     print(f'{lab} Fit: {[Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]}')
    ax_base.set_ylim(min(bases) * 1.1, max(cs) * 1.1)
    try:
        ax_base.set_xlim(0, max(refs) * 1.1)
    except TypeError:
        pass

    # ax_base.set_ylabel('Baseline of Slope vs Partition Width Fit')
    # ax_zeros.set_ylabel('Zeros of Slope vs Partition Width Fit')
    # ax_base.set_ylabel(r'Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit')
    # ax_zeros.set_ylabel(r'Zeros of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit')
    ax_base.set_ylabel(r'$b$')
    ax_zeros.set_ylabel(r'$z$')
    if ref_type is None:
        ax_base.set_xlabel('Centrality')
        ax_zeros.set_xlabel('Centrality')
        if fit:
            ax_base_res.set_xlabel('Centrality')
    else:
        ax_base.set_xlabel('Reference Multiplicity')
        ax_zeros.set_xlabel('Reference Multiplicity')
        if fit:
            ax_base_res.set_xlabel('Reference Multiplicity')
    # ax_base.grid()
    # ax_zeros.grid()
    if title:
        ax_base.set_title(title)
        ax_zeros.set_title(title)
        if fit:
            ax_base_res.set_title(title)

    # Get the handles and labels of all lines
    # handles_base, labels_base = ax_base.get_legend_handles_labels()
    # handles_zero, labels_zero = ax_zeros.get_legend_handles_labels()
    #
    # leg_base, leg_zero = [], []
    # locs_base = {0: 'lower center', 1: 'lower right'}
    # locs_zero = {0: 'upper center', 1: 'upper right'}
    # for i, data_set in enumerate(data_sets_plt):
    #     if len(leg_base) != 0:
    #         ax_base.add_artist(leg_base[-1])
    #     leg_base.append(ax_base.legend(handles_base[i * len(energies):(i + 1) * len(energies)],
    #                               labels_base[i * len(energies):(i + 1) * len(energies)], loc=locs_base[i]))
    #     leg_base[-1].set_title(data_sets_labels[data_set])
    #
    #     if len(leg_zero) != 0:
    #         ax_zeros.add_artist(leg_zero[-1])
    #     leg_zero.append(ax_zeros.legend(handles_zero[i * len(energies):(i + 1) * len(energies)],
    #                               labels_zero[i * len(energies):(i + 1) * len(energies)], loc=locs_zero[i]))
    #     leg_zero[-1].set_title(data_sets_labels[data_set])

    legend_base = ax_base.legend(ncol=len(data_sets_plt) + 1, handletextpad=-1.5, handlelength=0.5)
    legend_zeros = ax_zeros.legend(ncol=len(data_sets_plt) + 1, handletextpad=-1.5, handlelength=0.5)
    # legend_slope.get_frame().set_alpha(0)
    fig_base.tight_layout()
    fig_zeros.tight_layout()

    if fit:
        ax_base_res.set_ylabel(r'Residual of Fit to Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit')
        ax_base_res.grid()
        legend_base_res = ax_base_res.legend()
        fig_base_res.tight_layout()

        fig_consts_vs_energy, ax_consts_vs_energy = plt.subplots()
        ax_consts_vs_energy.set_ylabel('Constant Fit Parameter')
        ax_consts_vs_energy.set_xlabel('Energy (GeV)')

        fig_sqrt_vs_energy, ax_sqrt_vs_energy = plt.subplots()
        ax_sqrt_vs_energy.set_ylabel(r'$1/\sqrt{M}$ Fit Parameter')
        ax_sqrt_vs_energy.set_xlabel('Energy (GeV)')

        for data_set, data in fit_pars.items():
            ax_consts_vs_energy.errorbar(data['energy'], data['const_val'], yerr=data['const_err'], ls='none',
                                         label=data_sets_labels[data_set], marker='o')
            ax_sqrt_vs_energy.errorbar(data['energy'], data['isqrt_val'], yerr=data['isqrt_err'], ls='none',
                                       label=data_sets_labels[data_set], marker='o')
        ax_consts_vs_energy.legend()
        ax_consts_vs_energy.grid()
        ax_consts_vs_energy.set_title('Fit Constants vs Energy')
        fig_consts_vs_energy.canvas.manager.set_window_title('Fit Constants vs Energy')
        fig_consts_vs_energy.tight_layout()

        ax_sqrt_vs_energy.legend()
        ax_sqrt_vs_energy.grid()
        ax_sqrt_vs_energy.set_title(r'$1/\sqrt{M}$ Parameter vs Energy')
        fig_sqrt_vs_energy.canvas.manager.set_window_title('1/sqrt(M) Parameter vs Energy')
        fig_sqrt_vs_energy.tight_layout()

        ax_base.axvline(fit_boundary, ls='--', color='orange')
        ax_base_res.axvline(fit_boundary, ls='--', color='orange')


def plot_div_fits_vs_cent_single_set(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                     cent_ref=None, ref_type=None, ls='none', data_sets_energies_cmaps=None,
                                     data_sets_markers=None, show_xerr=True):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}

    fig_base, ax_base = plt.subplots(figsize=(7, 4.5), dpi=144)
    ax_base.axhline(0, color='black')
    fig_base.canvas.manager.set_window_title(f'Baselines vs Centrality')
    fig_zeros, ax_zeros = plt.subplots(figsize=(7, 4.5), dpi=144)
    fig_zeros.canvas.manager.set_window_title(f'Zeros vs Centrality')
    energies = pd.unique(df['energy'])
    bases, refs, cs = [], [], [0]  # Just to set x and y limits for plot

    for data_set in data_sets_plt:
        df_set = df[df['data_set'] == data_set]
        if data_sets_energies_cmaps is not None and (data_sets_colors is None or len(energies) > 1):
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(1, 0.4, len(energies))])
        elif data_sets_colors is not None:
            if type(data_sets_colors[data_set]) is dict:
                color, colors = None, data_sets_colors[data_set]
            else:
                color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None, None
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                if type(colors) is dict:
                    color = colors[energy]
                else:
                    color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy.sort_values(by='cent')
            if data_sets_labels is None:
                lab = data_set
            else:
                lab = data_sets_labels[data_set]
            if len(energies) > 1:
                lab += f' {energy}GeV'
            if data_sets_markers is None:
                marker = 'o'
            elif data_set in data_sets_markers and 'diff' in data_sets_markers[data_set]:
                marker = data_sets_markers[data_set]['diff']
            elif data_set in data_sets_markers and energy in data_sets_markers[data_set]:
                marker = data_sets_markers[data_set][energy]

            if cent_ref is None:
                x = [cent_map[cent_i] for cent_i in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                if show_xerr:
                    x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                             df_energy['cent']]
                else:
                    x_err = None
            lab = f'{energy} GeV'
            if colors is None and color is None:
                ax_base.errorbar(x, df_energy['baseline'], xerr=x_err, yerr=df_energy['base_err'],
                                 marker=marker, ls=ls, label=lab, alpha=0.6)
                ax_zeros.errobar(x, df_energy['zero_mag'], xerr=x_err, yerr=df_energy['zero_mag_err'],
                                 marker=marker, ls='none', label=lab, alpha=0.6)
                if 'base_sys' in df_energy.columns:
                    ax_base.errorbar(x, df_energy['baseline'], xerr=0, yerr=df_energy['base_sys'],
                                     elinewidth=4, marker='', ls='', alpha=0.3)
                if 'zero_sys' in df_energy.columns:
                    ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=0, yerr=df_energy['zero_sys'],
                                      elinewidth=4, marker='', ls='', alpha=0.3)
            else:
                ax_base.errorbar(x, df_energy['baseline'], xerr=x_err, yerr=df_energy['base_err'],
                                 marker=marker, ls=ls, color=color, label=lab, alpha=0.6)
                ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=x_err, yerr=df_energy['zero_mag_err'],
                                  marker=marker, ls='none', color=color, label=lab, alpha=0.6)
                if 'base_sys' in df_energy.columns:
                    ax_base.errorbar(x, df_energy['baseline'], xerr=0, yerr=df_energy['base_sys'],
                                     elinewidth=4, marker='', ls='', color=color, alpha=0.3)
                if 'zero_sys' in df_energy.columns:
                    ax_zeros.errorbar(x, df_energy['zero_mag'], xerr=0, yerr=df_energy['zero_sys'],
                                      elinewidth=4, marker='', ls='', color=color, alpha=0.3)
            bases.extend(list(df_energy['baseline']))
            refs.extend(list(x))
    ax_base.set_ylim(min(bases) * 1.1, max(cs) * 1.1)
    try:
        ax_base.set_xlim(0, max(refs) * 1.1)
    except TypeError:
        pass

    ax_base.set_ylabel(r'$b$', fontsize='x-large')
    ax_zeros.set_ylabel(r'$z$', fontsize='x-large')
    ax_base.legend()
    ax_zeros.legend()
    if ref_type is None:
        ax_base.set_xlabel('Centrality', fontsize='x-large')
        ax_zeros.set_xlabel('Centrality', fontsize='x-large')
    else:
        ax_base.set_xlabel('Reference Multiplicity', fontsize='x-large')
        ax_zeros.set_xlabel('Reference Multiplicity', fontsize='x-large')
    if title:
        ax_base.set_title(title)
        ax_zeros.set_title(title)

    fig_base.tight_layout()
    fig_zeros.tight_layout()

    fig_base.subplots_adjust(top=0.985, right=0.995, bottom=0.110, left=0.14)
    fig_zeros.subplots_adjust(top=0.995, right=0.995, bottom=0.110, left=0.085)


def plot_div_fits_vs_cent_62res(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None,
                                name_col='data_set', val_col='baseline', err_col='base_err'):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    fig_base, ax_base = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax_base.axhline(0, color='black')
    fig_base.canvas.manager.set_window_title(f'Baseline with 62 GeV Fit vs Centrality')
    fig_dev, ax_dev = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax_dev.axhline(0, color='black')
    fig_dev.canvas.manager.set_window_title(f'Deviation From 62 GeV Fit vs Centrality')

    energies = pd.unique(df['energy'])
    for data_set in data_sets_plt:
        df_set = df[df[name_col] == data_set]

        df_62 = df_set[df_set['energy'] == 62]
        if cent_ref is None:
            x = [cent_map[cent_i] for cent_i in df_62['cent']]
            x_err = None
        else:
            cent_energy = cent_ref[(cent_ref[name_col] == data_set) & (cent_ref['energy'] == 62)]
            x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                 df_62['cent']]
            x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                     df_62['cent']]

        # p0 = [-0.02, 0.01]
        # # func = inv_sqrtx_odr
        # odr_model = odr.Model(func)
        # odr_data = odr.RealData(x, df_62[val_col], sx=x_err, sy=df_62[err_col])
        # fit_odr = odr.ODR(odr_data, odr_model, beta0=p0, maxit=500)
        # odr_out = fit_odr.run()
        # fit_meases = [Measure(var, err) for var, err in zip(odr_out.beta, odr_out.sd_beta)]
        # print(f'{data_set} 62GeV Fit: {fit_meases}')

        func = inv_pow_x_const
        popt, pcov = cf(func, x, df_62[val_col], sigma=df_62[err_col], absolute_sigma=True)
        fit_meases = [Measure(var, err) for var, err in zip(popt, np.sqrt(np.diag(pcov)))]
        # print(f'{data_set} 62GeV Fit: {fit_meases}')

        x_fit_plt = np.linspace(1, 800, 2000)
        # ax_base.plot(x_fit_plt, inv_sqrtx(x_fit_plt, *odr_out.beta), alpha=0.6, color='black')
        ax_base.plot(x_fit_plt, func(x_fit_plt, *popt), alpha=0.6, color='black')

        if data_sets_energies_cmaps is not None:
            colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                           for i in np.linspace(1, 0.4, len(energies))])
        elif data_sets_colors is not None:
            color, colors = data_sets_colors[data_set], None
        else:
            colors, color = None
        # print(pd.unique(df_set['energy']))
        for energy in pd.unique(df_set['energy']):
            if colors is not None:
                color = next(colors)
            df_energy = df_set[df_set['energy'] == energy]
            df_energy.sort_values(by='cent')
            if data_sets_labels is None:
                lab = data_set
            else:
                lab = data_sets_labels[data_set]
            if len(energies) > 1:
                lab += f' {energy}GeV'

            if cent_ref is None:
                x = [cent_map[cent_i] for cent_i in df_energy['cent']]
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref[name_col] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]

            y = [Measure(base, base_err) for base, base_err in zip(df_energy[val_col], df_energy[err_col])]
            # y_fit = [func(fit_meases, xi) for xi in x]
            y_fit = [func(xi, *popt) for xi in x]
            # y_res = (np.array(y_fit) - odr_out.beta[-1]) / (np.array(y) - odr_out.beta[-1])
            # y_res = np.array(y_fit) / np.array(y)
            y_res = np.array(y) - np.array(y_fit)

            ls = 'none' if fit else '-'
            if colors is None and color is None:
                ax_dev.errorbar(x, [z.val for z in y_res], xerr=x_err, yerr=[z.err for z in y_res], marker='o',
                                ls=ls, label=lab, alpha=0.6)
                ax_base.errorbar(x, df_energy[val_col], xerr=x_err, yerr=df_energy[err_col], marker='o',
                                 ls=ls, label=lab, alpha=0.6)
            else:
                ax_dev.errorbar(x, [z.val for z in y_res], xerr=x_err, yerr=[z.err for z in y_res], marker='o',
                                ls=ls, color=color, label=lab, alpha=0.6)
                ax_base.errorbar(x, df_energy[val_col], xerr=x_err, yerr=df_energy[err_col], marker='o',
                                 ls=ls, color=color, label=lab, alpha=0.6)

    ax_base.set_ylim([-0.007, 0.001])
    ax_dev.set_ylabel(r'Deviation of Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width from 62GeV Fit')
    ax_base.set_ylabel(r'Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit')
    if ref_type is None:
        ax_dev.set_xlabel('Centrality')
        ax_base.set_xlabel('Centrality')
    else:
        ax_dev.set_xlabel('Reference Multiplicity')
        ax_base.set_xlabel('Reference Multiplicity')
    ax_dev.grid()
    ax_base.grid()
    if title:
        ax_dev.set_title(title)
        ax_base.set_title(title)
    legend_dev = ax_dev.legend()
    legend_base = ax_base.legend()

    # legend_slope.get_frame().set_alpha(0)
    fig_dev.tight_layout()
    fig_base.tight_layout()


def plot_div_fits_vs_cent_2panel(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                 fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                 alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                 data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                 legend_order=None, print_data=False):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}

    fit_pars = {'baseline': {'window_title': 'Baselines vs Centrality', 'val': 'baseline', 'err': 'base_err',
                             'sys': 'base_sys',
                             'ylabel': r'Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit'},
                'zeros': {'window_title': 'Zeros vs Centrality', 'val': 'zero_mag', 'err': 'zero_mag_err',
                          'sys': 'zero_sys',
                          'ylabel': r'Zeros of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit'}}

    for fit_par_name, fit_par in fit_pars.items():
        fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
        if fit_par_name == 'baseline':
            for ax_avg in ax_avgs:
                ax_avg.axhline(0, color='black')
        ax_avgs = dict(zip(data_sets_plt, ax_avgs))
        fig_avg.canvas.manager.set_window_title(fit_par['window_title'])
        energies = pd.unique(df['energy'])
        for data_set in data_sets_plt:
            ax_avg = ax_avgs[data_set]
            df_set = df[df['data_set'] == data_set]
            if data_sets_energies_cmaps is not None:
                colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                               for i in np.linspace(0.1, 0.9, len(energies))])
            elif data_sets_energies_colors is not None:
                colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
            elif data_sets_colors is not None:
                color, colors = data_sets_colors[data_set], None
            else:
                colors, color = None
            if marker_map is None:
                marker = 'o'
            else:
                marker = marker_map[data_set]['diff']
            for energy in pd.unique(df_set['energy']):
                if colors is not None:
                    color = next(colors)
                df_energy = df_set[df_set['energy'] == energy]
                df_energy = df_energy.sort_values(by='cent')
                if data_sets_labels is not None:
                    lab = data_sets_labels[data_set]
                else:
                    lab = ''
                if len(energies) > 1:
                    lab += f'{energy_map[energy]} GeV'

                if cent_ref is None:
                    x = df_energy['cent']
                    # x = [cent_map[xi] for xi in df_energy['cent']]
                    x_err = None
                else:
                    cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                    x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                         df_energy['cent']]
                    x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                             df_energy['cent']]
                if colors is None and color is None:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax_avg.fill_between(x, df_energy[fit_par['val']] - df_energy[fit_par['err']],
                                            df_energy[fit_par['val']] + df_energy[fit_par['err']], label=lab, alpha=0.4)
                    else:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=x_err, yerr=df_energy[fit_par['err']],
                                        marker=marker, ls=ls,
                                        label=lab, alpha=alpha)
                    if fit_par['sys'] in df_energy.columns:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=0, yerr=df_energy[fit_par['sys']], marker='',
                                        ls='',
                                        elinewidth=4, alpha=errbar_alpha)
                else:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax_avg.fill_between(x, df_energy[fit_par['val']] - df_energy[fit_par['err']],
                                            df_energy[fit_par['val']] + df_energy[fit_par['err']], label=lab, alpha=0.4,
                                            color=color)
                    else:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=x_err, yerr=df_energy[fit_par['err']],
                                        marker=marker, ls=ls,
                                        color=color, label=lab, alpha=alpha)
                    if fit_par['sys'] in df_energy.columns:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=0, yerr=df_energy[fit_par['sys']], marker='',
                                        ls='',
                                        color=color, elinewidth=4, alpha=errbar_alpha)
                if print_data:
                    print(f'\n{data_set} {energy}GeV {fit_par_name}')
                    print(f'\'x\': [{", ".join([str(xi) for xi in x])}]')
                    print(f'\'val\': [{", ".join([str(xi) for xi in df_energy[fit_par["val"]]])}]')
                    print(f'\'err\': [{", ".join([str(xi) for xi in df_energy[fit_par["err"]]])}]')
                    if fit_par['sys'] in df_energy.columns:
                        print(f'\'sys\': [{", ".join([str(xi) for xi in df_energy[fit_par["sys"]]])}]')


        for set_i, data_set in enumerate(data_sets_plt):
            ax_avg = ax_avgs[data_set]
            if set_i == 0:
                ax_avg.set_ylabel(fit_par['ylabel'])
            if set_i == 1:
                ax_avg.tick_params(axis='y', which='both', left=False, right=False)
            if ref_type is None:
                ax_avg.set_xlabel('Centrality')
            elif ref_type == 'npart':
                ax_avg.set_xlabel('Number of Participant Nucleons')
            else:
                ax_avg.set_xlabel('Reference Multiplicity')
            if xlim is not None:
                ax_avg.set_xlim(xlim)
            ax_avg.set_axisbelow(True)
            ax_avg.grid()
            if kin_info_loc is not None and set_i == 0:
                eta_line = r'|y| < 0.5'
                pt_line = r'0.4 < $p_T$ < 2.0 GeV'
                ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n72 Samples/Event', ha='left', va='bottom',
                            transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
            if star_prelim_loc is not None and data_set == 'bes_def':
                ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left',
                            transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9),
                            va='top')
            if legend_order is not None:
                handles, labels = ax_avg.get_legend_handles_labels()
                handles_dict = dict(zip(labels, handles))
                ordered_handles = [handles_dict[label] for label in legend_order]
                legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order)
            else:
                legend_avg = ax_avg.legend()
            # legend_avg = ax_avg.legend()
            legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
            legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
            # legend_avg.get_frame().set_alpha(0)
        if title and title != '':
            fig_avg.suptitle(title)
        fig_avg.tight_layout()
        fig_avg.subplots_adjust(top=0.985, right=0.995, bottom=0.098, left=0.052, wspace=0)


def plot_baseline_vs_cent_fit(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None,
                                 fit=False, cent_ref=None, ref_type=None, data_sets_energies_cmaps=None, ls='-',
                                 alpha=0.6, errbar_alpha=0.2, kin_info_loc=(0.18, 0.1), star_prelim_loc=None,
                                 data_sets_energies_colors=None, marker_map=None, data_sets_bands=None, xlim=None,
                                 legend_order=None, plot=True):
    cent_map = {8: '0-5%', 7: '5-10%', 6: '10-20%', 5: '20-30%', 4: '30-40%', 3: '40-50%', 2: '50-60%', 1: '60-70%',
                0: '70-80%', -1: '80-90%'}
    energy_map = {7: '7.7', 11: '11.5', 19: '19.6', 27: '27', 39: '39', 62: '62.4'}

    fit_par = {'window_title': 'Baselines vs Centrality', 'val': 'baseline', 'err': 'base_err', 'sys': 'base_sys',
               'ylabel': r'Baseline of $\langle\Delta\sigma^2\rangle$ vs Partition Width Fit'}

    if plot:
        fig_avg, ax_avgs = plt.subplots(figsize=(13.33, 5.1), dpi=144, ncols=len(data_sets_plt), sharey='all')
        for ax_avg in ax_avgs:
            ax_avg.axhline(0, color='black')
        ax_avgs = dict(zip(data_sets_plt, ax_avgs))
        fig_avg.canvas.manager.set_window_title(fit_par['window_title'])
        fig_fit_params, ax_fit_params = plt.subplots(figsize=(10, 6), dpi=144, nrows=3, sharex='all')
        fig_fit_params.canvas.manager.set_window_title('Baseline vs Centrality Fit Parameters')
    energies = pd.unique(df['energy'])
    all_fit_params = []
    for data_set in data_sets_plt:
        df_set = df[df['data_set'] == data_set]
        if plot:
            ax_avg = ax_avgs[data_set]
            if data_sets_energies_cmaps is not None:
                colors = iter([plt.cm.get_cmap(data_sets_energies_cmaps[data_set])(i)
                               for i in np.linspace(0.1, 0.9, len(energies))])
            elif data_sets_energies_colors is not None:
                colors = iter([data_sets_energies_colors[data_set][energy] for energy in pd.unique(df_set['energy'])])
            elif data_sets_colors is not None:
                color, colors = data_sets_colors[data_set], None
            else:
                colors, color = None
            if marker_map is None:
                marker = 'o'
            else:
                marker = marker_map[data_set]['diff']

        # Fit energies and plot
        fit_params = {'data_set': [], 'energy': [], 'a': [], 'a_err': [], 'b': [], 'b_err': [], 'c': [],
                      'c_err': []}
        for energy in pd.unique(df_set['energy']):
            df_energy = df_set[df_set['energy'] == energy]
            df_energy = df_energy.sort_values(by='cent')
            if plot:
                if colors is not None:
                    color = next(colors)
                if data_sets_labels is not None:
                    lab = data_sets_labels[data_set]
                else:
                    lab = ''
                if len(energies) > 1:
                    lab += f'{energy_map[energy]} GeV'

            if cent_ref is None:
                x = df_energy['cent']
                x_err = None
            else:
                cent_energy = cent_ref[(cent_ref['data_set'] == data_set) & (cent_ref['energy'] == energy)]
                x = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_val'].iloc[0] for cent in
                     df_energy['cent']]
                x_err = [cent_energy[cent_energy['cent'] == cent][f'mean_{ref_type}_sd'].iloc[0] for cent in
                         df_energy['cent']]
            x = np.array(x)

            p0 = (0.0002, -100, 0.5)
            popt, pcov = cf(inv_pow_m, x, df_energy[fit_par['val']], p0=p0, sigma=df_energy[fit_par['err']],
                            absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            # pmeas = [Measure(var, err) for var, err in zip(popt, perr)]
            fit_params['data_set'].append(data_set)
            fit_params['energy'].append(energy)
            fit_params['a'].append(popt[0])
            fit_params['a_err'].append(perr[0])
            fit_params['b'].append(popt[1])
            fit_params['b_err'].append(perr[1])
            fit_params['c'].append(popt[2])
            fit_params['c_err'].append(perr[2])
            x_fit_plt = np.linspace(x.min(), x.max(), 1000)

            if plot:
                if colors is None and color is None:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax_avg.fill_between(x, df_energy[fit_par['val']] - df_energy[fit_par['err']],
                                            df_energy[fit_par['val']] + df_energy[fit_par['err']], label=lab, alpha=0.4)
                    else:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=x_err, yerr=df_energy[fit_par['err']],
                                        marker=marker, ls=ls,
                                        label=lab, alpha=alpha)
                    if fit_par['sys'] in df_energy.columns:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=0, yerr=df_energy[fit_par['sys']], marker='',
                                        ls='',
                                        elinewidth=4, alpha=errbar_alpha)
                else:
                    if data_sets_bands is not None and data_set in data_sets_bands:
                        ax_avg.fill_between(x, df_energy[fit_par['val']] - df_energy[fit_par['err']],
                                            df_energy[fit_par['val']] + df_energy[fit_par['err']], label=lab, alpha=0.4,
                                            color=color)
                    else:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=x_err, yerr=df_energy[fit_par['err']],
                                        marker=marker, ls=ls,
                                        color=color, label=lab, alpha=alpha)
                    if fit_par['sys'] in df_energy.columns:
                        ax_avg.errorbar(x, df_energy[fit_par['val']], xerr=0, yerr=df_energy[fit_par['sys']], marker='',
                                        ls='',
                                        color=color, elinewidth=4, alpha=errbar_alpha)
                ax_avg.plot(x_fit_plt, inv_pow_m(x_fit_plt, *popt), alpha=0.6, color=color)
                ax_avg.axhline(popt[0], color=color, ls='--')
                ax_avg.axhspan(popt[0] - perr[0], popt[0] + perr[0], color=color, alpha=0.2)
                print(f'{lab} Fit: {[Measure(val, err) for val, err in zip(popt, perr)]}')

        if plot:
            ax_fit_params[0].errorbar(fit_params['energy'], fit_params['a'], yerr=fit_params['a_err'],
                                      marker=marker, ls='', label=data_set)
            ax_fit_params[1].errorbar(fit_params['energy'], fit_params['b'], yerr=fit_params['b_err'],
                                      marker=marker, ls='', label=data_set)
            ax_fit_params[2].errorbar(fit_params['energy'], fit_params['c'], yerr=fit_params['c_err'],
                                      marker=marker, ls='', label=data_set)
            ax_fit_params[0].set_ylabel('a')
            ax_fit_params[1].set_ylabel('b')
            ax_fit_params[2].set_ylabel('c')
            ax_fit_params[2].set_xlabel('Energy (GeV)')
            ax_fit_params[0].legend()
            ax_fit_params[0].axhline(0, color='black', alpha=0.5, zorder=0)
            fit_str = r'$a + \frac{b}{x^c}$'
            ax_fit_params[2].annotate(fit_str, (0.7, 0.2), xycoords='axes fraction', ha='center', va='center', fontsize=16)
            fig_fit_params.subplots_adjust(bottom=0.075, top=0.995, right=0.995, left=0.085, hspace=0.05)

        # Convert fit_params from dict of lists to list of dicts and extend all_fit_params
        fit_params = [{key: fit_params[key][i] for key in fit_params} for i in
                      range(len(fit_params['data_set']))]
        all_fit_params.extend(fit_params)

    if plot:
        for set_i, data_set in enumerate(data_sets_plt):
            ax_avg = ax_avgs[data_set]
            if set_i == 0:
                ax_avg.set_ylabel(fit_par['ylabel'])
            if set_i == 1:
                ax_avg.tick_params(axis='y', which='both', left=False, right=False)
            if ref_type is None:
                ax_avg.set_xlabel('Centrality')
            elif ref_type == 'npart':
                ax_avg.set_xlabel('Number of Participant Nucleons')
            else:
                ax_avg.set_xlabel('Reference Multiplicity')
            if xlim is not None:
                ax_avg.set_xlim(xlim)
            ax_avg.set_axisbelow(True)
            ax_avg.grid()
            if kin_info_loc is not None and set_i == 0:
                eta_line = r'|y| < 0.5'
                pt_line = r'0.4 < $p_T$ < 2.0 GeV'
                ax_avg.text(*kin_info_loc, f'Au+Au\n{eta_line}\n{pt_line}\n72 Samples/Event', ha='left', va='bottom',
                            transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9))
            if star_prelim_loc is not None and data_set == 'bes_def':
                ax_avg.text(*star_prelim_loc, 'STAR Preliminary', fontsize='large', ha='left',
                            transform=ax_avg.transAxes,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9),
                            va='top')
            if legend_order is not None:
                handles, labels = ax_avg.get_legend_handles_labels()
                handles_dict = dict(zip(labels, handles))
                ordered_handles = [handles_dict[label] for label in legend_order]
                legend_avg = ax_avg.legend(handles=ordered_handles, labels=legend_order)
            else:
                legend_avg = ax_avg.legend()
            # legend_avg = ax_avg.legend()
            legend_title = 'STAR' if 'bes' in data_set else 'AMPT'
            legend_avg.set_title(legend_title, prop={'size': 'large', 'weight': 'bold'})
            # legend_avg.get_frame().set_alpha(0)
        if title and title != '':
            fig_avg.suptitle(title)
        fig_avg.tight_layout()
        fig_avg.subplots_adjust(top=0.985, right=0.995, bottom=0.098, left=0.052, wspace=0)

    return pd.DataFrame(all_fit_params)


def plot_protons_fits_vs_amp(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None):
    fig_slope, ax_slope = plt.subplots()
    ax_slope.axhline(0, color='gray')
    fig_slope.canvas.manager.set_window_title(f'Slopes vs Simulation Amplitude')
    fig_int, ax_int = plt.subplots()
    ax_int.axhline(1, color='gray')
    fig_int.canvas.manager.set_window_title(f'Intercepts vs Simulation Amplitude')
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='amp')
        if data_sets_labels is None:
            lab = data_set
        else:
            lab = data_sets_labels[data_set]
        if data_sets_colors is None:
            ax_slope.errorbar(df_set['amp'], df_set['slope'], yerr=df_set['slope_err'], ls='none', marker='o',
                              label=lab)
            ax_int.errorbar(df_set['amp'], df_set['int'], yerr=df_set['int_err'], ls='none', marker='o', label=lab)
        else:
            ax_slope.errorbar(df_set['amp'], df_set['slope'], yerr=df_set['slope_err'], ls='none', marker='o',
                              color=data_sets_colors[data_set], label=lab)
            ax_int.errorbar(df_set['amp'], df_set['int'], yerr=df_set['int_err'], ls='none', marker='o',
                            color=data_sets_colors[data_set], label=lab)
    ax_slope.set_ylabel('Slope')
    ax_slope.set_xlabel('Simulation Amplitude')
    ax_slope.grid()
    ax_int.set_ylabel('Intercept')
    ax_int.set_xlabel('Simulation Amplitude')
    ax_int.grid()
    legend_slope = ax_slope.legend()
    # legend_slope.get_frame().set_alpha(0)
    legend_int = ax_int.legend()
    # legend_int.get_frame().set_alpha(0)
    fig_slope.tight_layout()
    fig_int.tight_layout()

    fig_slope_fit, ax_slope_fit = plt.subplots(figsize=(6.66, 5), dpi=144)
    fig_slope_fit.canvas.manager.set_window_title(f'Slopes vs Simulation Amplitude Fits')
    ax_slope_fit.axhline(0, color='black')
    colors = iter(['black', 'blue', 'green', 'red', 'purple', 'salmon'])
    cl_type_name = {'_clmul_': 'Attractive', '_aclmul_': 'Repulsive'}
    spreads = pd.unique(df['spread'])

    for cl_type in cl_type_name.keys():
        spreads = sorted(spreads) if cl_type == '_aclmul_' else sorted(spreads, reverse=True)
        for spread in spreads:
            color = next(colors)
            df_spread = df[(df['spread'] == spread) & (df['name'].str.contains(cl_type))]
            print(f'{cl_type}, spread {spread}, size: {df_spread.size}')
            ax_slope_fit.errorbar(df_spread['amp'], df_spread['slope'], df_spread['slope_err'], ls='none', marker='o',
                                  color=color, label=f'{cl_type_name[cl_type]} σ={spread}')
            popt, pcov = cf(line, df_spread['amp'], df_spread['slope'], sigma=df_spread['slope_err'],
                            absolute_sigma=True)
            ax_slope_fit.plot(df_spread['amp'], line(df_spread['amp'], *popt), ls='--', color=color)
    ax_slope_fit.set_xlabel('Simulation Amplitude')
    ax_slope_fit.set_ylabel('Slope of Raw/Mix SD vs Total Protons per Event')
    ax_slope_fit.legend()
    ax_slope_fit.grid()
    fig_slope_fit.tight_layout()


def plot_dsig_fits_vs_amp(df, data_sets_plt, data_sets_colors=None, data_sets_labels=None, title=None):
    fig_slope, ax_slope = plt.subplots()
    ax_slope.axhline(0, color='gray')
    fig_slope.canvas.manager.set_window_title(f'Dsigma vs Simulation Amplitude')
    for data_set in data_sets_plt:
        df_set = df[df['name'] == data_set]
        df_set.sort_values(by='amp')
        if data_sets_labels is None:
            lab = data_set
        else:
            lab = data_sets_labels[data_set]
        if data_sets_colors is None:
            ax_slope.errorbar(df_set['amp'], df_set['avg'], yerr=df_set['avg_err'], ls='none', marker='o',
                              label=lab)
        else:
            ax_slope.errorbar(df_set['amp'], df_set['avg'], yerr=df_set['avg_err'], ls='none', marker='o',
                              color=data_sets_colors[data_set], label=lab)
    ax_slope.set_ylabel('Avg')
    ax_slope.set_xlabel('Simulation Amplitude')
    ax_slope.grid()
    legend_slope = ax_slope.legend()
    # legend_slope.get_frame().set_alpha(0)
    fig_slope.tight_layout()

    fig_slope_fit, ax_slope_fit = plt.subplots()
    fig_slope_fit.canvas.manager.set_window_title(f'Dsigma Avgs vs Simulation Amplitude')
    ax_slope_fit.axhline(0, color='black')
    colors = iter(['black', 'blue', 'green', 'red', 'purple', 'salmon', 'cyan', 'magenta', 'yellow', 'orange', 'pink',
                   'brown', 'gray', 'olive', 'teal', 'lavender', 'violet', 'lime', 'turquoise', 'maroon'])
    cl_type_name = {'_clmul_': 'Attractive', '_aclmul_': 'Repulsive'}
    spreads = pd.unique(df['spread'])

    for cl_type in cl_type_name.keys():
        spreads = sorted(spreads) if cl_type == '_aclmul_' else sorted(spreads, reverse=True)
        for spread in spreads:
            color = next(colors)
            df_spread = df[(df['spread'] == spread) & (df['name'].str.contains(cl_type))]
            print(f'{cl_type}, spread {spread}, size: {df_spread.size}')
            if df_spread.size > 0:
                ax_slope_fit.errorbar(df_spread['amp'], df_spread['avg'], df_spread['avg_err'], ls='none', marker='o',
                                      color=color, label=f'{cl_type_name[cl_type]} σ={spread}')
                popt, pcov = cf(line, df_spread['amp'], df_spread['avg'], sigma=df_spread['avg_err'],
                                absolute_sigma=True)
                x = np.array([0] + list(df_spread['amp']))
                ax_slope_fit.plot(x, line(x, *popt), ls='--', color=color)
                # ax_slope_fit.plot(df_spread['amp'], line(df_spread['amp'], *popt), ls='--', color=color)
    ax_slope_fit.set_xlabel(r'Simulation Amplitude ($A$)')
    ax_slope_fit.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
    if title is not None:
        ax_slope_fit.set_title(title)
    ax_slope_fit.legend()
    ax_slope_fit.grid()
    fig_slope_fit.tight_layout()


def chi2_vs_protons(df, stat, div, cent, energy, data_type, data_sets_plt):
    chi2_sets = []
    df = df[(df['divs'] == div) & ((df['energy'] == energy) | (df['energy'] == 'sim'))]
    if 'data_type' in df:
        df = df[df['data_type'] == data_type]
    if 'cent' in df:
        df = df[df['cent'] == cent]
    if 'stat' in df:
        df = df[df['stat'] == stat]

    df_data = df[~df['name'].str.contains('sim_')]
    df_sim = df[df['name'].str.contains('sim_')]

    data_sets = np.unique(df_data['name'])
    sim_sets = np.unique(df_sim['name'])
    for data_set in data_sets:
        if data_set not in data_sets_plt:
            continue
        df_data_set = df_data[df_data['name'] == data_set]
        df_data_set.sort_values(by=['total_protons'])
        data_vals, data_errs = np.array(df_data_set['val']), np.array(df_data_set['err'])

        for sim_set in sim_sets:
            df_sim_set = df_sim[(df_sim['name'] == sim_set)]
            # Sim should have all possible tproton values, so just filter out those that aren't in data
            df_sim_set = df_sim_set[df_sim_set['total_protons'].isin(df_data_set['total_protons'])]
            df_sim_set.sort_values(by=['total_protons'])
            if list(df_data_set['total_protons']) != list(df_sim_set['total_protons']):
                print('Data and sim don\'t match total protons!')

            sim_vals, sim_errs = np.array(df_sim_set['val']), np.array(df_sim_set['err'])
            chi2 = np.sum((data_vals - sim_vals) ** 2 / (data_errs ** 2 + sim_errs ** 2))

            chi2_sets.append({'data_name': data_set, 'sim_name': sim_set, 'chi2': chi2, 'n': len(data_vals),
                              'divs': div, 'amp': df_sim_set['amp'].iloc[0], 'spread': df_sim_set['spread'].iloc[0],
                              'energy': energy, 'data_type': data_type})

    return chi2_sets


def plot_chi2_protons(df, energy, n_sims_plt=6):
    sim_sets_plt = {}
    chi_list = []
    for data_set in np.unique(df['data_name']):
        chi2_sums = []
        df_data = df[df['data_name'] == data_set]
        for sim_set in np.unique(df_data['sim_name']):
            df_sim = df_data[df_data['sim_name'] == sim_set]
            chi2_sum = df_sim['chi2'].sum()
            chi2_sums.append({'sim_name': sim_set, 'chi2_sum': chi2_sum, 'amp': df_sim['amp'].iloc[0],
                              'spread': df_sim['spread'].iloc[0], 'data_set': data_set, 'energy': energy})
        chi_list.extend(chi2_sums)
        chi2_sums = pd.DataFrame(chi2_sums)
        chi2_sums = chi2_sums.sort_values(by='chi2_sum').head(n_sims_plt)
        sim_sets_plt.update({data_set: chi2_sums['sim_name']})

    chi_res = []
    for data_set in np.unique(df['data_name']):
        fig, ax = plt.subplots()
        ax.set_title(data_set)
        ax.set_ylabel('Chi^2')
        ax.set_xlabel('Bin Width')
        df_data = df[df['data_name'] == data_set]
        for sim_set in sim_sets_plt[data_set]:
            df_sim = df_data[df_data['sim_name'] == sim_set].sort_values(by='divs')
            ax.axhline(0, ls='--', color='black')
            ax.plot(df_sim['divs'], df_sim['chi2'], marker='o', alpha=0.8, label=f'{sim_set}')
        ax.legend()
        fig.tight_layout()

    return sim_sets_plt, chi_res, chi_list


def plot_chi_sim_pars(chi_res):
    fig, ax = plt.subplots()
    ax.set_xlabel('Amp')
    ax.set_ylabel('Spread')
    for data_set in np.unique(chi_res['data_set']):
        df_set = chi_res[chi_res['data_set'] == data_set]
        ax.scatter(df_set['amp'], df_set['spread'], label=data_set)
    for x, y, e in zip(chi_res['amp'], chi_res['spread'], chi_res['energy']):
        ax.annotate(f'{e}GeV', (x, y), textcoords='offset points', xytext=(0, 10), ha='center')
    ax.legend()
    ax.grid()
    fig.tight_layout()


def stat_vs_divs(df, stat, total_protons, cent, energies, data_types, data_sets_plt, exclude_divs=[], y_ranges=None,
                 plot=False, fit=False):
    data = []
    for data_type in data_types:
        for data_set in data_sets_plt:
            for energy in energies:
                df_pre = df
                if 'data_type' in df_pre:
                    df_pre = df_pre[df_pre['data_type'] == data_type]
                if 'cent' in df_pre:
                    df_pre = df_pre[df_pre['cent'] == cent]
                if 'stat' in df_pre:
                    df_pre = df_pre[df_pre['stat'] == stat]
                df_set = df_pre[
                    (df_pre['name'] == data_set) & (df_pre['total_protons'] == total_protons) & (
                            df_pre['energy'] == energy)]
                df_set = df_set.sort_values(by=['divs'])
                if len(df_set) == 0:
                    continue
                if energy == 'sim':
                    lab = f'{data_set}_{data_type}'
                else:
                    lab = f'{data_set}_{data_type}_{energy}GeV'
                df_set = df_set.sort_values(by=['total_protons'])
                data.append((df_set, lab, data_set, energy))

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f'{stat} {total_protons} protons')
        ax.set_xlabel('Division Width')
        ax.axhline(1, ls='--', color='black')
        if y_ranges:
            ax.set_ylim(y_ranges[stat])
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))

    fit_pars = []
    for df, lab, data_set, energy in data:
        c = next(color)
        if plot:
            ax.errorbar(df['divs'], df['val'], df['err'], label=lab, marker='o', ls='', color=c)
            if 'sys' in df:
                ax.errorbar(df['divs'], df['val'], df['sys'], marker='', ls='', elinewidth=3, color=c, alpha=0.3)

        if fit and df.size > 1:
            df = df[~df.divs.isin(exclude_divs)]
            popt, pcov = cf(quad_180, df['divs'], df['val'], sigma=df['err'], absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            fit_pars.append({'data_set': data_set, 'energy': energy, 'curvature': popt[0], 'baseline': popt[1],
                             'spread': df['spread'].iloc[0], 'amp': df['amp'].iloc[0],
                             'curve_err': perr[0], 'base_err': perr[1], 'color': c})
            x = np.linspace(0, 360, 100)
            if plot:
                ax.plot(x, quad_180(x, *popt), ls='--', color=c)

    if plot:
        ax.legend()
        fig.tight_layout()

    return pd.DataFrame(fit_pars)


def interp_vs_div(df_pars, stat, quad_par_line=line_yint1, r_amp_line=line_yint0, plot=False, fit=False):
    base_curve_lin_fits = []
    spread_rs_amp = []

    df_data = df_pars[~df_pars['data_set'].str.contains('sim_')]
    df_sim = df_pars[df_pars['data_set'].str.contains('sim_')]
    spreads = np.unique(df_sim['spread'])
    data_sets = np.unique(df_data['data_set'])

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(stat)
        ax.set_xlabel('baseline')
        ax.set_ylabel('curvature')
        fix_dev, ax_dev = plt.subplots()
        ax_dev.set_title(f'{stat} Deviation from linear')
        ax_dev.set_xlabel('baseline')
        ax_dev.set_ylabel('curvature deviation from fit')
        ax_dev.axhline(0, ls='--')
        fig_rs, ax_rs = plt.subplots()
        ax_rs.set_title('Rs vs Amp')
        ax_rs.set_xlabel('amp')
        ax_rs.set_ylabel('r')
        fig_ramp_slope, ax_ramp_slope = plt.subplots()
        ax_ramp_slope.set_title('Rs vs Amp Slopes')
        ax_ramp_slope.set_xlabel('Spread')
        ax_ramp_slope.set_ylabel('r_amp slope')
        # color = iter(plt.cm.rainbow(np.linspace(0, 1, len(spreads) + len(data_sets))))
    spreads_list = []
    ramp_slopes = []
    ramp_slope_errs = []
    all_rs = []
    if plot:
        for data_set in data_sets:
            df_set = df_data[df_data['data_set'] == data_set]
            # c = next(color)
            ax.errorbar(df_set['baseline'], df_set['curvature'], xerr=df_set['base_err'], yerr=df_set['curve_err'],
                        marker='o', color=df_set['color'], label=data_set, ls='none', alpha=0.6)
            for x, y, e in zip(df_data['baseline'], df_data['curvature'], df_data['energy']):
                ax.annotate(f'{e}GeV', (x, y), textcoords='offset points', xytext=(0, 10), ha='center')

    for spread in spreads:
        # if plot:
        #     c = next(color)
        df_spread = df_sim[df_sim['spread'] == spread]
        rs = np.sqrt(np.power(df_spread['curvature'], 2) + np.power((df_spread['baseline'] - 1), 2))
        all_rs += [r for r in rs]
        if len(df_spread) > 1:
            popt, pcov = cf(quad_par_line, df_spread['baseline'], df_spread['curvature'], sigma=df_spread['curve_err'],
                            absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            if type(spread) == str:
                spread_float = float(f'0.{spread.strip("spread")}') * 10
            else:
                spread_float = float(spread)
            # New column of df equal to sqrt(curvature**2 + (baseline - 1)**2)
            if len(popt) == 1:  # Ideally just append this to original df
                base_curve_lin_fits.append({'spread': spread_float, 'slope': popt[0], 'slope_err': perr[0]})
            elif len(popt) == 2:
                base_curve_lin_fits.append({'spread': spread_float, 'slope': popt[0], 'slope_err': perr[0],
                                            'int': popt[1],
                                            'int_err': perr[1]})  # Ideally just append this to original df
            amp_floats = [float(f'0.{amp.strip("amp")}') if type(amp) == str else float(amp)
                          for amp in df_spread['amp']]
            popt1, pcov1 = cf(r_amp_line, amp_floats, rs)
            perr1 = np.sqrt(np.diag(pcov1))
            spreads_list.append(spread_float)
            ramp_slopes.append(popt1[0])
            ramp_slope_errs.append(perr1[0])
            if plot:
                x_plt = np.linspace(min(df_sim['baseline']), 1, 3)
                ax.plot(x_plt, quad_par_line(x_plt, *popt), ls='--', alpha=0.7, color=df_spread['color'])
                ax_dev.scatter(df_spread['baseline'],
                               df_spread['curvature'] - line_yint1(df_spread['baseline'], *popt),
                               color=df_spread['color'])
                x_plt1 = np.array([0, 0.15])
                ax_rs.plot(x_plt1, r_amp_line(x_plt1, *popt1), ls='--', alpha=0.8, color=df_spread['color'])
                ax_rs.scatter(amp_floats, rs, color=df_spread['color'], marker='o', alpha=0.6, label=spread)
        if plot:
            ax.errorbar(df_spread['baseline'], df_spread['curvature'], xerr=df_spread['base_err'], marker='o',
                        yerr=df_spread['curve_err'], color=df_spread['color'], label=spread, ls='none', alpha=0.6)
            # if spread == 'ampt':
            #     # print(rs)
            #     ax_rs.axhline(rs[0], color=c, ls=':', label=spread, alpha=0.7, zorder=0)
            #     ampt_slope = Measure(df_spread['curvature'][0], df_spread['curve_err'][0]) / \
            #                  Measure(df_spread['baseline'][0], df_spread['base_err'][0])

    base_curve_lin_fits = pd.DataFrame(base_curve_lin_fits)

    if plot and len(spreads_list) > 0:
        interp = interp1d(spreads_list, ramp_slopes, kind='cubic')
        x_plt = np.linspace(min(spreads_list), max(spreads_list), 1000)
        ax_ramp_slope.plot(x_plt, interp(x_plt), ls='--', color='salmon')
        ax_ramp_slope.errorbar(spreads_list, ramp_slopes, ramp_slope_errs, ls='none', marker='o', alpha=0.9)
        ax.legend()
        ax.set_ylim(top=1.05 * max(df_sim['curvature']))
        ax_rs.legend()
        ax.grid()
        ax_ramp_slope.grid()
        ax_rs.grid()
        ax_rs.set_ylim(top=1.05 * max(all_rs))
        fig.tight_layout()
        fig_rs.tight_layout()
        fig_ramp_slope.tight_layout()

    return base_curve_lin_fits


def plot_fits(df_fits):
    for stat, df in df_fits.items():
        if 'int' in df:
            fig_int, ax_int = plt.subplots()
            ax_int.errorbar(df['spread'], df['int'], df['int_err'], ls='none')
            ax_int.set_title(f'{stat} Intercepts')
            ax_int.set_xlabel('Spread')
            ax_int.set_ylabel('Intercept')

        fig_slope, ax_slope = plt.subplots()
        print(df)
        interp = interp1d(df['spread'], df['slope'], kind='cubic')
        x_plt = np.linspace(min(df['spread']), max(df['spread']), 1000)
        ax_slope.plot(x_plt, interp(x_plt), ls='--', color='salmon')
        ax_slope.errorbar(df['spread'], df['slope'], df['slope_err'], ls='none', marker='o', alpha=0.9)
        ax_slope.set_title(f'{stat} Slopes')
        ax_slope.set_xlabel('Spread')
        ax_slope.set_ylabel('Slope')
        ax_slope.grid()
        fig_slope.tight_layout()


def plot_base_zeros(df, data_sets_plt, data_sets_labels, data_sets_colors, cent='all', plot_sims=False):
    fig_base_zeros, ax_base_zeros = plt.subplots(dpi=144)
    fig_zeros_base_amps, ax_zeros_base_amps = plt.subplots(dpi=144)

    ax_base_zeros.axhline(0, color='black')
    ax_zeros_base_amps.axvline(0, color='black')

    ax_base_zeros.grid()
    ax_zeros_base_amps.grid()
    if plot_sims:
        fig_zeros_base_spreads, ax_zeros_base_spreads = plt.subplots(dpi=144)
        fig_zeros_base_amps_spreads, ax_zeros_base_amps_spreads = plt.subplots(dpi=144)
        ax_zeros_base_spreads.axvline(0, color='black')
        ax_zeros_base_amps_spreads.axvline(0, color='black')
        ax_zeros_base_spreads.grid()
        ax_zeros_base_amps_spreads.grid()

    for data_set in data_sets_plt:
        df_set = df[df['data_set'].str.contains(data_set)]
        if cent != 'all':
            df_set = df_set[df_set['cent'] == cent]
        ax_base_zeros.errorbar(df_set['zero_mag'], df_set['baseline'], yerr=df_set['base_err'], marker='o',
                               ls='none', xerr=df_set['zero_mag_err'], color=data_sets_colors[data_set],
                               label=data_sets_labels[data_set])
        ax_zeros_base_amps.errorbar(df_set['baseline'], df_set['zero_mag'], xerr=df_set['base_err'], marker='o',
                                    ls='none', yerr=df_set['zero_mag_err'], color=data_sets_colors[data_set],
                                    label=data_sets_labels[data_set])
        if plot_sims:
            ax_zeros_base_spreads.errorbar(df_set['baseline'], df_set['zero_mag'], xerr=df_set['base_err'], marker='o',
                                           ls='none', yerr=df_set['zero_mag_err'], color=data_sets_colors[data_set],
                                           label=data_sets_labels[data_set])
            ax_zeros_base_amps_spreads.errorbar(df_set['baseline'], df_set['zero_mag'], xerr=df_set['base_err'],
                                                marker='o', ls='none', yerr=df_set['zero_mag_err'],
                                                color=data_sets_colors[data_set], label=data_sets_labels[data_set])

    if plot_sims:
        df_sims = df[df['data_set'].str.contains('sim_')]
        df_sims = df_sims[(df_sims['baseline'] < 0) & (df_sims['zero_mag'] < 220) & (df_sims['zero_mag'] > 110)]

        amps = sorted(pd.unique(df_sims['amp']))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(amps))))
        for amp in amps:
            df_amp = df_sims[df_sims['amp'] == amp]
            df_amp.sort_values('spread')
            c = next(color)
            ax_zeros_base_amps.fill_between(df_amp['baseline'], df_amp['zero_mag'] + df_amp['zero_mag_err'],
                                            df_amp['zero_mag'] + df_amp['zero_mag_err'], alpha=0.5, color=c)
            ax_zeros_base_amps.fill_betweenx(df_amp['zero_mag'], df_amp['baseline'] + df_amp['base_err'],
                                             df_amp['baseline'] + df_amp['base_err'], alpha=0.5, color=c,
                                             label=f'A={amp}')
            ax_zeros_base_amps_spreads.fill_between(df_amp['baseline'], df_amp['zero_mag'] + df_amp['zero_mag_err'],
                                                    df_amp['zero_mag'] + df_amp['zero_mag_err'], alpha=0.5, color=c)
            ax_zeros_base_amps_spreads.fill_betweenx(df_amp['zero_mag'], df_amp['baseline'] + df_amp['base_err'],
                                                     df_amp['baseline'] + df_amp['base_err'], alpha=0.5, color=c,
                                                     label=f'A={amp}')

        spreads = sorted(pd.unique(df_sims['spread']))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(spreads))))
        for spread in spreads:
            df_spread = df_sims[df_sims['spread'] == spread]
            df_spread.sort_values('amp')
            c = next(color)
            ax_zeros_base_spreads.fill_between(df_spread['baseline'], df_spread['zero_mag'] + df_spread['zero_mag_err'],
                                               df_spread['zero_mag'] + df_spread['zero_mag_err'], alpha=0.5, color=c)
            ax_zeros_base_spreads.fill_betweenx(df_spread['zero_mag'], df_spread['baseline'] + df_spread['base_err'],
                                                df_spread['baseline'] + df_spread['base_err'], alpha=0.5, color=c,
                                                label=rf'$\sigma=${spread:.2f}')
            ax_zeros_base_amps_spreads.fill_between(df_spread['baseline'],
                                                    df_spread['zero_mag'] + df_spread['zero_mag_err'],
                                                    df_spread['zero_mag'] + df_spread['zero_mag_err'], alpha=0.5,
                                                    color=c)
            ax_zeros_base_amps_spreads.fill_betweenx(df_spread['zero_mag'],
                                                     df_spread['baseline'] + df_spread['base_err'],
                                                     df_spread['baseline'] + df_spread['base_err'], alpha=0.5, color=c,
                                                     label=rf'$\sigma=${spread:.2f}')

        all_handles_zeros_base_amp, _ = ax_zeros_base_amps.get_legend_handles_labels()
        data_handles_zeros_base_amp = [h for h in all_handles_zeros_base_amp if
                                       isinstance(h, mpl.container.ErrorbarContainer)]
        sim_handles_zeros_base_amp = [h for h in all_handles_zeros_base_amp if h not in data_handles_zeros_base_amp]

        all_handles_zeros_base_spread, _ = ax_zeros_base_spreads.get_legend_handles_labels()
        data_handles_zeros_base_spread = [h for h in all_handles_zeros_base_spread
                                          if isinstance(h, mpl.container.ErrorbarContainer)]
        sim_handles_zeros_base_spread = [h for h in all_handles_zeros_base_spread
                                         if h not in data_handles_zeros_base_spread]

        # all_handles_zeros_base_amp_spread, _ = ax_zeros_base_spreads.get_legend_handles_labels()
        # data_handles_zeros_base_amp_spread = [h for h in all_handles_zeros_base_spread
        #                                       if isinstance(h, mpl.container.ErrorbarContainer)]
        # sim_handles_zeros_base_spread = [h for h in all_handles_zeros_base_spread
        #                                  if h not in data_handles_zeros_base_spread]

    ax_base_zeros.set_xlabel('Zeros')
    ax_base_zeros.set_ylabel('Baseline')
    ax_zeros_base_amps.set_xlabel('Baseline')
    ax_zeros_base_amps.set_ylabel('Zeros')

    ax_base_zeros.set_xlim(110, 230)
    ax_zeros_base_amps.set_ylim(110, 230)

    ax_base_zeros.set_ylim(bottom=-0.0065)
    ax_zeros_base_amps.set_xlim(left=-0.0065, right=0.0005)

    ax_base_zeros.legend()
    if plot_sims:
        data_legend_zeros_base_amp = ax_zeros_base_amps.legend(handles=data_handles_zeros_base_amp, loc='upper center')
        sim_legend_zeros_base_amp = ax_zeros_base_amps.legend(handles=sim_handles_zeros_base_amp, loc='upper left')
        ax_zeros_base_amps.add_artist(data_legend_zeros_base_amp)
        ax_zeros_base_amps.add_artist(sim_legend_zeros_base_amp)
    else:
        ax_zeros_base_amps.legend()

    fig_base_zeros.tight_layout()
    fig_base_zeros.canvas.manager.set_window_title('Baselines vs Zeros')
    fig_zeros_base_amps.tight_layout()
    fig_zeros_base_amps.canvas.manager.set_window_title('Zeros vs Baselines Amp Contours')

    if plot_sims:
        ax_zeros_base_spreads.set_xlabel('Baseline')
        ax_zeros_base_spreads.set_ylabel('Zeros')
        ax_zeros_base_spreads.set_ylim(110, 230)
        ax_zeros_base_spreads.set_xlim(left=-0.0065, right=0.0005)
        data_legend_zeros_base_spread = ax_zeros_base_spreads.legend(handles=data_handles_zeros_base_spread,
                                                                     loc='upper center')
        sim_legend_zeros_base_spread = ax_zeros_base_spreads.legend(handles=sim_handles_zeros_base_spread,
                                                                    loc='upper left')
        ax_zeros_base_spreads.add_artist(data_legend_zeros_base_spread)
        ax_zeros_base_spreads.add_artist(sim_legend_zeros_base_spread)
        fig_zeros_base_spreads.tight_layout()
        fig_zeros_base_spreads.canvas.manager.set_window_title('Zeros vs Baselines Spread Contours')

        ax_zeros_base_amps_spreads.set_xlabel('Baseline')
        ax_zeros_base_amps_spreads.set_ylabel('Zeros')
        ax_zeros_base_amps_spreads.set_ylim(110, 230)
        ax_zeros_base_amps_spreads.set_xlim(left=-0.0065, right=0.0005)
        data_legend_zeros_base_amp_spread = ax_zeros_base_amps_spreads.legend(handles=data_handles_zeros_base_spread,
                                                                              loc='upper center')
        # sim_legend_zeros_base_spread = ax_zeros_base_amps_spreads.legend(handles=sim_handles_zeros_base_spread,
        #                                                                  loc='upper left')
        # sim_legend_zeros_base_amp = ax_zeros_base_amps_spreads.legend(handles=sim_handles_zeros_base_amp,
        #                                                               loc='upper right')
        ax_zeros_base_amps_spreads.add_artist(data_legend_zeros_base_amp_spread)
        # ax_zeros_base_amps_spreads.add_artist(sim_legend_zeros_base_spread)
        # ax_zeros_base_amps_spreads.add_artist(sim_legend_zeros_base_amp)
        fig_zeros_base_amps_spreads.tight_layout()
        fig_zeros_base_amps_spreads.canvas.manager.set_window_title('Zeros vs Baselines Amps & Spread Contours')


def flow_vs_v2(df, div, res, out_dir=None):
    df = df[df['name'].str.contains(f'res{res}')]
    df = df[(df['stat'] == 'standard deviation') & (df['divs'] == div) & (df['data_type'] == 'raw')]
    v2s, slope_vals, slope_errs = [], [], []
    p = div / 360.0
    q = 1 - p
    for data_set in pd.unique(df['name']):
        v2 = float('0.' + data_set[data_set.find('_v2') + 3:])
        df_v2 = df[df['name'] == data_set]
        fig, ax = plt.subplots(figsize=(6.66, 5), dpi=144)
        ax.axhline(1, color='black', ls='--')
        raw_div_binom = [Measure(val, err) / np.sqrt(n * p * q) for n, val, err in
                         zip(df_v2['total_protons'], df_v2['val'], df_v2['err'])]
        y_vals, y_errs = [x.val for x in raw_div_binom], [x.err for x in raw_div_binom]
        ax.errorbar(df_v2['total_protons'], y_vals, y_errs, marker='o', ls='none', color='blue')
        ax.set_xlabel('Total Protons in Event')
        ax.set_ylabel('Raw / Binomial')
        ax.set_title(data_set)
        popt, pcov = cf(line_yint1, df_v2['total_protons'], y_vals, sigma=y_errs, absolute_sigma=True)
        ax.plot(df_v2['total_protons'], line_yint1(df_v2['total_protons'], *popt), ls='--', color='red')
        v2s.append(v2)
        slope_vals.append(popt[0])
        slope_errs.append(np.sqrt(np.diag(pcov))[0])

    v2s, slope_vals, slope_errs = zip(*sorted(zip(v2s, slope_vals, slope_errs)))
    v2s, slope_vals, slope_errs = np.array(v2s), np.array(slope_vals), np.array(slope_errs)
    fig, ax = plt.subplots(figsize=(6.66, 5), dpi=144)
    ax.set_xlabel('v2')
    ax.set_ylabel(r'$\sigma_{raw}/\sigma_{binomial}$ vs Total Protons Slope')
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.errorbar(v2s, slope_vals, slope_errs, marker='o', ls='none', alpha=0.8)
    popt, pcov = cf(quad_func, v2s, slope_vals, sigma=slope_errs, absolute_sigma=True)
    print('With constant and linear terms: ', [f'{val} +- {err}' for val, err in zip(popt, np.sqrt(np.diag(pcov)))])
    popt2, pcov2 = cf(x2, v2s, slope_vals, sigma=slope_errs, absolute_sigma=True)
    print('Just quadratic term: ', [f'{val} +- {err}' for val, err in zip(popt2, np.sqrt(np.diag(pcov2)))])
    xs = np.linspace(0, 0.07, 200)
    ax.plot(xs, quad_func(xs, *popt), ls='--', color='red', label='3 Parameter')
    ax.plot(xs, x2(xs, *popt2), ls='--', color='green', label='1 Parameter')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if out_dir:
        fig.savefig(f'{out_dir}slope_vs_v2_div_{div}.png')
        with open(f'{out_dir}slope_vs_v2_data_div_{div}.txt', 'w') as file:
            file.write(f'y=a*x^2 fit:\t{popt2[0]} {np.sqrt(pcov2[0][0])}\n')
            fit_string = '\t'.join([f'{val} {err}' for val, err in zip(popt, np.sqrt(np.diag(pcov)))])
            file.write(f'y=a*x^2+bx+c fit:\t{fit_string}\n\n')
            file.write('v2\tslope slope_err\n')
            for v2, slope_val, slope_err in zip(v2s, slope_vals, slope_errs):
                file.write(f'{v2}\t{slope_val} {slope_err}\n')


def plot_closures(df_sig, df_combo, df_corrected, alpha=0.6, df_bkg=None, title=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 0.3], hspace=0)
    ax = fig.add_subplot(gs[0])
    ax.axhline(0, color='black')
    ax_diff = fig.add_subplot(gs[1], sharex=ax)
    ax_diff.axhline(0, color='black')
    ax.errorbar(df_sig['divs'], df_sig['avg'], yerr=df_sig['avg_err'], ls='none', marker='o', color='blue',
                alpha=alpha, label='Signal')
    bkg_markers = ['o', 's', '^']
    if df_bkg is not None:
        if type(df_bkg) is list:
            for bkg_i, bkg in enumerate(df_bkg):
                ax.errorbar(bkg['divs'], bkg['avg'], yerr=bkg['avg_err'], ls='none', marker=bkg_markers[bkg_i],
                            color='green', alpha=alpha, label=f'Background #{bkg_i + 1}')
        else:
            ax.errorbar(df_bkg['divs'], df_bkg['avg'], yerr=df_bkg['avg_err'], ls='none', marker='o', color='green',
                        alpha=alpha, label='Background')

    ax.errorbar(df_combo['divs'], df_combo['avg'], yerr=df_combo['avg_err'], ls='none', marker='o', color='orange',
                alpha=alpha, label='Combination')

    # df_diff_err = np.array(df_combo['avg']) * \
    #               np.sqrt(np.mean(np.array(df_combo['avg']) - np.array(df_corrected['avg']))) / 2
    if type(df_bkg) is list:
        errs = []
        for bkg in df_bkg:
            errs.append((np.abs(np.array(df_sig['avg']) * np.mean(bkg['avg']))) ** 0.75)
        df_diff_err = np.sqrt(np.sum(np.array(errs), axis=0) ** 2)
    else:
        df_bkg = np.array(df_combo['avg']) - np.array(df_corrected['avg']) if df_bkg is None else df_bkg
        df_diff_err = (np.abs(np.array(df_sig['avg']) * np.mean(df_bkg['avg']))) ** 0.75
    ax.errorbar(df_corrected['divs'], df_corrected['avg'], yerr=df_corrected['avg_err'], ls='none', marker='x',
                color='red', alpha=0.9, label='Corrected')
    ax.errorbar(df_corrected['divs'], df_corrected['avg'], yerr=df_diff_err, ls='none', marker=None, elinewidth=4,
                color='red', alpha=0.2)
    df_diff = np.array(df_sig['avg_meas']) - np.array(df_corrected['avg_meas'])
    ax_diff.errorbar(df_sig['divs'], [x.val for x in df_diff], yerr=[x.err for x in df_diff], ls='none',
                     marker='o', alpha=alpha, color='red', label='Signal - Corrected')
    ax_diff.errorbar(df_sig['divs'], [x.val for x in df_diff], yerr=df_diff_err, ls='none', elinewidth=4,
                     marker=None, alpha=0.2, color='red')

    ax.legend(loc='upper right')
    ax_diff.legend()
    ax_diff.set_xlim((0, 360))
    ax.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
    # ax_diff.set_ylabel('Correction Deviation')
    ax_diff.set_ylabel(r'$\langle\Delta\sigma^2\rangle$')
    ax_diff.set_xlabel(r'Azimuthal Partition Width ($w$)')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax_diff.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)


def truncate_vs_tproton(df, err_thresh):
    return


def calc_dsigma(df, data_types='diff'):
    if type(data_types) == str:
        data_types = [data_types]
    dfs = []
    for data_type in data_types:
        df_type = df[df['data_type'] == data_type].copy()
        p, tp = df_type['divs'] / 360, df_type['total_protons']
        if data_type in ['raw', 'mix']:
            df_type.loc[:, 'val'] = (df_type['val'] - (tp * p * (1 - p))) / (tp * (tp - 1))
            df_type.loc[:, 'err'] = df_type['err'] / (tp * (tp - 1))
        elif data_type == 'diff':
            df_type.loc[:, 'val'] = df_type['val'] / (tp * (tp - 1))
            df_type.loc[:, 'err'] = df_type['err'] / (tp * (tp - 1))
        else:
            print(f'Don\'t recognize data type: {data_type}')
            df_type = None
        dfs.append(df_type)

    return dfs if len(dfs) > 1 else dfs[0]


def subtract_avgs(df_a, df_b, val_col='avg', err_col='avg_err', meas_col='avg_meas'):
    if meas_col in df_a.columns:
        index_cols = list(set(df_a.columns) - {val_col, err_col, meas_col})
        df_sub = df_a.set_index(index_cols) - df_b.set_index(index_cols)
        df_sub.loc[:, err_col] = df_sub[meas_col].apply(lambda x: x.err)
        # df_sub.loc[:, err_col] = df_sub.assign(err_col=lambda x: x[meas_col].err)  # Think this is right but check
    else:
        index_cols = list(set(df_a.columns) - {val_col, err_col})
        df_sub = df_a.set_index(index_cols) - df_b.set_index(index_cols)
        df_sub.loc[:, err_col] = np.sqrt(df_a.set_index(index_cols)[err_col] ** 2 +
                                         df_b.set_index(index_cols)[err_col] ** 2)

    return df_sub.reset_index()


def read_v2_slope_coefs(in_dir):
    coefs = {}
    files = os.listdir(in_dir)
    for file_name in files:
        if 'slope_vs_v2_data_' in file_name:
            div = int(file_name.split('.')[0].split('_')[-1])
            with open(f'{in_dir}{file_name}', 'r') as file:
                coef = file.readlines()[0].split('\t')[-1].split(' ')
                coef_val, coef_err = float(coef[0]), float(coef[1])
            coefs.update({div: Measure(coef_val, coef_err)})
    return coefs


def read_flow_values(in_dir, v_type='v2'):
    vs = {}
    dir_names = os.listdir(in_dir)
    for dir_name in dir_names:
        if os.path.isdir(os.path.join(in_dir, dir_name)):
            energy = int(dir_name.strip('GeV'))
            vs.update({energy: {}})
            file_path = f'{in_dir}{dir_name}/{v_type}.txt'
            if not os.path.isfile(file_path):
                print(f'{file_path} not found')
                continue
            with open(file_path, 'r') as file:
                lines = file.readlines()[1:]  # Skip column headers
                for line in lines:
                    line = line.split('\t')
                    if len(line) > 2:
                        cent = int(line[0])
                        line_v = line[1].split(' ')
                        v_val = float(line_v[0])
                        v_err = float(line_v[1])
                        vs[energy].update({cent: Measure(v_val, v_err)})
    # for energy in vs:  # One of the resolutions at 70-80% centrality was negative so v2 values nan
    #     for cent in vs[energy]:  # This is a quick fix to just set the v2 value to the next more central v2
    #         if np.isnan(vs[energy][cent].val):  # Not ideal, hopefully fix the resolution issue later
    #             next_cent = cent + 1
    #             while np.isnan(vs[energy][next_cent]):
    #                 next_cent += 1
    #             vs[energy][cent] = vs[energy][]
    return vs


def ampt_v2_closure_sub(slope_df, data_set_name, new_name, v2, coef, cent):
    df_new = slope_df[slope_df['name'] == data_set_name]
    df_new = df_new.assign(name=new_name)
    new_slope = [slope - coef * v2[energy][cent] ** 2 for energy, slope in zip(df_new['energy'], df_new['slope_meas'])]
    new_slope_vals, new_slope_errs = list(zip(*[(slope.val, slope.err) for slope in new_slope]))
    df_new = df_new.assign(slope=new_slope_vals)
    df_new = df_new.assign(slope_err=new_slope_errs)

    return pd.concat([slope_df, df_new], ignore_index=True)


def ampt_v2_closure_sub_dsigma(avg_df, data_set_name, new_name, v2, div, cent):
    df_new = avg_df[avg_df['name'] == data_set_name]
    df_new = df_new.assign(name=new_name)
    new_avg = [avg - v2_divs(np.deg2rad(div), v2[energy][cent]) for energy, avg in
               zip(df_new['energy'], df_new['avg_meas'])]
    new_avg_vals, new_avg_errs = list(zip(*[(avg.val, avg.err) for avg in new_avg]))
    df_new = df_new.assign(avg=new_avg_vals)
    df_new = df_new.assign(avg_err=new_avg_errs)

    return pd.concat([avg_df, df_new], ignore_index=True)


def subtract_dsigma_flow(avg_df, data_set_name, new_name, vs, div=None, cent=None, new_only=False,
                         val_col='avg', err_col='avg_err', meas_col='avg_meas'):
    df_new = avg_df[avg_df['name'] == data_set_name]
    if len(df_new) > 0:
        df_new = df_new.assign(name=new_name)
        if div is not None:
            df_new['div'] = div
        if cent is not None:
            df_new['cent'] = cent
        new_avg = [avg - flow_correction(np.deg2rad(div), vs, energy, cent) for energy, cent, div, avg in
                   zip(df_new['energy'], df_new['cent'], df_new['divs'], df_new[meas_col])]
        new_avg = [flow_error(vs, energy, cent, avg) for energy, cent, avg in
                   zip(df_new['energy'], df_new['cent'], new_avg)]
        new_avg_vals, new_avg_errs = list(zip(*[(avg.val, avg.err) for avg in new_avg]))
        df_new[meas_col] = new_avg
        df_new[val_col] = new_avg_vals
        df_new[err_col] = new_avg_errs

    if new_only:
        return df_new
    else:
        return pd.concat([avg_df, df_new], ignore_index=True)


def plot_lyons_sys(df, df_def_name, df_sys_dict, sys_prior_dict=None, group_cols=None,
            val_col='val', err_col='err', name_col='name', sys_col='sys'):
    """
    Calculate systematic uncertainties on the default set in df.
    Only ready to take single variation per systematic variable.
    :param df: Dataframe
    :param df_def_name: Default set name
    :param df_sys_dict: Dictionary of systematic sets to be analyzed. {set_type: [set_names]}
    :param sys_prior_dict: Dictionary of systematic prior distributions. If None, all gaus. {set_name: prior}
            gaus: Gaussian prior. Variation(s) correspond to 1 sigma estimate. If 2 variations, take max.
            flat_one_side: Uniform prior. Variation corresponds to edge of distribution, with default on other edge.
                            Divide by 1/sqrt(12) to get 1 sigma estimate.
            flat_two_side: Uniform prior. Variation(s) correspond to edge of distribution, with default in the center.
                            Divide by 2/sqrt(12)=1/sqrt(3) to get 1 sigma estimate.
    :param group_cols: Name of columns by which to group values.
    For example, energies and centralities should be separate
    :param val_col: Name of column containing the values of interest
    :param err_col: Name of column containing the statistical uncertainties on the values of interest
    :param name_col: Name of column for the name of the data sets
    :param sys_col: Name of the column for the systematic to output
    :return: Dataframe with default values, statistical uncertainties, and estimated systematic uncertainties
    """
    if group_cols is None:
        group_cols = ['divs', 'energy', 'cent', 'data_type', 'total_protons']

    sys_set_names = [y for x in df_sys_dict.values() for y in x]
    df_filtered = df[df[name_col].isin(list(sys_set_names) + [df_def_name])]
    df_set = df_filtered.groupby(group_cols)
    df_def_sys = []

    for group_name, group_df in df_set:
        if df_def_name not in group_df[name_col].values:
            continue  # No default value so no systematic

        group_df_def = group_df[group_df[name_col] == df_def_name]
        assert len(group_df_def) == 1
        def_val, def_err = group_df_def[val_col].values[0], group_df_def[err_col].values[0]
        barlow = 0

        if len(group_df) <= 1:
            pass  # No systematic values so sys = 0
        else:
            group_df_sys = group_df[group_df[name_col] != df_def_name]

            barlow = 0
            for sys_type, sys_names in df_sys_dict.items():
                group_df_sys_type = group_df_sys[group_df_sys[name_col].isin(sys_names)]
                if len(group_df_sys_type) == 0:
                    continue  # No variations of this type
                sys_val = group_df_sys_type[val_col].values
                sys_err = group_df_sys_type[err_col].values

                names = group_df_sys_type[name_col].values
                diff_vals = def_val - sys_val
                diff_errs = np.sqrt(np.abs(def_err**2 - sys_err**2))
                print(f'Systematic type: {sys_type}')
                print(f'Group df sys_type: {group_df_sys_type}')
                print(f'diff_vals: {diff_vals}')
                print(f'diff_errs: {diff_errs}')
                s = solve_for_lyons_s(diff_vals, diff_errs)
                diff_errs_s = np.sqrt(diff_errs ** 2 + s ** 2)
                print(f'Extra systematic to cover: {s}')
                fig, ax = plt.subplots()
                ax.errorbar(names, diff_vals, diff_errs, ls='none', marker='o', elinewidth=4, label='Statistical Errors')
                ax.errorbar(names, diff_vals, diff_errs_s, ls='none', marker='.', elinewidth=1, label='Stat + Sys Errors')
                ax.axhline(0, color='black', zorder=0)
                ax.set_title(f'Systematic Type: {sys_type}')
                ax.set_ylabel(f'Default - {sys_type} Variation')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()

                # fig, ax = plt.subplots()
                # ax.errorbar()

                # barlow_i = calc_sys(def_val, def_err, sys_val, sys_err, 'indiv') ** 2
                # if sys_prior_dict is None or sys_type not in sys_prior_dict or sys_prior_dict[sys_type] == 'gaus':
                #     barlow_i = barlow_i  # Do nothing, treat as gaussian 1 sigma. Dumb if statement but hopefully clear.
                # elif sys_prior_dict[sys_type] == 'flat_one_side':
                #     barlow_i = barlow_i / 12  # 1 sigma estimate for a uniform distribution with default at one edge
                # elif sys_prior_dict[sys_type] == 'flat_two_side':
                #     barlow_i = barlow_i / 3  # 1 sigma estimate for a uniform distribution with default at center
                # barlow += np.max(barlow_i) if barlow_i.size > 0 else 0
            barlow = np.sqrt(barlow)

        df_entry = group_df_def.reset_index().to_dict(orient='records')[0]
        df_entry.update({sys_col: barlow})
        del df_entry['index']
        df_def_sys.append(df_entry)

    return pd.DataFrame(df_def_sys)


def solve_for_lyons_s(def_minus_sys, err, s_max_mult=100):
    """
    Solve for systematic uncertainty numerically using Lyons method.
    :param def_minus_sys: Array of default value minus systematic variations
    :param err: Array of uncertainties on this difference -- if subset |\sqrt{err_d^2 - err_s^2}|
    :param s_max_mult: Multiplier for max of def_minus_sys to set upper bound of search interval
    :return: s -- Extra systematic to cover
    """
    # Filter out entries in which err is nan or <= 0 or def_minus_sys is nan
    mask = (~np.isnan(err)) & (err > 0) & (~np.isnan(def_minus_sys))
    err = err[mask]
    def_minus_sys = def_minus_sys[mask]

    n = len(def_minus_sys)

    if n == 0:  # No valid variations so no extra systematic needed
        return 0.0

    def lyons_func(s):
        return np.sum(def_minus_sys**2 / (err**2 + s**2)) / n - 1

    if lyons_func(0) <= 0:  # No extra systematic needed
        return 0.0

    # Search interval: s must be >= 0.
    # f(s) is decreasing in s, so we can bracket between 0 and something large.
    s_max = max(1.0, s_max_mult * np.max(def_minus_sys**2))  # heuristic upper bound

    while lyons_func(s_max) > 0:
        s_max *= 2

    root = brentq(lyons_func, 0.0, s_max)
    return root


def flow_correction(div, vs, energy, cent):
    correction = 0
    for order, v in vs.items():
        correction += vn_divs(div, v[energy][cent], order)
    return correction


def flow_error(vs, energy, cent, new_avg):
    for order, v in vs.items():
        div_avg = vn_divs(np.pi / (2 * order), v[energy][cent], order)
        new_err = (abs(new_avg.val * div_avg.val)) ** 0.75
        new_avg += Measure(0, new_err)
    return new_avg


def map_to_sim(baseline, zeros, interpolations):
    """
    Given baseline and zeros of quadratic fit to Raw SD / Mix SD Slope vs Partition Width, map to simulation amp, spread
    parameters with input interpolations
    :param baseline: Constant term in quadratic fit of Raw SD / Mix SD Slope vs Partition Width
    :param zeros: Reparameterized curvature term in quadratic fit of Raw SD / Mix SD Slope vs Partition Width
    :param interpolations: Interpolations from simulations for converting baseline and zeros to sim amp and spread
    :return: Interpolated simulation amplitude and spread, mapped from input baseline and zeros
    """
    if baseline > 0:
        interpolations = interpolations['_clmul_']
    else:
        interpolations = interpolations['_aclmul_']
    spread = interpolations['spread_zero'][zeros]
    amp_slope = interpolations['amp_slope_spread'][spread]
    amp = amp_slope * baseline

    return amp, spread


class MyBounds:
    def __init__(self, xmax=[0, 0.06], xmin=[0, 4.2]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
