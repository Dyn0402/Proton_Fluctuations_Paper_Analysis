#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 28 7:02 PM 2021
Created in PyCharm
Created as QGP_Scripts/test

@author: Dylan Neff, dylan
"""

import numpy as np
from DistStats import DistStats


def sim_events(dist, trials, moment_pars, binning, percentiles, nevents_state):
    events = nevents_state[0]
    print(f'Events: {events}')
    trial_moments = {x: [] for x in moment_pars.keys()}
    for trial in range(trials):
        hist, bin_edges = np.histogram(dist.rvs(size=events, random_state=nevents_state[1]), binning)
        hist = dict(zip((bin_edges[1:] + bin_edges[:-1]) / 2.0, hist))
        for moment, moment_val in calc_moments(hist, moment_pars).items():
            trial_moments[moment].append(moment_val)
    return comb_trials(trial_moments, percentiles)


def calc_moments(hist, moment_pars):
    moments = {}
    stats = DistStats(hist)
    for moment, pars in moment_pars.items():
        moments.update({moment: pars['method'](stats)})

    return moments


def sim_events_single(dist, moment_pars, binning, nevents_state):
    events = nevents_state[0]
    print(f'Events: {events}')
    means, errs = {}, {}
    hist, bin_edges = np.histogram(dist.rvs(size=events, random_state=nevents_state[1]), binning)
    hist = dict(zip((bin_edges[1:] + bin_edges[:-1]) / 2.0, hist))
    for moment, moment_meas in calc_moments_single(hist, moment_pars).items():
        means.update({moment: moment_meas.val})
        errs.update({moment: moment_meas.err})

    return means, errs, events


def calc_moments_single(hist, moment_pars):
    moments = {}
    stats = DistStats(hist)
    for moment, pars in moment_pars.items():
        if 'method_single' in pars:
            moments.update({moment: pars['method_single'](stats)})

    return moments


def sim_cent_trial(dist, moment_pars, binning, events, state):
    print(f'Trial #{state[0]}')
    cbwc_moments = {x: 0 for x in moment_pars.keys()}
    event_sum = sum(events)
    for n in events:
        hist, bin_edges = np.histogram(dist.rvs(size=n, random_state=state[1]), binning)
        hist = dict(zip((bin_edges[1:] + bin_edges[:-1]) / 2.0, hist))
        for moment, moment_val in calc_moments(hist, moment_pars).items():
            cbwc_moments[moment] += moment_val * n / event_sum

    return cbwc_moments


def comb_trials(trial_moments, percentiles):
    means, errs, percs = {}, {}, {}
    for moment, trial_vals in trial_moments.items():
        means.update({moment: np.mean(trial_vals)})
        errs.update({moment: np.std(trial_vals) / np.sqrt(len(trial_vals))})
        perc = np.percentile(trial_vals, percentiles)
        percs.update({moment: dict(zip(percentiles, perc))})

    return means, errs, percs


def comb_trials_cent(trial_moments, percentiles):
    means, errs, percs = {}, {}, {}
    for moment, trial_vals in trial_moments.items():
        means.update({moment: np.median(trial_vals)})
        errs.update({moment: np.std([x.val for x in trial_vals]) / np.sqrt(len(trial_vals))})
        perc = np.percentile([x.val for x in trial_vals], percentiles)
        percs.update({moment: dict(zip(percentiles, perc))})

    return means, errs, percs


def get_c1(x):
    return x.get_cumulant(1).val


def get_c2(x):
    return x.get_cumulant(2).val


def get_c3(x):
    return x.get_cumulant(3).val


def get_c4(x):
    return x.get_cumulant(4).val


def get_c5(x):
    return x.get_cumulant(5).val


def get_c6(x):
    return x.get_cumulant(6).val


def get_k1(x):
    return x.get_k_stat(1).val


def get_k2(x):
    return x.get_k_stat(2).val


def get_k3(x):
    return x.get_k_stat(3).val


def get_k4(x):
    return x.get_k_stat(4).val


def get_k5(x):
    return x.get_k_stat(5).val


def get_k6(x):
    return x.get_k_stat(6).val


def get_c4_div_c2(x):
    return x.get_cumulant(4).val / x.get_cumulant(2).val


def get_k4_div_k2(x):
    return x.get_k_stat(4).val / x.get_k_stat(2).val


def get_c6_div_c2(x):
    return x.get_cumulant(6).val / x.get_cumulant(2).val


def get_k6_div_k2(x):
    return x.get_k_stat(6).val / x.get_k_stat(2).val


def get_c4_div_c2_sub_k4_div_k2(x):
    return x.get_cumulant(4).val / x.get_cumulant(2).val - x.get_k_stat(4).val / x.get_k_stat(2).val


def get_c6_div_c2_sub_k6_div_k2(x):
    return x.get_cumulant(6).val / x.get_cumulant(2).val - x.get_k_stat(6).val / x.get_k_stat(2).val


def get_c1_meas(x):
    return x.get_cumulant(1)


def get_c2_meas(x):
    return x.get_cumulant(2)


def get_c3_meas(x):
    return x.get_cumulant(3)


def get_c4_meas(x):
    return x.get_cumulant(4)


def get_c5_meas(x):
    return x.get_cumulant(5)


def get_c6_meas(x):
    return x.get_cumulant(6)


def get_k1_meas(x):
    return x.get_k_stat(1)


def get_k2_meas(x):
    return x.get_k_stat(2)


def get_k3_meas(x):
    return x.get_k_stat(3)


def get_k4_meas(x):
    return x.get_k_stat(4)


def get_k5_meas(x):
    return x.get_k_stat(5)


def get_k6_meas(x):
    return x.get_k_stat(6)


def get_c4_div_c2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_cumulant(4) / x.get_cumulant(2)


def get_k4_div_k2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_k_stat(4) / x.get_k_stat(2)


def get_c6_div_c2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_cumulant(6) / x.get_cumulant(2)


def get_k6_div_k2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_k_stat(6) / x.get_k_stat(2)


def get_c4_div_c2_sub_k4_div_k2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_cumulant(4) / x.get_cumulant(2) - x.get_k_stat(4) / x.get_k_stat(2)


def get_c6_div_c2_sub_k6_div_k2_meas(x):  # Delta theorem errors will be wrong, need to apply directly to quantity
    return x.get_cumulant(6) / x.get_cumulant(2) - x.get_k_stat(6) / x.get_k_stat(2)


def get_mean_meas(x):
    return x.get_mean()


def get_sd_meas(x):
    return x.get_sd()


def get_skewness_meas(x):
    return x.get_skewness()


def get_kurtosis_meas(x):
    return x.get_kurtosis()


def get_nekurtosis_meas(x):
    return x.get_non_excess_kurtosis()


def main():
    print('donzo')


if __name__ == '__main__':
    main()
