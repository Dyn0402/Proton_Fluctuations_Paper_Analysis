#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 06 7:27 PM 2020
Created in PyCharm
Created as QGP_Scripts/AzimuthBinData.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


class AzimuthBinData:
    def __init__(self, div=0, path=''):
        """

        :param div: Azimuthal division size in degrees
        :param path: Path to bin data text file
        """
        self.path = path
        self.div = div

        self.max_particle = 0
        self.max_bin = None

        self.data = {}
        self.ratio_dist = {}
        self.pull_dist = {}

        if path != '':
            self.read_data()

    def read_data(self):
        with open(self.path, 'r') as file:
            lines = file.readlines()
            # self.max_particle = int(lines[-1].strip().split('\t')[0])
            for line in lines:
                line = line.strip().split('\t')
                if len(line) == 2:
                    total_particles = int(line[0])
                    bin_particles_string = line[1].split(' ')
                    if len(bin_particles_string) > 0:
                        self.data[total_particles] = []
                    i = 0
                    for entry in bin_particles_string:
                        entry = entry.split(':')
                        while int(entry[0]) > i:
                            self.data[total_particles].append(0)
                            i += 1
                        self.data[total_particles].append(int(entry[1]))
                        i += 1
                else:
                    break
            self.max_particle = max(self.data)

    def ratio_trans(self):
        if len(self.ratio_dist) == 0:  # Skip if pull_dist is already filled
            for total_particles in self.data:
                for bin_particles in range(len(self.data[total_particles])):
                    ratio = bin_particles / total_particles
                    if ratio in self.ratio_dist:
                        self.ratio_dist[ratio] += self.data[total_particles][bin_particles]
                    else:
                        self.ratio_dist.update({ratio: self.data[total_particles][bin_particles]})

    def pull_trans(self):
        if len(self.pull_dist) == 0:  # Skip if pull_dist is already filled
            for total_particles in self.data:
                for bin_particles in range(len(self.data[total_particles])):
                    p = self.div / 360
                    diff = bin_particles - p * total_particles
                    pull = diff / (total_particles * p * (1 - p))**0.5
                    if pull in self.pull_dist:
                        self.pull_dist[pull] += self.data[total_particles][bin_particles]
                    else:
                        self.pull_dist.update({pull: self.data[total_particles][bin_particles]})

    def get_ratio_dist(self, binning=None, norm=False):
        self.ratio_trans()
        if binning is None:
            return self.ratio_dist
        else:
            return self.bin_dist('ratio_dist', binning, norm)
            # hist = [0 for i in range(len(binning) - 1)]
            # for bin_index in range(len(hist)):
            #     for val, count in self.ratio_dist.items():
            #         if binning[bin_index] <= val < binning[bin_index + 1]:
            #             hist[bin_index] += count
            # if norm:
            #     total = sum(hist)
            #     for bin_index in range(len(hist)):
            #         hist[bin_index] /= total
            # return hist

    def get_pull_dist(self, binning=None, norm=False):
        self.pull_trans()
        if binning is None:
            return self.pull_dist
        else:
            return self.bin_dist('pull_dist', binning, norm)
            # hist = [0 for i in range(len(binning) - 1)]
            # for bin_index in range(len(hist)):
            #     for val, count in self.pull_dist.items():
            #         if binning[bin_index] <= val < binning[bin_index + 1]:
            #             hist[bin_index] += count
            # if norm:
            #     total = sum(hist)
            #     for bin_index in range(len(hist)):
            #         hist[bin_index] /= total
            # return hist

    def get_dist(self):
        return self.data

    def get_total_particle_dist(self):
        total_part_dist = {}
        for total, bin_dist in self.data.items():
            total_part_dist.update({total: np.sum(bin_dist)})

        return total_part_dist

    def get_max_bin(self):
        if self.max_bin is None:
            self.max_bin = 0
            for total_particles, bins in self.data.items():
                for bin_particles, counts in enumerate(bins):
                    if bin_particles > self.max_bin and counts > 0:
                        self.max_bin = bin_particles
        return self.max_bin

    def print_dist(self):
        for total_particles, bins in self.data.items():
            out_line = f'{total_particles}\t'
            for bin_particles, counts in enumerate(bins):
                out_line += f'{bin_particles}:{counts} '
            print(out_line)

    def plot_ratio_dist(self, binning=None, norm=False, logy=True, show=True):
        self.ratio_trans()
        min_bin_edge = 0
        max_bin_edge = 1.01
        if binning is None:
            binning = np.linspace(min_bin_edge, max_bin_edge, 10)
        else:
            try:  # If list like
                len(binning)
            except TypeError:  # Hopefully an integer
                binning = np.linspace(min_bin_edge, max_bin_edge, binning)
        hist = self.bin_dist('ratio_dist', binning, norm)
        bin_centers = (binning[1:] + binning[:-1]) / 2
        bin_widths = binning[1:] - binning[:-1]

        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, bin_widths, align='center', zorder=3)
        ax.set_xlabel('Ratio')
        ax.grid(zorder=0)
        ax.set_title(f'Ratio\n{self.path}')
        if logy:
            ax.semilogy()
        if show:
            plt.show()

    def plot_pull_dist(self, binning=None, norm=False, logy=True, show=True):
        self.pull_trans()
        pull_min = min(self.pull_dist)
        pull_max = max(self.pull_dist)
        pull_range = pull_max - pull_min
        min_bin_edge = pull_min
        max_bin_edge = pull_max + pull_range / 100
        if binning is None:
            pull_min = min(self.pull_dist)
            pull_max = max(self.pull_dist)
            pull_range = pull_max - pull_min
            binning = np.linspace(min_bin_edge, max_bin_edge, 10)
        else:
            try:  # If list like
                len(binning)
            except TypeError:  # Hopefully an integer
                binning = np.linspace(min_bin_edge, max_bin_edge, binning)
        hist = self.bin_dist('pull_dist', binning, norm)
        bin_centers = (binning[1:] + binning[:-1]) / 2
        bin_widths = binning[1:] - binning[:-1]

        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, bin_widths, align='center', zorder=3)
        ax.set_xlabel('Pull')
        ax.grid(zorder=0)
        ax.set_title(f'Pull\n{self.path}')
        if logy:
            ax.semilogy()
        if show:
            plt.show()

    def bin_dist(self, dist_name, binning, norm):
        hist = [0 for i in range(len(binning) - 1)]
        for bin_index in range(len(hist)):
            for val, count in self.__getattribute__(dist_name).items():
                if binning[bin_index] <= val < binning[bin_index + 1]:
                    hist[bin_index] += count
        if norm:
            total = sum(hist)
            for bin_index in range(len(hist)):
                hist[bin_index] /= total
        return hist
