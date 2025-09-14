#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 04 7:07 PM 2021
Created in PyCharm
Created as QGP_Scripts/BootstrapAzBin.py

@author: Dylan Neff, Dylan
"""

from AzimuthBinData import AzimuthBinData


class BootstrapAzBin:
    def __init__(self, div=0, path=''):
        self.path = path
        self.div = div

        self.data = None
        self.data_bs = []

        if path != '':
            self.read_bootstrap_data()

    def read_bootstrap_data(self):
        with open(self.path, 'r') as file:
            data = {}
            lines = file.readlines()

            for line in lines:
                if 'bootstrap' in line:
                    self.append_set(data)
                    data = {}
                    continue

                line = line.strip().split('\t')
                if len(line) == 2:
                    try:
                        total_particles = int(line[0])
                        bin_particles_string = line[1].split(' ')
                        if len(bin_particles_string) > 0:
                            data[total_particles] = []
                        i = 0
                        for entry in bin_particles_string:
                            entry = entry.split(':')
                            while int(entry[0]) > i:
                                data[total_particles].append(0)
                                i += 1
                            data[total_particles].append(int(entry[1]))
                            i += 1
                    except ValueError:
                        print(f'Bootstrap read error {self.path}')

            self.append_set(data)

    def get_dist(self):
        return self.data.get_dist()

    def get_total_part_dist(self):
        return self.data.get_total_part_dist()

    def get_dist_bs(self):
        return [bs.get_dist() for bs in self.data_bs]

    def append_set(self, data):
        if data:  # Check if dict is empty
            if self.data is None:
                self.data = AzimuthBinData(self.div)
                self.data.data = data.copy()
                self.data.max_particle = max(data)
            else:
                self.data_bs.append(AzimuthBinData(self.div))
                self.data_bs[-1].data = data.copy()
                self.data_bs[-1].max_particle = max(data)
        else:
            print(f'Misread: {self.path}')
