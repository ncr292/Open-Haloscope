# -*- coding: utf-8 -*-
# Main file with the varius tools used in the Open Haloscope project

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as c


class OHUtils():
    ## Utilities class
    # Class used for some handy functions that always help

    # useful variables
    def __init__(self):
        self.variable = 0

    # helper functions
    def load_experiment_json(haloscope_name):
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        data_folder = 'data'
        experiments_folder = 'experiments'
        haloscope_name += '.json'

        haloscope_json = os.path.join(dirname, data_folder, experiments_folder, haloscope_name)

        return haloscope_json

    def get_runs_folder():
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        data_folder = 'data'
        runs_folder = 'runs'
        data_path = os.path.join(dirname, data_folder, runs_folder)

        return data_path

    def get_logs_folder():
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        data_folder = 'data'
        logs_folder = 'logs'
        logs_path = os.path.join(dirname, data_folder, logs_folder)

        return logs_path

    def add_limit(fname, color, alpha=0.4, label=None, zorder=1):
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
        data_folder = 'data'
        limit_folder = 'limits'
    
        limit = os.path.join(dirname, data_folder, limit_folder, fname)
        
        x, y = np.loadtxt(limit, unpack=True)
        plt.fill_between(x, y, 1, color=color, alpha=alpha,
                         linewidth=0, zorder=zorder, label=label)
        
        plt.plot(x, y, color=color, linewidth=0, zorder=zorder+0.1)