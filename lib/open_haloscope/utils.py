# -*- coding: utf-8 -*-
# Main file with the varius tools used in the Open Haloscope project
# Written by Nicolò Crescini

import numpy as np
from scipy import constants as c


class OHUtils():
    ## Utilities class
    # Class used for some handy functions that always help

    # useful variables
    def __init__(self):
        self.test = 1

    def dB(x):
        return 10 * np.log10(x)

    # unit conversions
    def eV_to_Hz(m):
        f = m * c.e / c.hbar  / (2 * c.pi)
        return f








        



