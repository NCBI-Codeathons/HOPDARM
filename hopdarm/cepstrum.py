#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft, ifft

def cepstrum(x, *args):
    x = np.asarray(x)
    spectrum = fft(x, *args)
    return ifft(np.log(np.abs(spectrum))).real

def sigmoid(x):
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

class FeatureVector (object):
    def __init__(self, x):
        self.data = sigmoid(cepstrum(x))

    def __getattr__(self, attr):
        if hasattr(self.data):
            return getattr(self.data)


