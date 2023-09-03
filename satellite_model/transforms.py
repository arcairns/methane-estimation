import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import random
import numpy as np
import torch
from rasterio.plot import reshape_as_image


class ToTensor(object):
    def __call__(self, sample):
        s2 = torch.from_numpy(sample["s2"].copy())
        ghgsat = torch.from_numpy(sample["ghgsat"].copy())

        if sample.get("s5p") is not None:
            s5p = torch.from_numpy(sample["s5p"].copy())

        out = {}
        for k,v in sample.items():
            if k == "s2":
                out[k] = s2
            elif k == "ghgsat": 
                out[k] = ghgsat
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v
        return out

class DatasetStatistics(object):
    def __init__(self):
        #s2
        self.channel_means = np.array([1628.9171809333334, 1819.9017041333334, 2171.1155995333334, 2450.8130693666667,
                                       2792.5486688, 3445.0945518, 3777.8394639333333, 3922.648574733333, 3993.7179329666665,
                                       4020.0622427, 4149.4285022, 3560.875527633333])

        self.channel_std = np.array([489.8216747671231, 580.7537756147514, 660.904810839969, 896.2297484234196, 839.8859249293796,
                                     838.9905210988723, 1036.5783085335959, 1043.6990900526894, 1031.3088954598848, 997.5078538616523,
                                     1044.218804321008, 1131.374559609031])

        # stats over the input s5p data
        self.s5p_mean = 1903.6368360933943
        self.s5p_std = 14.304698910027422

        # values across input ghghsat data
        self.ghgsat_mean = 0.73448247
        self.ghgsat_std = 28.815554

class Normalize(object):
    """normalize a sample, i.e. the image and CH4 value, by subtracting mean and dividing by std"""
    def __init__(self, statistics):
        self.statistics = statistics

    def __call__(self, sample):

        img = sample.get("s2").copy()
        img = reshape_as_image(img)
        img = np.moveaxis((img - self.statistics.channel_means) / self.statistics.channel_std, -1, 0)

        if sample.get("ghgsat") is not None:
            ghgsat = sample.get("ghgsat").copy()
            ghgsat = np.array((ghgsat - self.statistics.ghgsat_mean) / self.statistics.ghgsat_std)

        if sample.get("s5p") is not None:
            s5p = sample.get("s5p").copy()
            s5p = np.array((s5p - self.statistics.s5p_mean) / self.statistics.s5p_std)

        out = {}
        for k,v in sample.items():
            if k == "s2":
                out[k] = img
            elif k == "ghgsat":
                out[k] = ghgsat
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v
          
        return out

    @staticmethod
    def undo_ch4_standardization(statistics, ghgsat):
        return (ghgsat * statistics.ghgsat_std) + statistics.ghgsat_mean

