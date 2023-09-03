import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset

import xarray as xr
import rioxarray

class CH4PredictionDataset(Dataset):

    def __init__(self, datadir, samples, sources, transforms=None, station_imgs=None): 
        assert(sources in ["S2", "S2S5P"])

        self.datadir = datadir
        self.transforms = transforms
        self.sources = sources
        self.station_imgs = station_imgs 
        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
##        print("station_img:", self.station_imgs.get(sample["observation"]).shape)
        if self.station_imgs is not None:
            sample["s2"] = self.station_imgs.get(sample["observation"])
            
        if self.transforms:
            sample = self.transforms(sample)
##            print("after transforms shape:", sample["s2"].shape)

        return sample

    def __len__(self):
        return len(self.samples)

    def display_sample(self, sample, title=None):
        img = sample["s2"]
        band_data = self._normalize_for_display(img)

        if "S5P" in self.sources:
            fig, axs = plt.subplots(1, 2, figsize=(7,7))
            s2_ax = axs[0]
        else:
            fig, s2_ax = plt.subplots(1, figsize=(5,5))
        s2_ax.imshow(band_data[:, :, [3,2,1]])
        s2_ax.set_title("Sentinel2 data")

        if "S5P" in self.sources:
            im = axs[1].imshow(sample["s5p"])
            axs[1].set_title("Sentinel-5P data")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

        if title is not None:
            fig.suptitle(title)

        plt.show()

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)

