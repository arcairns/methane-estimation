import os
from re import S

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

import linecache
import os
import tracemalloc
from time import sleep

import torch
import torch.nn.functional as F
import random

import xarray as xr
import rioxarray

from train_utils import eval_metrics

def read_param_file(filepath):
    with open(filepath, "r") as f:
        output = f.read()
    return output

class PassthroughLoss:
    """use any normal torch loss function with
    the heteroscedastic network architecture.
    Simply ignores the second output."""
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, y, y_hat):
         # assumes that y_hat has two components, corresponding to [mean, sigma2]
        if len(y_hat.shape) == 2:
            ym = y_hat[:, 0]
            prec = y_hat[:, 1]
        elif len(y_hat.shape) == 1:
            ym = y_hat[0]
            prec = y_hat[1]
        else:
            raise ValueError("wrong y_hat shape: " + str(y_hat.shape))

        return self.loss(y, ym)

def heteroscedastic_loss(y, y_hat):
    # assumes that y_hat has two components, corresponding to [mean, sigma2]
    if len(y_hat.shape) == 2:
        ym = y_hat[:, 0]
        prec = y_hat[:, 1]
    elif len(y_hat.shape) == 1:
        ym = y_hat[0]
        prec = y_hat[1]
    else:
        raise ValueError("wrong y_hat shape: " + str(y_hat.shape))

    ymd = (ym - y.squeeze())
    sigma2 = torch.exp(-prec)

    l = (0.5 * (sigma2 * ymd * ymd)) + 0.5 * prec
    return l.sum()

def step(x, y, model, loss, optimizer, heteroscedastic, architecture):

    y_hat = model(x)
    if architecture == "CNN":
        y = np.reshape(y, 750000) #  reshape the target values for the CNN into one array rather than 2D array
    if architecture == "UNet":
        y_hat = F.interpolate(y_hat, size = (1000, 750), mode = 'nearest') # downsample predictions to 20m, to match y

    snapshot = tracemalloc.take_snapshot() # monitoring memory
##    display_top(snapshot) # uncomment this if you'd like to see which lines of code are taking up most memory at this point

    loss_epoch = loss(y.squeeze(), y_hat.squeeze())
    if heteroscedastic:
        if len(y_hat.shape) == 2:
            y_hat = y_hat[:, 0]
        elif len(y_hat.shape) == 1:
            y_hat = y_hat[0]
        else:
            raise ValueError("wrong y_hat shape:" + str(y_hat.shape))

    optimizer.zero_grad()
    loss_epoch.backward() # back propagation
    optimizer.step() # gradient descent

    if len(y.shape) == 0:
        y = y.unsqueeze(0)
    if len(y_hat.shape) == 0:
        y_hat = y_hat.unsqueeze(0)

    metric_results = eval_metrics(y.detach().cpu(), y_hat.squeeze().detach().cpu())

    return loss_epoch.detach().cpu(), metric_results

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(datadir, samples_file, sources):
    """load samples to memory, returns array of samples and array of stations
    each sample is a dict
    this version loads all samples from one station in one go (e.g. for multiple months), s.t. the S5P data for the station is only read once"""
    assert(sources in ["S2", "S2S5P"])

    if not isinstance(samples_file, pd.DataFrame): #checks if samples_file is a pd dataframe
        samples_df = pd.read_csv(samples_file, index_col="idx") #samples_df is the df in pandas
    else:
        samples_df = samples_file
    s5p_dates = None

    samples = [] # array with dictionary in it for each station. This has ghgsat and s5p data.
    stations = {} # dictionary with stations as keys and values the s2 data arrays
    
    try:
        for station in tqdm(samples_df.observation.unique()):  
            station_obs = samples_df[samples_df.observation == station] 
            
            for idx in station_obs.index.values:
                sample = samples_df.loc[idx].to_dict()
                sample["idx"] = idx 
                sample["s5p"] = np.load(os.path.join(datadir, "sentinel-5p", station, sample["s5p_path"]))
                sample["ghgsat"] = np.load(os.path.join(datadir, "ghgsat", station, sample["ghgsat_path"]))
                samples.append(sample)
                
                stations[sample["observation"]] = np.load(os.path.join(datadir, "sentinel-2", station, sample["s2_path"]))

    except IndexError as e:
        print(e)
        print("idx:", idx)
        print()

    return samples, stations

def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Memory Usage: Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    
