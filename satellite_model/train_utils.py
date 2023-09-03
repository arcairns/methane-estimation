import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import sys
import copy
from tqdm  import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CH4PredictionDataset
from transforms import ToTensor, DatasetStatistics, Normalize
from model import get_model


def eval_metrics(y, y_hat):
    if len(y.shape) == 3:
        y = y.flatten()
        y_hat = y_hat.flatten()
        
    elif len(y.shape) != 1:
        raise ValueError()

    r2 = r2_score(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)

    return [r2, mae, mse]


def split_samples(samples, stations, test_size=0.2, val_size=0.2, seed=1):

    stations_train, stations_test = train_test_split(stations, test_size=test_size, shuffle=True, random_state = seed)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size,shuffle=True, random_state = seed))
    stations_test = set(stations_test)

    samples_train = [s for s in samples if s["observation"] in stations_train]
    samples_test = [s for s in samples if s["observation"] in stations_test]
    samples_val = [s for s in samples if s["observation"] in stations_val]

    print("Number of stations train:",len(stations_train))
    print("Number of samples train:", len(samples_train))
            
    return samples_train, samples_val, samples_test, stations_train, stations_val, stations_test

def test(sources, model, dataloader, device, datastats, dropout, heteroscedastic):
    model.eval()
    if dropout:
        model.head.turn_dropout_on(use=False)
    predictions = []
    measurements = []

    with torch.no_grad():
        if "S5P" in sources:
            for idx, sample in enumerate(dataloader):
                s2 = sample["s2"].float().to(device) 
                s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                y = sample["ghgsat"].float().to(device).squeeze()
                model_input = {"s2" : s2, "s5p" : s5p}
                y_hat = model(model_input).squeeze()
                
                if heteroscedastic:
                    if len(y_hat.shape) == 2:
                        y_hat = y_hat[:, 0]
                    elif len(y_hat.shape) == 1:
                        y_hat = y_hat[0]
                    else:
                        raise ValueError() 
                measurements.append(y.cpu().numpy())
                predictions.append(y_hat.cpu().numpy())
        else:
            for idx, sample in enumerate(dataloader):
                s2 = sample["s2"].float().to(device)
                y = sample["ghgsat"].float().to(device).squeeze()

                y_hat = model(s2).squeeze() 
                if heteroscedastic:
                    if len(y_hat.shape) == 2:
                        y_hat = y_hat[:, 0]
                    elif len(y_hat.shape) == 1:
                        y_hat = y_hat[0]
                    else:
                        raise ValueError()
                
                measurements.append(y.cpu().numpy().item())
                predictions.append(y_hat.cpu().numpy().item())

    measurements = np.array(measurements)
    predictions = np.array(predictions)

    return measurements, predictions
