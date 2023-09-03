import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import sys
import copy
import random
import argparse
from datetime import datetime
from distutils.util import strtobool

import mlflow
import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import tracemalloc

from dataset import CH4PredictionDataset
from transforms import ToTensor, DatasetStatistics, Normalize
from model import get_model 
from utils import load_data, none_or_true, dotdict, set_seed, step, heteroscedastic_loss, PassthroughLoss
from train_utils import eval_metrics, split_samples, test

bool_args = ["verbose",
            "early_stopping",
            "heteroscedastic",
            ]

if __name__ == '__main__':

    tracemalloc.start()
    
    parser = argparse.ArgumentParser(description='train_s2s5p_model')

    # parameters
    parser.add_argument('--samples_file', default="path_to_samples_file", type=str) # *** path needed, cannot have underscore in the path before the samples file ***
    parser.add_argument('--datadir', default="path_to_data_directory", type=str) # *** path needed ***
    parser.add_argument('--verbose', default="True", type=str)
    parser.add_argument('--epochs', default=1, type=int) 
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--result_dir', default="results/", type=str)
    parser.add_argument('--checkpoint', default=None, type=none_or_true)
    parser.add_argument('--early_stopping', default="False", type=str)
    parser.add_argument('--weight_decay_lambda', default=0.001, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=None, type=none_or_true)
    parser.add_argument('--dropout_p_second_to_last_layer', default=0.0, type=float)
    parser.add_argument('--dropout_p_last_layer', default=0.0, type=float)
    parser.add_argument('--heteroscedastic', default="False", type=str)
    parser.add_argument('--architecture', default="CNN", type=str) # CNN or UNet

    torch.set_num_threads(1)
    args = parser.parse_args()
    config = dotdict({k : strtobool(v) if k in bool_args else v for k,v in vars(args).items()})

    sources = config.samples_file.split("_")[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    checkpoint_name = "pretrained" if config.checkpoint is not None else "from_scratch" 

    experiment = "_".join([datetime.today().strftime('%Y-%m-%d-%H:%M'), sources, checkpoint_name])
    if config.verbose: print("Initializing mlflow experiment:", experiment)

    try: 
        experiment_id = mlflow.create_experiment(experiment)
    except:
        experiment_id = dict(mlflow.get_experiment_by_name(experiment))['experiment_id']

    if config.verbose:
        print("Samples file: ", config.samples_file)
        print("Data directory: ",config.datadir)
        print("Soruces: ",sources)
        print("Checkpoint: ",config.checkpoint)
        print("Device: ",device)
        print("Model architecture: ",config.architecture)
        print("Start loading samples...")

    samples, stations = load_data(config.datadir, config.samples_file, sources) 

    if config.heteroscedastic:
        msel = nn.MSELoss() # MAE loss can be achieve with nn.L1Loss()
        loss = PassthroughLoss(msel)
        print("Start heteroscedastic model training with MSELoss")
    else:
        loss = nn.MSELoss() 
        
    datastats = DatasetStatistics()
    tf = transforms.Compose([Normalize(datastats),  ToTensor()])

    performances_test = []
    performances_val = []
    performances_train = []

    for run in tqdm(range(1, config.runs+1), unit="run"):

        # fix a different seed for each run
        seed = run

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("samples_file", config.samples_file)
            mlflow.log_param("heteroscedastic", config.heteroscedastic)
            mlflow.log_param("datadir", config.datadir)
            mlflow.log_param("sources", sources)
            mlflow.log_param("batch_size", config.batch_size)
            mlflow.log_param("result_dir", config.result_dir)
            mlflow.log_param("pretrained_checkpoint", config.checkpoint)
            mlflow.log_param("device", device)
            mlflow.log_param("early_stopping", config.early_stopping)
            mlflow.log_param("learning_rate", config.learning_rate)
            mlflow.log_param("run", run)
            mlflow.log_param("dropout", config.dropout)
            mlflow.log_param("dropout_p_second_to_last_layer", config.dropout_p_second_to_last_layer)
            mlflow.log_param("dropout_p_last_layer", config.dropout_p_last_layer)
            mlflow.log_param("weight_decay", config.weight_decay_lambda)
            mlflow.log_param("epochs", config.epochs)
            mlflow.log_param("seed", seed)
            mlflow.log_param("architecture", config.architecture)

            # set the seed for this run
            set_seed(seed)

            if config.dropout:
                dropout_config = {
                        "p_second_to_last_layer" : config.dropout_p_second_to_last_layer,
                        "p_last_layer" : config.dropout_p_last_layer,
                        }
            else:
                dropout_config = None

            # initialize dataloaders + model
            if config.verbose: print("Initializing dataset")
            samples_train, samples_val, samples_test, stations_train, stations_val, stations_test = split_samples(samples, list(stations.keys()), 0.2, 0.2, seed)
            if config.verbose: print("First stations_train:", list(stations_train)[:10])
            dataset_test = CH4PredictionDataset(config.datadir, samples_test, sources, transforms=tf, station_imgs=stations) 
            dataset_train = CH4PredictionDataset(config.datadir, samples_train, sources, transforms=tf, station_imgs=stations) 
            dataset_val = CH4PredictionDataset(config.datadir, samples_val, sources, transforms=tf, station_imgs=stations) 
            dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, num_workers=4, shuffle=True, pin_memory=False)
            dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
            dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
            dataloader_train_for_testing = DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)

            if config.verbose: print("Initializing model")
            model = get_model(sources, config.architecture, device, config.checkpoint, dropout=dropout_config, heteroscedastic=config.heteroscedastic)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay_lambda)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, threshold=1e6, min_lr=1e-7, verbose=True)

            if config.verbose: print("Start training")

            # train the model
            train_by_epoch = {"epoch": [], "r2": [], "mae": [], "mse": []} 
            for epoch in range(config.epochs):
                model.train() # set the module to training mode
                if config.dropout:
                    model.head.turn_dropout_on()

                loss_history = []
                loss_epoch = []
                r2_epoch = []
                mae_epoch = []
                mse_epoch = []

                if epoch == 5 and config.heteroscedastic:
                    if config.verbose: print("Changing to heteroscedastic loss...")
                    loss = heteroscedastic_loss

                for idx, sample in enumerate(dataloader_train):
                    model_input = sample["s2"].float().to(device)
                    if "S5P" in sources:
                        s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                        model_input = {"s2" : model_input, "s5p" : s5p}
                        
                    y = sample["ghgsat"].float().to(device)
                    
                    loss_batch, metric_results = step(model_input, y, model, loss, optimizer, config.heteroscedasti, config.architecture)
                    loss_epoch.append(loss_batch.item())
                    r2_epoch.append(metric_results[0])
                    mae_epoch.append(metric_results[1])
                    mse_epoch.append(metric_results[2])

                loss_epoch = np.array(loss_epoch).mean()
                r2_train_epoch = np.array(r2_epoch).mean()
                mae_train_epoch = np.array(mae_epoch).mean()
                mse_train_epoch = np.array(mse_epoch).mean()

                train_by_epoch["epoch"].append(epoch)
                train_by_epoch["r2"].append(r2_train_epoch)
                train_by_epoch["mae"].append(mae_train_epoch)
                train_by_epoch["mse"].append(mse_train_epoch)
                    
                scheduler.step(loss_epoch)
                torch.cuda.empty_cache()
                loss_history.append(loss_epoch)

                val_y, val_y_hat = test(sources, model, dataloader_val, device, datastats, config.dropout, config.heteroscedastic)

                if config.architecture == "UNet":
                    val_y_hat = np.array([[val_y_hat]])
                    val_y_hat = F.interpolate(torch.Tensor(val_y_hat), size = (int(val_y_hat.shape[2]), 1000, 750), mode = 'nearest')
                    val_y_hat = val_y_hat.squeeze().numpy()

                valid_val = (val_y_hat < 60) & (val_y_hat > -60)
                eval_val = eval_metrics(val_y, val_y_hat)

                if config.verbose: print("Fraction of valid estimates:", sum(valid_val)/len(valid_val)) 

                if config.early_stopping:
                    # stop training if evaluation performance does not increase
                    if epoch > 25 and sum(valid_val) > len(valid_val) - 5:
                        if eval_val[0] > np.mean([performances_val[-3][2], performances_val[-2][2], performances_val[-1][2]]):
                            # performance on evaluation set is decreasing
                            if config.verbose: print(f"Early stop at epoch: {epoch}")
                            mlflow.log_param("early_stop_epoch", epoch)
                            break

                if config.verbose: print(f"Epoch: {epoch}, {eval_val}")
                performances_val.append([run, epoch] + eval_val)

                mlflow.log_metrics({"val_r2_epoch" : eval_val[0], "val_mae_epoch" : eval_val[1], "val_mse_epoch" : eval_val[2]}, step=epoch)
                mlflow.log_metrics({"train_loss_epoch" : loss_epoch, "train_r2_epoch" : r2_train_epoch, "train_mae_epoch" : mae_train_epoch, "train_mse_epoch" : mae_train_epoch}, step=epoch)
                mlflow.log_metric("current_epoch", epoch, step=epoch)
                
            output_name = "_".join([sources, str(checkpoint_name), str(config.epochs), "epochs"]).replace(".csv", "")
            if config.verbose: print("output name" + output_name)
            torch.save(model.state_dict(), "artifacts/model_state/"+output_name+".model")
            for name, station_list in [("stations_train", stations_train), ("stations_val", stations_val), ("stations_test", stations_test)]:
                # save the dataset train/val/test split
                with open("artifacts/test_split/"+name+output_name+".txt", "w") as f:
                    for station in station_list:
                        f.write("%s\n" % station)

            test_y, test_y_hat = test(sources, model, dataloader_test, device, datastats, config.dropout, config.heteroscedastic)
            train_y, train_y_hat = test(sources, model, dataloader_train_for_testing, device, datastats, config.dropout, config.heteroscedastic)

            if config.architecture == "UNet":
                test_y_hat = np.array([[test_y_hat]])
                train_y_hat = np.array([[train_y_hat]])
                test_y_hat = F.interpolate(torch.Tensor(test_y_hat), size = (int(test_y_hat.shape[2]), 1000, 750), mode = 'nearest')
                test_y_hat = test_y_hat.squeeze().numpy()
                train_y_hat = F.interpolate(torch.Tensor(train_y_hat), size = (int(train_y_hat.shape[2]), 1000, 750), mode = 'nearest') 
                train_y_hat = train_y_hat.squeeze().numpy()

            eval_test = eval_metrics(test_y, test_y_hat)
            eval_train = eval_metrics(train_y, train_y_hat)
                
            # save img of predictions as artifact
            img, (ax1,ax2) = plt.subplots(1,2, figsize=(12,7))
            for ax in (ax1,ax2):
                ax.set_xlim((-50,50))
                ax.set_ylim((-50,50))
                ax.plot((-50,50),(-50,50), "r-")
                ax.set_xlabel("Measurements")
                ax.set_ylabel("Predictions")
            ax1.scatter(train_y, train_y_hat, s=2)
            ax1.set_title("train")
            ax2.scatter(test_y, test_y_hat, s=2)
            ax2.set_title("test")
            
            plt.savefig(os.path.join("artifacts", "predictions_"+output_name+".png"))

            mlflow.log_metric("test_r2", eval_test[0]) 
            mlflow.log_metric("test_mae", eval_test[1])
            mlflow.log_metric("test_mse", eval_test[2])

            performances_test.append(eval_test)
            performances_train.append(eval_train)

        performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2", "mae", "mse"])
        performances_test = pd.DataFrame(performances_test, columns=["r2", "mae", "mse"])
        performances_train = pd.DataFrame(performances_train, columns=["r2", "mae", "mse"])

        # save image of learning over time and overfitting as artifact, AC added this
        img, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2, figsize=(12,17))
        for ax in (ax1,ax2,ax3, ax4):
            ax.set_xlim((0,50))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Predictions")
        ax1.set_title("log MAE")
        ax1.set_yscale('log')
        ax1.scatter(train_by_epoch["epoch"], train_by_epoch["mae"], s=4, c="red", label = "train")
        ax1.plot(train_by_epoch["epoch"], train_by_epoch["mae"], c="red")
        ax1.scatter(performances_val["epoch"], performances_val["mae"], s=4, c="blue", label = "validation")
        ax1.plot(performances_val["epoch"], performances_val["mae"],c="blue")

        ax2.set_title("MAE")
        ax2.scatter(train_by_epoch["epoch"], train_by_epoch["mae"], s=4, c="red", label = "train")
        ax2.plot(train_by_epoch["epoch"], train_by_epoch["mae"], c="red")
        ax2.scatter(performances_val["epoch"], performances_val["mae"], s=4, c="blue", label = "validation")
        ax2.plot(performances_val["epoch"], performances_val["mae"],c="blue")
                        
        ax3.set_title("log MSE")
        ax3.set_yscale('log')
        ax3.scatter(train_by_epoch["epoch"], train_by_epoch["mse"], s=4, c="red")
        ax3.plot(train_by_epoch["epoch"], train_by_epoch["mse"], c="red")
        ax3.scatter(performances_val["epoch"], performances_val["mse"], s=4, c="blue")
        ax3.plot(performances_val["epoch"], performances_val["mse"], c="blue")
        
        ax4.set_title("R2")
        ax4.scatter(train_by_epoch["epoch"], train_by_epoch["r2"], s=4, c="red")
        ax4.plot(train_by_epoch["epoch"], train_by_epoch["r2"], c="red")
        ax4.scatter(performances_val["epoch"], performances_val["r2"], s=4, c="blue")
        ax4.plot(performances_val["epoch"], performances_val["r2"], c="blue")
        ax1.legend()
        
        plt.savefig(os.path.join("artifacts", "epochs_"+output_name+".pdf"))

    if config.checkpoint is not None:
        checkpoint_name = config.checkpoint.split("/")[1].split(".")[0]

    # save the model
    if config.verbose: print("Writing model...")
    torch.save(model.state_dict(), output_name+".model")
    if config.verbose: print("Writing model...")
    torch.save(model.state_dict(), output_name+".model")
    
    # save results
    if config.verbose: print("Writing results...")

    performances_test.to_csv(os.path.join(config.result_dir, "test_"+output_name+".csv"), index=False)
    performances_train.to_csv(os.path.join(config.result_dir, "train_"+output_name+".csv"), index=False)
    performances_val.to_csv(os.path.join(config.result_dir, "val_"+output_name+".csv"), index=False)

    if config.architecutre == "UNet":
        test_y_hat = test_y_hat.flatten()
    test_results_df = pd.DataFrame()
    actuals, estimates = [], []
    for idx, predictions in enumerate(test_y_hat):
        actuals = actuals + list(test_y[idx].flatten())
        estimates = estimates + list(predictions)

    test_results_df["actuals"] = actuals
    test_results_df["estimates"] = estimates

    test_results_df.to_csv(os.path.join(config.result_dir, "test_estimates_"+output_name+".csv"), index=False)
    if config.verbose: print("done.")
