import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# reproduce the model by Scheibenreif, Mommert and Borth with 5-90 images, compared to the 3000 images that were used in their paper, to investigate how well the CNN performs.
base_path = "path" # *** path needed ***

create csvs for each different amount from random stations
all_stations = pd.read_csv(os.path.join(base_path, "data", "samples_S2S5P_2018_2020_eea.csv"))
                           
data_amounts = [3100] #[5,10,20,50,90]

for run in range(10):
    for amount in data_amounts:
        stations = all_stations.sample(n=amount)
        stations.to_csv(os.path.join(base_path, "data", "eeaFromScratchNoDataAug", "samples_S2S5P_2018_2020_eaa"+"_run_"+str(run)+"_amount_"+str(amount)+".csv"))
    
args = glob("eeaPretrainedNoDataAug/*3100.csv")
i = 0
for arg in args:
    print(i)
    i += 1
    file_end = str(arg.split("_")[6])+"_"+str(arg.split("_")[8])
    results_path = "satellite_model/results/eeaPretrainedNoDataAug"
    print("file end",file_end)
    if len(glob(os.path.join(base_path, results_path, "*"+file_end))) == 0:
        os.system("python training.py --samples_file {}".format(arg))
    while len(glob(os.path.join(base_path, results_path, "*"+file_end))) == 0:
        print(glob(os.path.join(base_path, results_path, "*"+file_end)))
        print(os.path.join(base_path, results_path, "*"+file_end))
        time.sleep(60)

# plot the result depending on data size
scratch = glob("/path/test_*.csv") # *** path needed ***
pretrained = glob("/path/test_*.csv") # *** path needed ***

pretrained_df, scratch_df = [], []

for result in pretrained:
    print(result)
    # get the amount of data from the filename
    amount = result.split("_")[9].replace(".csv", "")
    
    run = result.split("_")[8].replace(".csv", "")
    #append data to list
    df = pd.read_csv(result)
    print(df)
    df["amount"] = int(amount)
    print(amount)
    df["run"] = int(run)
    pretrained_df.append(df)

print(pretrained_df)
pretrained_df = pd.concat(pretrained_df)
print(pretrained_df)

for result in scratch:
    print(result)
    # get the amount of data from the filename
    amount = result.split("_")[10].replace(".csv", "")
    run = result.split("_")[9].replace(".csv", "") 
    #append data to list
    df = pd.read_csv(result)
    df["amount"] = int(amount)
    df["run"] = int(run)
    scratch_df.append(df)

scratch_df = pd.concat(scratch_df)

summary_df = result_df.groupby(['amount'])['mae', 'mse', 'r2'].agg(['mean','median', 'std', 'sem']) # need to fix so it copes with NANs for the 5 value
summary_df.columns = summary_df.columns.map('_'.join)
summary_df = summary_df.reset_index(drop=True)
print(summary_df)

pretrained_data, scratch_data, amounts = [], [], [5,10,20,50,90]

for amount in amounts:
    boxplot_amount = list(pretrained_df[pretrained_df['amount'] == amount]['mae'])
    pretrained_data.append(boxplot_amount)
print(pretrained_data)

for amount in amounts:
    boxplot_amount = list(scratch_df[scratch_df['amount'] == amount]['mae'])
    scratch_data.append(boxplot_amount)
print(scratch_data)

def set_box_color(bp, color):
    for item in ['boxes', 'whiskers', 'medians', 'caps']:
        plt.setp(bp[item], color=color)
    plt.setp(bp["fliers"], markeredgecolor=color)

plt.figure(figsize=(12,6))
box1_amounts = [item - 0.6 for item in amounts]
box2_amounts = [item + 0.6 for item in amounts]

box1 = plt.boxplot(scratch_data, positions=box1_amounts, widths=0.6)
box2 = plt.boxplot(pretrained_data, positions=box2_amounts, widths=0.6)
set_box_color(box1, '#12436D') 
set_box_color(box2, '#F46A25')

plt.plot([], c='#12436D', label='From scratch')
plt.plot([], c='#F46A25', label='Pretrained')
plt.legend(fontsize=12)
plt.xticks(amounts, amounts)
plt.title('NO2 model trained on differents amount of data', fontsize=12)
plt.xlabel('Amount of data', fontsize=12)
plt.ylabel('Mean absolute error', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("EEA_data.pdf")
plt.show()

# plot results from testing on epa data, different format as stored in two spreadsheets rather than individual files
epa_scratch = pd.read_csv("results")
epa_scratch["amount"] = [int(name.replace(".model","").split("_")[-1]) for name in epa_scratch["model"]]

epa_pretrained = pd.read_csv("pretrained_results")
epa_pretrained["amount"] = [int(name.replace(".model","").split("_")[-1]) for name in epa_pretrained["model"]]

plt.figure(figsize=(12,6))

for amount in amounts:
    boxplot_amount = list(epa_scratch[epa_scratch['amount'] == amount]['mae'])
    scratch_data.append(boxplot_amount)
print(scratch_data)

for amount in amounts:
    boxplot_amount = list(epa_pretrained[epa_pretrained['amount'] == amount]['mae'])
    pretrained_data.append(boxplot_amount)
print(pretrained_data)

box1 = plt.boxplot(scratch_data, positions=box1_amounts, widths=0.6)
box2 = plt.boxplot(pretrained_data, positions=box2_amounts, widths=0.6)
set_box_color(box1, '#12436D') 
set_box_color(box2, '#F46A25')

plt.plot([], c='#12436D', label='From scratch')
plt.plot([], c='#F46A25', label='Pretrained')
plt.legend(fontsize=12)
plt.xticks(amounts, amounts)
plt.title('Results of NO2 models trained on different amounts of data and tested on EPA data', fontsize=12)
plt.xlabel('Amount of data', fontsize=12)
plt.ylabel('Mean absolute error', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("EPA_data.pdf")


# plot actuals vs predicted on epa data
##actuals = pd.read_csv("/results/inferenceOnEpa/actuals.csv")
##epa_result_path_50 = glob("/results/inferenceOnEpa/fromScratchResults/*50.model.csv")
##epa_result_path_90 = glob("/results/inferenceOnEpa/fromScratchResults/*90.model.csv")
##epa_result_paths = epa_result_path_90 + epa_result_path_50
##
##epa_results = []
##print(epa_result_paths)
##for path in epa_result_paths:
##    epa_results.append(pd.read_csv(path))
##
##
##plt.figure(figsize=(10,8))
##for predictions in epa_results:
##    plt.scatter(actuals, predictions, c="black", s=8)
##plt.plot([0,200], [0,200], "r-")
##plt.xlim([0,70])
##plt.ylim([0,200])
##plt.title(r'Comparison of NO$_2$ measurements and predictions', fontsize=12)
##plt.xlabel(r'NO$_2$ measurement [$\mu g/m^3$]', fontsize=12)
##plt.ylabel(r'NO$_2$ estimate [$\mu g/m^3$]', fontsize=12)
##plt.savefig("EPA_estimate_prediction.pdf")
##plt.show()

