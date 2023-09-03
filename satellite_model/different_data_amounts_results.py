import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

base_path = "p"

#create csvs for each different amount from random stations
all_stations = pd.read_csv(os.path.join(base_path, "data", "samples_S2S5P_2018_2020_epa_91.csv"))
                           
data_amounts = [5,10,20,50,90]

for run in range(10):
    for amount in data_amounts:
        stations = all_stations.sample(n=amount)
        stations.to_csv(os.path.join(base_path, "data", "runsFromScratch", "samples_S2S5P_2018_2020_epa"+"_run_"+str(run)+"_amount_"+str(amount)+".csv"))

    
args = glob("/Users/amycairns/Desktop/Global-NO2-Estimation-main/data/runsFromScratch/*.csv")

for arg in args:
    file_end = str(arg.split("_")[6])+"_"+str(arg.split("_")[8])
    print(file_end)
    if len(glob(os.path.join(base_path, "satellite_model/results", "*"+file_end))) == 0:
        os.system("python training.py --samples_file {}".format(arg))
    while len(glob(os.path.join(base_path, "satellite_model/results", "*"+file_end))) == 0:
        print(glob(os.path.join(base_path, "satellite_model/results", "*"+file_end)))
        print(os.path.join(base_path, "satellite_model/results", "*"+file_end))
        time.sleep(60)
    
        
# plot the result depending on data size
results = glob("/Users/amycairns/Desktop/Global-NO2-Estimation-main/satellite_model/results/test_*.csv")

result_df = []

for result in results:
    # get the amount of data from the filename
    amount = result.split("_")[9].replace(".csv", "")
    run = result.split("_")[8].replace(".csv", "")
    #append data to list
    df = pd.read_csv(result)
    df["amount"] = int(amount)
    df["run"] = int(run)
    result_df.append(df)

result_df = pd.concat(result_df)
print(result_df)

summary_df = result_df.groupby(['amount'])['mae', 'mse', 'r2'].agg(['mean','median', 'std', 'sem']) # need to fix so it copes with NANs for the 5 value
summary_df.columns = summary_df.columns.map('_'.join)

print("summary_df")
print(summary_df)

print(summary_df.index)

print(summary_df["mae_mean"])

plt.figure(figsize = (10,8))

boxplot_data, amounts = [], [5,10,20,50,90]

for amount in amounts:
    boxplot_amount = list(result_df[result_df['amount'] == amount]['mse'])
    boxplot_data.append(boxplot_amount)

fig, ax = plt.subplots()
ax.boxplot(boxplot_data, names=amounts)
plt.show()
