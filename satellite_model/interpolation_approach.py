
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
import os

from transforms import DatasetStatistics
from train_utils import eval_metrics

def Normalize(object, source):
	stats = DatasetStatistics()
	if source == "ghgsat":
		ghgsatNorm = np.array((object - stats.ghgsat_mean) / stats.ghgsat_std)
		return ghgsatNorm
	if source == "s5p":
		s5pNorm = np.array((object - stats.s5p_mean) / stats.s5p_std)
		return s5pNorm



# dict for metrics
results = {"observation": [], "r2": [], "mae": [], "mse": [], "s5p_mean": [], "ghgsat_mean": [], "s5p_median": [], "ghgsat_median": []}

base_path = '/Volumes/Expansion/California-CH4-Estimation/data/'

for obs in os.listdir(base_path):
	print(obs)
	print(os.listdir(base_path))
	s5p = np.load(os.path.join(base_path, obs, 'low_res_methane.npy'))
	ghgsat = np.load(os.path.join(base_path, obs, 'flag_mask_20m_high_res_methane.npy'))

	#downsample s5p to 20m resolution to match the same resolution as ghgsat
	print("s5p shape before transformations: ", s5p.shape)
	s5p = [[s5p]]
	s5p = F.interpolate(torch.Tensor(s5p), size = (1000, 750), mode = 'nearest')
	s5p = s5p.squeeze().numpy()
	print("s5p shape after transformations: ", s5p.shape)

	plt.figure(figsize=(12,7))
	#plt.xlim([-50,50]), plt.ylim([-500,500])
	#plt.plot((-500,50),(-50,50), "r-")
	plt.xlabel("GHGSat"), plt.ylabel("Raw s5p")
	plt.scatter(ghgsat, s5p, s=2)
	plt.title("Distribution of GHGSat and S5P methane values")
	plt.savefig(os.path.join(base_path, obs,'raw_comparison.png'))
	plt.show()

	#normalise
	s5p = Normalize(s5p, "s5p")
	ghgsat = Normalize(ghgsat, "ghgsat")

	results["observation"].append(obs)
	results["s5p_mean"].append(np.mean(s5p))
	results["ghgsat_mean"].append(np.mean(ghgsat))
	results["s5p_median"].append(np.median(s5p))
	results["ghgsat_median"].append(np.median(ghgsat))

	# calculate r2 and mse loss
	s5p, ghgsat = s5p.flatten(), ghgsat.flatten()
	eval = eval_metrics(s5p, ghgsat)
	print(eval)

	results["r2"].append(eval[0])
	results["mae"].append(eval[1])
	results["mse"].append(eval[2])

	# graph of normalised ghgsat and normalised s5p data
	plt.figure(figsize=(12,7))
	plt.xlim([-50,50]), plt.ylim([-50,50])
	plt.plot((-50,50),(-50,50), "r-")
	plt.xlabel("Normalized GHGSat"), plt.ylabel("Normalized S5P")
	plt.scatter(ghgsat, s5p, s=2)
	plt.title("Comparison of normalized GHGSat and S5P methane values")
	plt.savefig(os.path.join(base_path, obs,'normalized_comparison.png'))
	plt.show()

# save the metrics for each observation
results_df = pd.DataFrame(results)
results_df.to_csv("metrics.csv", index=False)