# methane-estimation
This project uses remote sense data for a pixel-wise problem to estimating methane. This expands on a study by Scheibenreif, Mommert and Borth (2022) by outputting dense gas predictions from the model, investigating dissertation architectures and parameters, and implemented a U-Net architecture. This is a MSc dissertation project at the University of Bath. 

Pre-processing data pipeline
- This code allows you to download Sentinel-5P and Sentinel-2 data, crop, resample and apply data augmentation.
- GHGSat data was made available through ESA's EarthNet project ()
- The folder structure is:
- ------- location1
- ------------observation1
- ----------------ghgsat
- ----------------sentinel-2
- ----------------sentinel-5P
  
Model code
-  The model requires a csv file which includes the filenames different sources.
- The model requires folder structure:
- --------------ghgsat
- -------------------observation1
- -------------------observationn
- --------------sentinel-2
- -------------------observation1
- -------------------observationn
- --------------sentinel-5P
- -------------------observation1
- -------------------observationn

L. Scheibenreif, M. Mommert and D. Borth, "Toward Global Estimation of Ground-Level NO2 Pollution With Deep Learning and Remote Sensing," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 4705914, doi: 10.1109/TGRS.2022.3160827.
