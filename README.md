# DIONE - Super Resolution using Sentinel-2 and Deimos imagery  
This repo contains code to train a multitemporal super-resolution model for Sentinel-2 imagery using Deimos.

You can find more information about this project in the blog post [Multi-temporal Super-Resolution on Sentinel-2 Imagery](https://medium.com/sentinel-hub/multi-temporal-super-resolution-on-sentinel-2-imagery-6089c2b39ebc)

## Introduction
This project is part of the DIONE project  where one of the missions is using novel techniques to  improve the capabilities of satellite technology while integrating various data sources, such as very high resolution imagery, to, for example, enable monitoring of smaller agricultural parcels through the use of super resolution models.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 870378.

Please visit the [Dione website](https://dione-project.eu/) for further information.
## Requirements
The super resolution pipeline uses SentinelHub service to download Sentinel-2 and Deimos imagery. Amazon AWS S3 bucket was used to store the data.   
_Deimos imagery is not public, however any other Very High Resolution imagery can be used by adjusting the general workflow._

## Installation and usage

To install the sr package, clone locally the repository, and from within the repository, run the following commands:
```
pip install -r requirements.txt
python setup.py install --user
```
Procedure is executed in notebooks, the basic functionality of each notebook is described below: 
* `00-parse-deimos-metadata.ipynb`: **Deimos specific**. Parses metadata for each ingested Deimos tile and saves to dataframe. 
* `00a-add-per-tile-median.ipynb` **Deimos specific**. Calculates median for each Deimos tile. 
* `00b-calculate-cloudfree-deimos-stats.ipynb` **Deimos specific**. Calculates Deimos tile statistics on cloudless areas. 
* `01-download-to-eopatches.ipynb` Download Sentinel-2 and ingested Deimos imagery to EOPatches
* `02a-add-clm-deimos.ipynb` Add cloud mask information to Deimos EOPatches
* `02b-add-clm-stats-to-patches.ipynb` Add cloudless normalization statistics to EOPatches 
* `03-sampling.ipynb` Sample smaller patchlets from EOPatches 
* `04-sampled-to-npz.ipynb` Construct NPZ files from patchlets. 
* `05a-train-test-split.ipynb` Split NPZ files into train/test/validation sets.. 
* `05b-find-cloudy-neighbours.ipynb` Shadow detection by filtering neighbours of cloudy EOPatches
* `05c-calculate-s2-normalizations.ipynb`  Calculate per country Sentinel-2 normalization statistics. 
* `06-train.ipynb` Model training. 
* `07-predict.ipynb`  Predict the model on smaller patchlets. 
* `07b-predict-eopatches.ipynb` Predict the model on whole EOPatches. 
