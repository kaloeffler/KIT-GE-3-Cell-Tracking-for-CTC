This repository contains the code to the publication:  
 A graph-based cell tracking algorithm with few manually tunable parameters and automated segmentation error correction  
Katharina Löffler, Tim Scherr, Ralf Mikut  
bioRxiv 2021.03.16.435631; doi: https://doi.org/10.1101/2021.03.16.435631  

## Setup
### 1) create folder structure
- create local directory LOCAL_DIR
- clone code to LOCAL_DIR
- add folders data and results in LOCAL_DIR
so the final structure is  
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
└───code
└───results (synthetically degraded segmentation masks will be stored here)
```

### 2) create environment
- install packages from requirements
- install gurobi 9.1.1 (see help_gurobi.txt)

## run tracking
- the tracking algorithm can be used with any 2D/3D image data with a segmentation which needs to be provided by the user
- it is assumed that the image data and segmentation data have a similar naming convention as used by the cell tracking challenge (http://celltrackingchallenge.net)


## Reproduce synthetic data sets
### 1) download data sets
- go to http://celltrackingchallenge.net
and download : Fluo-N2DH-SIM+ and FLuo-N3DH-SIM+ and unpack and save in data directory:
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
│   └───Fluo-N2DH-SIM+
│   └───Fluo-N3DH-SIM+
└───code
└───results (synthetically degraded segmentation masks stored will be stored here)
```

### 2) run code
- run create_synth_segm_data.py to create synthetically degraded segmentation mask images 

