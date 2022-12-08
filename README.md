# A graph-based cell tracking algorithm with few manually tunable parameters and automated segmentation error correction

<img src="https://user-images.githubusercontent.com/28811849/204012282-81ab2be9-22f1-45a6-ad04-f78b4b7751fa.png" width=70% height=70%>
Source: Löffler et al. (2022), doi: https://doi.org/10.1371/journal.pone.0249257.g001 
License: CC BY 4.0


---

This repository contains the code to the publication:  

 "A graph-based cell tracking algorithm with few manually tunable parameters and automated segmentation error correction"
Löffler K, Scherr T, Mikut R 
(2021)
PLOS ONE 16(9): e0249257. https://doi.org/10.1371/journal.pone.0249257

**The deep learning based segmentation approach which is used together with the graph-based tracking for submission to the Cell Tracking Challenge as team KIT-Sch-GE(2) (renamed to [KIT-GE(3)](http://celltrackingchallenge.net/participants/KIT-GE/)) is mainted at https://github.com/TimScherr/KIT-GE-3-Cell-Segmentation-for-CTC .**

The code has been tested on Windows and Linux using Python 3.8.

## Setup
### 1) Create folder structure
- create a project directory LOCAL_DIR
- create two folders named data and results in LOCAL_DIR
- clone the code and install dependencies:
```
conda create --name venv_graph_tracking_kit_sch_ge_2021 python==3.8
conda activate venv_graph_tracking_kit_sch_ge_2021
git clone https://github.com/kaloeffler/KIT-GE-3-Cell-Tracking-for-CTC.git
pip install -e ./2021-cell-tracking
```
so the final structure is  
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
└───2021-cell-tracking (contains our tracking code)
└───results (synthetically degraded segmentation masks will be stored here)
```

### 2) Install gurobi
see help_gurobi.txt

## Run tracking
- the tracking algorithm can be used with any 2D/3D image data with a segmentation which needs to be provided by the user
- it is assumed that the image data and segmentation data have a similar naming convention as used by the cell tracking challenge (http://celltrackingchallenge.net)
```
python -m run_tracking --image_path IMAGE_PATH --segmentation_path SEGMENTATION_PATH --results_path RESULTS_PATH

```

## Reproduce synthetic data sets
### 1) Download data sets
- go to http://celltrackingchallenge.net
and download the training data sets Fluo-N2DH-SIM+ and Fluo-N3DH-SIM+, unpack and save in data directory:
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
│   └───Fluo-N2DH-SIM+
│   └───Fluo-N3DH-SIM+
└───2021-cell-tracking
└───results (synthetically degraded segmentation masks will be stored here)
```

### 2) Run code
- run create_synth_segm_data.py to create synthetically degraded segmentation mask images
```
python -m create_synth_segm_data
```

### Citation
If you use our work in your research, please cite:

```bibtex
@article{loeffler2021,
    doi = {10.1371/journal.pone.0249257},
    author = {Löffler, Katharina and Scherr, Tim and Mikut, Ralf},
    journal = {PLOS ONE},
    title = {A graph-based cell tracking algorithm with few manually tunable parameters and automated segmentation error correction},
    year = {2021},
    volume = {16},
    pages = {1-28},
    number = {9},
}
```

