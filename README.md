##Setup
### 1) create folder structure
- create local directory LOCAL_DIR
- clone code to LOCAL_DIR
- add folders data and results in LOCAL_DIR
so the final structure is  
--LOCAL_DIR  
    |  
    -- data (contains the ground truth data sets)  
    |       
    -- code   
    |     
    -- results (synthetically degraded segmentation masks stored will be stored here)
### 2) create environment
- install packages from requirements
- install gurobi 9.1.1 (see help_gurobi.txt)

## run tracking

## Reproduce data sets
### 1) download data sets
- go to http://celltrackingchallenge.net
and download : Fluo-N2DH-SIM+ and FLuo-N3DH-SIM+ and unpack and save in data directory

### 2) run code
- run create_synth_segm_data.py to create synthetically 

