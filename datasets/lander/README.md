# Lander Dataset

## Introduction
This example dataset demonstrates mapped landmark (ML) capabilities. To run, inside the src folder:
```python ../examples/run_lander.py --log_level info --start_timestamp 673000000000 --data_folder ../datasets/lander```

## Dataset Directory Structure
* cam0: Camera ML matches. Single data.csv containing timestamps and corresponding file in map folder
* gt: Ground truth. Single data.csv containing timestamps, xyz positions, quaternions.
* imu0: IMU data (simulated in this example). Single data.csv containing timestamps, xyz angular rates (radians per sec), xyz linear accelerations (m per sec squared).
* map: One csv file for every camera update, each containing a list of absolute (map) landmark match coordinates keyed by pixel coordinates.
* util: Python script for generating map csv files containing landmark matches. Make sure to download prerequisite data files first by running ```cmake . && make``` inside the folder (**gdal required**). Then run ```python moonmatch.py```. Input: camera images inside ```frames``` folder (examples provided). Output: ```*.matches.csv``` files that can go in ```../map```
