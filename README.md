# Gradient-based Iterative Asteroid Detection in TESS

A pipeline for detecting asteroids (faint-moving objects). We refer to our paper ([**pdf**](https://iopscience.iop.org/article/10.3847/1538-3881/adea3c/pdf)) associated with this Python code


## Running This Pipeline

- For the pre-processing steps, run
```
python 0_read_tess_fits_data.py
python 1_flatfield_destrap_dva.py
python 2_bright_asteroid_reduction.py
```

- To detect asteroids in a 16 pixel * 16 pixel sized patch, run
```
python 3_iterative_detection.py 512 1024
```

Alternatively, to detect asteroids in 2048 pixel * 2048 pixel FFI stacks, run
```
bash run_all_2048_2048.sh
```
Then concatenate the generated *.csv* files together for the post-processing steps.

- To de-duplicate the results with the **DBSCAN** algorithm, run
```
python 4_deduplication.py
```

- Show some shift-stacked images with asteroids, run
```
check_detections.py
```

## Citation
If you find this code useful, please cite our paper

```
@article{Zhang_2025,
doi = {10.3847/1538-3881/adea3c},
url = {https://dx.doi.org/10.3847/1538-3881/adea3c},
year = {2025},
month = {aug},
publisher = {The American Astronomical Society},
volume = {170},
number = {3},
pages = {187},
author = {Zhang, Peilin and Luo, Rui and Chen, Man},
title = {Detecting Asteroids in TESS Full-frame Images Using a Gradient-based Iterative Algorithm},
journal = {The Astronomical Journal},
abstract = {The shift-stacking method is commonly used to detect asteroids in astronomical multiframe image data. However, the computation required for the traditional shift-stacking method is substantial. In this study, we observed a relation between velocity hypotheses and signal intensities in the stacked images. Based on this observation, we propose a gradient-based iterative algorithm to detect asteroids. The time complexity for detecting faint moving objects can be drastically reduced by avoiding calculations throughout the entire velocity space. A series of preprocessing steps were implemented to increase the effectiveness of the algorithm. We used the Transiting Exoplanet Survey Satellite full-frame image data consisting of 128 frames and covering a 12° × 12° area to demonstrate the performance of our pipeline. Within the data, 3143 independent detections were returned, of which 2266 are correlated to known asteroids. The completeness for asteroids brighter than 20 and 21 mag is 82.7% and 71.6%, respectively, and the detectability limit using our pipeline is estimated to be a visual magnitude of 21.8.}
}
```


