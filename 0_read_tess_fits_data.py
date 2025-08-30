import os
import numpy as np
from astropy.io import fits

## get and sort all data filenames
folder = '../fits_data/'
files = np.sort(os.listdir(folder))

## select fits files only
ffi_fits_files = []
for f in files:
    if f.endswith('.fits'):
        ffi_fits_files.append(f)

## number of all fits files
n_frames = len(ffi_fits_files)
print('total number of frames:', n_frames)

## science pixels for TESS FFI
crop_x0 = 0
crop_x1 = 2048
crop_y0 = 44
crop_y1 = 44 + 2048

## read TESS FFI data from fits files
stack = []
for idx in range(n_frames):
    with fits.open(folder+ffi_fits_files[idx]) as hdul:
        img = hdul[1].data[crop_x0:crop_x1, crop_y0:crop_y1] ##save image stack without margin
    stack.append(img)

## save the data as .npy file for the preprocessing step
stack = np.array(stack)
print(stack.shape)
np.save('raw_data.npy', stack)
