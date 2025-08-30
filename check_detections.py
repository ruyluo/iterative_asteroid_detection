#import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from scipy import ndimage
#from scipy import signal
import time

from utils import *

stack = np.load('data_with_skeletonization.npy')[:128]#[:128,1024:,1024:]
print('using skeletonized data')
print(stack.shape)

torch.cuda.empty_cache()
stack = np.array(stack[:,np.newaxis,:,:])
stack = torch.tensor(stack, dtype=torch.float).cuda()


df = pd.read_csv('deduplicated_detection.csv')
locx = df['locx']
locy = df['locy']
velox = df['velox']
veloy = df['veloy']
mag = df['mag']

rg = 5

for y, x, vx, vy, m in zip(locx, locy, velox, veloy, mag):
    region_x_min, region_x_max, region_y_min, region_y_max = x-rg, x+rg+1, y-rg, y+rg+1
    patch_ssum = shift_and_sum(stack, region_x_min, region_x_max, region_y_min, region_y_max, vx, vy)
    patch_ssum_numpy = patch_ssum.data.cpu().numpy()
    print(np.max(patch_ssum_numpy), m)
    plt.imshow(patch_ssum_numpy, cmap='gray_r')
    plt.show()
    plt.close()
