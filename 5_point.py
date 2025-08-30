#import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
#from scipy import signal
import time

from utils import *
stack = np.load('temps.npy')[:128]#[:128,1024:,1024:]
stack = np.array(stack[:,np.newaxis,:,:])

############ SKELETONIZATION ###############

#figs, (ax1, ax2) = plt.subplots(1,2,sharex=True, sharey=True)

#ax1.imshow(stack[0])
'''

n_frames = stack.shape[0]
for i in range(n_frames):
    img = stack[i].copy()

    img = gaussian_filter(img, sigma=1)

    bright = img > 0.5

    a = np.zeros_like(img)
    a[bright] = 1

    sk = skeletonize(a).astype(np.float)

    sk = gaussian_filter(sk, sigma=1) 

    stack[i][bright] = 0
    stack[i] = stack[i] + sk

#ax2.imshow(stack[0])
#plt.show()
#plt.close()
print('skeletonization process finished')'''
    
###############################################


#stack[stack>3] = 0
#stack = np.clip(stack, a_min=-3, a_max=3)
#stack = np.clip(stack, a_min=0, a_max=1)
print(stack.shape)


torch.cuda.empty_cache()

stack = torch.tensor(stack, dtype=torch.float).cuda()

#blurred_stack = blur(stack, 5)

#####################################################3
folder = './csv_files/'
#csv_file = 'detection2_16_all.csv'
csv_file = 'mp_deduplicated_detection2.csv'
#mp_info = pd.read_csv('mp_results_2.csv')
#mp_info = pd.read_csv('mp_cluster.csv')
mp_info = pd.read_csv(folder+csv_file)
#mp_info = pd.read_csv('mp_point_test.csv')
locx = mp_info['locy']
locy = mp_info['locx']# - 44
velox = mp_info['velox']
veloy = mp_info['veloy']
brightest = mp_info['mag']

locx = np.array(locx)
locy = np.array(locy)
velox = np.array(velox)
veloy = np.array(veloy)
brightest = np.array(brightest)

###############################################################
'''mp_order = np.argsort(-brightest) ## largest pixel value first

locx = locx[mp_order]
locy = locy[mp_order]
velox = velox[mp_order]
67veloy = veloy[mp_order]
brightest = brightest[mp_order]'''
###############################################################

rg = 5
radius = 6
disk = np.zeros((2*rg+1,2*rg+1))
for i in range(2*rg+1):
    for k in range(2*rg+1):
        if (i-rg)**2 + (k-rg)**2 < radius:
            disk[i,k] = 1

#plt.imshow(disk)
#plt.show()
#plt.close()
condition1 = np.logical_and(locx> rg , locx<2048-rg)
condition2 = np.logical_and(locy> rg , locy<2048-rg)
condition = np.logical_and(condition1, condition2)

locx = locx[condition]
locy = locy[condition]
velox = velox[condition]
veloy = veloy[condition]
brightest = brightest[condition]


## point_like data
p_x, p_y, p_vx, p_vy, p_brightness = [], [], [], [], []
idx = 0
for x, y, vx, vy, m in zip(locx, locy, velox, veloy, brightest):
    idx += 1
    if idx % 100 == 0:
        print(idx)        
    region_x_min, region_x_max, region_y_min, region_y_max = x-rg, x+rg+1, y-rg, y+rg+1
    patch_ssum2 = shift_and_sum(stack, region_x_min, region_x_max, region_y_min, region_y_max, vx, vy)
    patch_ssum_numpy2 = patch_ssum2.data.cpu().numpy()
    
    patch_ssum_numpy2 = patch_ssum_numpy2 - np.median(patch_ssum_numpy2)

    #if  700<y<720 and 474<x<484:

    ############ NOTICE: summation might be a negative number ########################
    peak = np.max(patch_ssum_numpy2)
    point_pixels = np.sum(patch_ssum_numpy2>0.5*peak)

    d1 = np.sum(patch_ssum_numpy2)
    d2 = np.sum(disk * patch_ssum_numpy2)

    #if peak > 0.5:
    '''print(peak, point_pixels, d2, d1)
    plt.imshow(patch_ssum_numpy2, cmap='gray_r')
    plt.show()
    plt.close()'''
    

    ############### POINT CONDITIONS ###################
    if peak>0.7 or (point_pixels <=4 and (d2>d1 or d2>0.85*d1)):
        p_x.append(x)
        p_y.append(y)
        p_vx.append(vx)
        p_vy.append(vy)
        p_brightness.append(m)

p_x = np.array(p_x)
p_y = np.array(p_y)
p_vx = np.array(p_vx)
p_vy = np.array(p_vy)
p_brightness = np.array(p_brightness)

print(len(p_vy), 'point sources selected.')


d = {'locx':p_y, 'locy':p_x, 'velox':p_vx, 'veloy':p_vy, 'mag':p_brightness}

df = pd.DataFrame(d)

df.to_csv(folder+'temps_point_'+csv_file, index=False)

            
'''
for x, y, vx, vy, m in zip(locx, locy, velox, veloy, brightest):
    if m > 0.:
        region_x_min, region_x_max, region_y_min, region_y_max = x-rg, x+rg+1, y-rg, y+rg+1
        print(x, y, vx, vy, m)
        #patch_ssum1 = shift_and_sum(stack[:64],region_x_min, region_x_max, region_y_min, region_y_max, vx, vy)
        #patch_ssum_numpy1 = patch_ssum1.data.cpu().numpy()
        patch_ssum2 = shift_and_sum(stack, region_x_min, region_x_max, region_y_min, region_y_max, vx, vy)
        patch_ssum_numpy2 = patch_ssum2.data.cpu().numpy()

        patch_ssum_numpy2 = patch_ssum_numpy2 - np.median(patch_ssum_numpy2)

c9
        ############ summation might be a negative number ########################
        peak = np.max(patch_ssum_numpy2)
        point_pixels = np.sum(patch_ssum_numpy2>0.5*peak)
        print('-----',np.sum(patch_ssum_numpy2), np.sum(disk*patch_ssum_numpy2),'--', point_pixels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(patch_ssum_numpy2, cmap='gray_r')
        ax1.plot(rg, rg, 'c+')
        im2 = ax2.imshow(disk * patch_ssum_numpy2, cmap='gray_r')
        ax2.plot(rg, rg, 'c+')
        #im3 = ax3.imshow(patch_ssum_numpy2, cmap='gray_r')
        #ax3.plot(8, 8, 'c+')
        #divider = make_axes_locatable(ax3)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        #cbar = plt.colorbar(im2, cax=cax)
        #cbar.set_label('Shared Colorbar')

        plt.tight_layout()
        plt.show()
        # ssum = shift_and_sum(stack, region_x_min, region_x_max, region_y_min, region_y_max, vx, vy)
        # img = ssum.data.cpu().numpy() ## use pytorch maybe
        # plt.imshow(img, cmap='gray_r')
        # plt.title(str(y+44)+' '+str(x)+' '+str(vx)+' '+str(vy)+' '+str(m))
        # plt.show()
        # plt.close()'''
