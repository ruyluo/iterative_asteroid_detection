import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN


mp_info = pd.read_csv('detection_1024_1024_rg16.csv')
locx = mp_info['locx']
locy = mp_info['locy']
velox = mp_info['velox']
veloy = mp_info['veloy']
pmag = mp_info['brightest']
nframes = mp_info['n_frames']

locx = np.array(locx)
locy = np.array(locy)
velox = np.array(velox)
veloy = np.array(veloy)
pmag = np.array(pmag)
nframes = np.array(nframes)

## rescale the velocity to be comparable to the position (pixel)
velox = velox * 20.0
veloy = veloy * 20.0

## exclude signals with less than 64 stacked frames (those in the edge region, very likely to be false positive)
condition = nframes>64

locx = locx[condition]
locy = locy[condition]
velox = velox[condition]
veloy = veloy[condition]
pmag = pmag[condition]

## remove the faint signals (likely to be false positive),
## here, one can change the threhold of 0.19 to alter the number of detections

condition = pmag>0.19 

locx = locx[condition]
locy = locy[condition]
velox = velox[condition]
veloy = veloy[condition]
pmag = pmag[condition]

## DBSCAN algorithm
X = np.array([locx,locy, velox, veloy])
X = X.T

labels = DBSCAN(eps=3, min_samples=2).fit_predict(X)

print('number of detections before DBSCAN', len(X)),
print('number of clusters', len(set(labels)))
print('number of outliers', np.sum(labels==-1))

## reset the velocities as the initial values
velox = velox / 20.0
veloy = veloy / 20.0

## output the result of deduplication (each cluster and outlier)
out_file = open('deduplicated_detection.csv', 'w')
out_file.write('locx,locy,velox,veloy,brightest\n')

## the brightest signal within each cluster
cnt = 0
for l in range(np.max(labels)+1):
    ind = np.argmax(pmag[labels==l])
    x = locx[labels==l][ind]
    y = locy[labels==l][ind]
    vx = velox[labels==l][ind]
    vy = veloy[labels==l][ind]
    m = pmag[labels==l][ind]
    cnt += 1
    out_file.write(str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(m)+'\n')

## outliers
cnt = 0
for x, y, vx, vy, m in zip(locx[labels==-1], locy[labels==-1], velox[labels==-1], veloy[labels==-1], pmag[labels==-1]):
    out_file.write(str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(m)+'\n')
    cnt += 1

out_file.close()

