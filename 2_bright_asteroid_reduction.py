from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
from utils import *

stack = np.load('data_dva_removed.npy')
n_frames = stack.shape[0]


## bright asteroid signal reduction with the skeletonization algorithm
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

## save the .npy file for the iterative asteroid detection algorithm
np.save('data_with_skeletonization.npy', stack)
