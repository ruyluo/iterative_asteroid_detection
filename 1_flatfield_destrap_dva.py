from utils import *

n_frames = 256

stack = np.load('raw_data.npy')

## replace the problematic frames by the average of the previous and next frames
if stack.shape[0]>146:
    stack[146] = (stack[145] + stack[147]) / 2.0
if stack.shape[0]>290:
    stack[290] = (stack[289] + stack[291]) / 2.0
if stack.shape[0]>434:
    stack[434] = (stack[433] + stack[435]) / 2.0

## using 256 frames for the preprocessing step
stack = stack[24:24+n_frames]

## flat field correction and de-strap
for i in range(n_frames):
    stack[i] = flatfield(stack[i], scale=64)
    stack[i] = destrap(stack[i])

## tag the bright stars (static sources) in the frames
stack_raw_mean = np.mean(stack, axis=0)
static_source = stack_raw_mean > 10

## clip the extra bright pixels in the images
stack = np.clip(stack, a_min=-10, a_max=15)

## remove the signals of static sources
for i in range(stack.shape[0]):
    stack[i][static_source] = 0

## remove DVA effect
stack_raw_sum = np.sum(stack, axis=0)
pixel_detrend_linear(stack)
pixel_detrend_linear_with_mask(stack)

## replace variable stars and problematic pixels with median pixel value of the frame
stack_max = np.max(stack, axis=0)
remove = np.logical_and(stack_raw_sum > 1e3, stack_max > 1.0)
med = np.median(stack)
for i in range(n_frames):
    stack[i][remove] = med

## save the data for the next step
np.save('data_dva_removed.npy', stack)
