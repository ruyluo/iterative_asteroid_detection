import numpy as np    
from skimage.transform import rescale
import torch
import torch.nn as nn

def flatfield(img, scale=64):
    assert(img.shape[0]%scale==0)

    size0 = img.shape[0]//scale
    size1 = img.shape[1]//scale

    downscale = np.zeros((size0, size1))
    for i in range(size0):
        for k in range(size1):
            downscale[i,k] = np.median(img[i*scale:i*scale+scale, k*scale:k*scale+scale])

    downscale = rescale(downscale, scale)

    return img - downscale

# -----------------------------------------------------------------------------------------

def destrap(img):
    line_median = np.median(img, axis=0, keepdims=True)
    return img - line_median

# -----------------------------------------------------------------------------------------
def pixel_detrend_linear(X):
    n = X.shape[0]
    sum_n = n * (n-1) / 2
    sum_nn = (n-1)*n*(2*n-1)/6
    t = np.arange(0,n,1)
    rX = np.rollaxis(X, 0, 3)
    sum_tx = rX@t
    sum_x = np.sum(X, axis=0)
    a = (n * sum_tx - sum_n * sum_x) / (n * sum_nn - sum_n**2)
    b = (sum_n * sum_tx - sum_nn * sum_x) / (sum_n**2 - n*sum_nn)
    for i in range(n):
        X[i] = X[i] - a*i - b


def pixel_detrend_linear_with_mask(X):
    n = X.shape[0] # number of frames
    pixel_wise_std = np.std(X, axis=0)
    pixel_wise_mean = np.mean(X, axis=0)

    sum_t = np.zeros_like(X[0]) ## summation of time
    sum_tt = np.zeros_like(X[0]) ## summation of time squared
    sum_tx = np.zeros_like(X[0]) ## summation of time * flux
    sum_x = np.zeros_like(X[0]) ## summation of flux

    for i in range(n):
        normal = (np.abs(X[i] - pixel_wise_mean) < 3.0 * pixel_wise_std).astype(float)
        sum_t = sum_t + (np.ones_like(X[0]) * i ) * normal
        sum_tt = sum_tt + (np.ones_like(X[0]) * i*i ) * normal
        sum_tx = sum_tx + (X[i] * i) * normal
        sum_x = sum_x + X[i] * normal

    a = (n * sum_tx - sum_t * sum_x) / (n * sum_tt - sum_t**2 + 1e-7)
    b = (sum_t * sum_tx - sum_tt * sum_x) / (sum_t**2 - n*sum_tt + 1e-7)

    for i in range(n):
        X[i] = X[i] - a*i - b

    return a, b


def draw_vh_map(stack, rgx0, rgx1, rgy0, rgy1, vx_min = -0.6, vx_max = 0.6, vy_min = 0.3, vy_max = 1.3, Nx = 50, Ny = 50, method='max'):
    n_frames = stack.size(0)
    size_x = stack.size(2)
    size_y = stack.size(3)

    vh_obj = []
    c = 0
    bounding_warning = True

    obj_max = -10000.0
    vx_best = 0
    vy_best = 0

    for dx in np.linspace(vx_min, vx_max, Nx):
        v = []
        for dy in np.linspace(vy_min, vy_max, Ny):
            stack_image = stack[0, 0, rgx0:rgx1, rgy0:rgy1]
            sum_cnt = 0
            for idx in range(1, n_frames, 1):
                osx = int(round(dx*idx))
                osy = int(round(dy*idx))
                if 0 <= rgx1+osx < size_x and 0 <= rgy1+osy < size_y and \
                   0 <= rgx0+osx < size_x and 0 <= rgy0+osy < size_y:
                    stack_image = stack_image + stack[idx, 0, rgx0+osx:rgx1+osx, rgy0+osy:rgy1+osy]
                    sum_cnt += 1
                elif bounding_warning:
                    print('shifting out of bound!')
                    bounding_warning = False
            stack_image = stack_image / sum_cnt

            if method == 'ms':
                objective = torch.mean(torch.pow(torch.abs(stack_image), 2)).item()
            elif method == 'max':
                objective = torch.max(stack_image).item()

            if objective>obj_max:
                obj_max = objective
                vx_best = dx
                vy_best = dy
            v.append(objective)
        vh_obj.append(v)
    vh_obj = np.array(vh_obj)

    return vh_obj, vx_best, vy_best


def shift_and_sum(stack, rgx0, rgx1, rgy0, rgy1, vx, vy):
    n_frames = stack.size(0)
    size_x = stack.size(2)
    size_y = stack.size(3)
    stack_image = stack[0, 0, rgx0:rgx1, rgy0:rgy1]
    sum_cnt = 1.0

    for idx in range(1, n_frames, 1):
        osx = int(round(vx*idx))
        osy = int(round(vy*idx))
        #print(osx, osy)
        if 0 <= rgx1+osx < size_x and 0 <= rgy1+osy < size_y and \
           0 <= rgx0+osx < size_x and 0 <= rgy0+osy < size_y:
            stack_image = stack_image + stack[idx, 0, rgx0+osx:rgx1+osx, rgy0+osy:rgy1+osy]
            sum_cnt += 1.0
    stack_image = stack_image / sum_cnt

    return stack_image

def shift_and_sum_n(stack, rgx0, rgx1, rgy0, rgy1, vx, vy):
    n_frames = stack.size(0)
    size_x = stack.size(2)
    size_y = stack.size(3)
    stack_image = stack[0, 0, rgx0:rgx1, rgy0:rgy1]
    sum_cnt = 1.0

    for idx in range(1, n_frames, 1):
        osx = int(round(vx*idx))
        osy = int(round(vy*idx))
        if 0 <= rgx1+osx < size_x and 0 <= rgy1+osy < size_y and \
           0 <= rgx0+osx < size_x and 0 <= rgy0+osy < size_y:
            stack_image = stack_image + stack[idx, 0, rgx0+osx:rgx1+osx, rgy0+osy:rgy1+osy]
            sum_cnt += 1.0
    stack_image = stack_image / sum_cnt

    return stack_image, sum_cnt


def gradient(blurred_stack, x_min, x_max, y_min, y_max, vx, vy, obj='ms', eps=0.01):
    ssum = shift_and_sum(blurred_stack, x_min, x_max, y_min, y_max, vx, vy)
    if obj=='ms':
        objective = torch.mean(torch.pow(torch.abs(ssum), 2)).item()
    elif obj=='max':
        objective = torch.max(ssum).item()
    elif obj=='ratio':
        objective = torch.max(ssum).item() / torch.mean(torch.pow(torch.abs(ssum), 2)).item()

    ssum_x = shift_and_sum(blurred_stack, x_min, x_max, y_min, y_max, vx+eps, vy)
    if obj=='ms':
        objective_x = torch.mean(torch.pow(torch.abs(ssum_x), 2)).item()
    elif obj=='max':
        objective_x = torch.max(ssum_x).item()
    elif obj=='ratio':
        objective_x = torch.max(ssum_x).item() / torch.mean(torch.pow(torch.abs(ssum_x), 2)).item()

    ssum_y = shift_and_sum(blurred_stack, x_min, x_max, y_min, y_max, vx, vy+eps)
    if obj=='ms':
        objective_y = torch.mean(torch.pow(torch.abs(ssum_y), 2)).item()
    elif obj=='max':
        objective_y = torch.max(ssum_y).item()
    elif obj=='ratio':
        objective_y = torch.max(ssum_y).item() / torch.mean(torch.pow(torch.abs(ssum_y), 2)).item()
    step_x = objective_x - objective
    step_y = objective_y - objective

    length = np.sqrt(step_x**2 + step_y**2)
    step_x = step_x / (length + 1e-6)
    step_y = step_y / (length + 1e-6)

    return step_x, step_y, objective

