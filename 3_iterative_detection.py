import argparse
from utils import *

parser = argparse.ArgumentParser()
## x_axis: the normal of ecliptic plane
## y_axis: along the ecliptic plane
parser.add_argument('region_x_min', help='region_x_min', type=int)
parser.add_argument('region_y_min', help='region_y_min', type=int)
args = parser.parse_args()

## load the preprocess data
stack = np.load('data_with_skeletonization.npy')[:128]

region_x_min = args.region_x_min
region_y_min = args.region_y_min

torch.cuda.empty_cache()

stack = np.array(stack[:,np.newaxis,:,:])
stack = torch.tensor(stack, dtype=torch.float).cuda()

stride = 8
rg = 16 ## patch size
thres1 = 0.1 ##search if the patch with higher possibility of detecting an asteroid
step = 0.01 ## step size

## output csv file
file_mp_info = open('detection_'+str(region_x_min)+'_'+str(region_y_min)+'_rg16.csv', 'w')
file_mp_info.write('locx,locy,velox,veloy,brightest,n_frames\n')

## the index for the patches
if region_x_min + 32*stride + rg > 2048:
    NX = 31
else:
    NX = 32
    
if region_y_min + 32*stride + rg > 2048:
    NY = 31
else:
    NY = 32 

## apply the gradient-based algorithm for each patch
for i in range(NX):
    print(i)
    for k in range(NY):
        ## patch positions
        x_min = region_x_min + i*stride
        x_max = region_x_min + i*stride + rg
        y_min = region_y_min + k*stride
        y_max = region_y_min + k*stride + rg

        ## trial initial velocities
        best_init_vx, best_init_vy, best_init_pixel = [], [], []
        for init_vx in np.arange(-0.36, 0.361, 0.072):
            for init_vy in np.arange(0.6, 1.001, 0.08):
                patch_ssum = shift_and_sum(stack, x_min, x_max, y_min, y_max, init_vx, init_vy)
                best_init_vx.append(init_vx)
                best_init_vy.append(init_vy)
                best_init_pixel.append(torch.max(patch_ssum))

        ## the gradient-based algorithm       
        if np.max(best_init_pixel) > thres1:
            arg_best = np.argmax(best_init_pixel)
            vx = best_init_vx[arg_best]
            vy = best_init_vy[arg_best]
            best_pixel_value = best_init_pixel[arg_best]

            best_vx = vx
            best_vy = vy

            patch_ssum, p_n = shift_and_sum_n(stack, x_min, x_max, y_min, y_max, vx, vy)

            argmax = torch.argmax(patch_ssum)
            p_max = torch.max(patch_ssum)

            ## the potential asteroid position in the patch
            brightest_x = x_min + argmax.item() // rg
            brightest_y = y_min + argmax.item() % rg
            best_n = p_n

            step = 0.01 ## step size
            tolerence = 0
            vx_history, vy_history = [], []
            for it in range(200):
                vx_history.append(vx)
                vy_history.append(vy)
                tolerence += 1

                patch_ssum, p_n = shift_and_sum_n(stack, x_min, x_max, y_min, y_max, vx, vy)
                p_max = torch.max(patch_ssum).item()

                ## update the record if a better velocity hypothesis is found
                if best_pixel_value < p_max:
                    best_pixel_value = p_max
                    best_vx = vx
                    best_vy = vy
                    best_n = p_n
                    tolerence =0 

                    argmax = torch.argmax(patch_ssum)
                    brightest_x = x_min + argmax.item() // rg
                    brightest_y = y_min + argmax.item() % rg

                g_x, g_y, objective = gradient(stack, x_min, x_max, y_min, y_max, vx, vy, obj='max', eps=0.01)
                vy += g_y * step
                vx += g_x * step

                ## stop the searching when no better result yield in 50 consecutive steps
                if tolerence > 50:
                    break

            ## output the asteroid information to the .csv file
            mp_info = '%d'%brightest_x+',%d'%brightest_y+',%.9f'%best_vx+',%.9f'%best_vy+',%.9f'%best_pixel_value+',%d'%best_n+'\n'
            print(mp_info [:-1])
            file_mp_info.write(mp_info)

file_mp_info.close()
