import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import imageio
import matplotlib.cm as cm
    
def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    
    return im2

def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    
    return im2

def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
        
    if size==8:
        print('Not yet implemented\n')
        return -1

def local_energy(visible_arr,hidden_arr, x_val,y_val, const_list):
    h_val = const_list[0]
    total_pixels = hidden_arr.shape[0]*hidden_arr.shape[1]
    energy = h_val*hidden_arr[x_val,y_val]
    for x_n,y_n in neighbours(x_val,y_val,hidden_arr.shape[0],hidden_arr.shape[1]):
        energy += hidden_arr[x_n,y_n]
    energy = energy/total_pixels
    return energy

def total_energy(visible_arr,hidden_arr,const_list):
    energy = 0.
    for i in range(visible_arr.shape[0]):
        for j in range(visible_arr.shape[1]):
            energy += local_energy(visible_arr,hidden_arr,i,j,const_list)
    return energy

def gibbs_sampling(visible_arr, hidden_arr, px_x, px_y, total_energy, const_list):
    current_energy = local_energy(visible_arr, hidden_arr,px_x,px_y, const_list)
    other_energy = total_energy - current_energy
    new_hidden_arr = np.copy(hidden_arr)
    np.random.seed(42)
    x = np.random.randint(0, hidden_arr.shape[0])
    y = np.random.randint(0, hidden_arr.shape[1])
    new_energy = local_energy(visible_arr, hidden_arr,x,y, const_list)
    p = 1 / (1 + np.exp(-2 * const_list[1] * new_energy))
    if np.random.random() < p:
        new_hidden_arr[x,y] = 1
    else:
        new_hidden_arr[x,y] = -1

    flipped_energy = local_energy(visible_arr, new_hidden_arr,px_x,px_y, const_list)
    if flipped_energy < current_energy:
        total_energy = other_energy + flipped_energy
        hidden_arr = new_hidden_arr
    
    return (hidden_arr,total_energy)

def denoising(total_energy, energy_this_round, noisy_img_arr, hidden_image, const_list):
    for sim_round in range(20):
        for i in range(hidden_image.shape[0]):
            for j in range(hidden_image.shape[1]):
                hidden_image,total_energy = gibbs_sampling(noisy_img_arr,hidden_image,i,j, total_energy,const_list)

        # if (total_energy - energy_this_round) == 0:
        #     print(sim_round)
        #     break
        # energy_this_round = total_energy
    return hidden_image

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1

im = imageio.imread('/Users/linfeng/workspace/MachineLearning/src/cat.png')
im = im/255
fig = plt.figure()
# ax = fig.add_subplot(321)
# ax.imshow(im,cmap='gray')

im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(121)
ax2.imshow(im2,cmap='gray')

img_gray_arr_2 = np.asarray(im2,int)
img_mean_2 = np.mean(img_gray_arr_2)
img_arr_2 = np.copy(img_gray_arr_2)
img_arr_2[img_gray_arr_2<img_mean_2] = -1
img_arr_2[img_gray_arr_2>=img_mean_2] = 1
noisy_img_arr_2 = np.copy(img_arr_2)

const_list = [0,.3,.02]
hidden_image_2 = np.copy(noisy_img_arr_2)
total_energy_2= total_energy(noisy_img_arr_2, hidden_image_2, const_list)

hidden_image_2 = denoising(total_energy_2, total_energy_2, noisy_img_arr_2, hidden_image_2, const_list)

ax4 = fig.add_subplot(122)
ax4.imshow(hidden_image_2,cmap='gray')

plt.show()