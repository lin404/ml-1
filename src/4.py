import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import imageio
import matplotlib.cm as cm
import numpy as np
    
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

def local_energy(visible_arr, hidden_arr,x, y):
        return visible_arr[x,y] + sum(hidden_arr[xx,yy] for (xx, yy) in neighbours(x,y,visible_arr.shape[0],visible_arr.shape[1]))

def total_energy(visible_arr,hidden_arr,const_list):
    energy = 0.
    for i in range(visible_arr.shape[0]):
        for j in range(visible_arr.shape[1]):
            energy += local_energy(visible_arr,hidden_arr,i,j)
    return energy

def gibbs_move(visible_arr, hidden_arr, x, y, beta):
    # n = np.random.randint(0, hidden_arr.shape[0] * hidden_arr.shape[1])
    # y = n // hidden_arr.shape[0]
    # x = n % hidden_arr.shape[0]
    current_energy = local_energy(visible_arr, hidden_arr,x,y)
    p = 1 / (1 + np.exp(-2 * beta * current_energy))
    if np.random.uniform(0,1) <= p:
        hidden_arr[x,y] = 1
    else:
        hidden_arr[x,y] = -1

    return hidden_arr

def denoising_1(noisy_img_arr, hidden_image, const_list):
    loop = 5
    avg = np.zeros_like(hidden_image).astype(np.float64)
    for sim_round in range(loop):
        for x in range(hidden_image.shape[0]):
            for y in range(hidden_image.shape[1]):
                hidden_image = gibbs_move(noisy_img_arr, hidden_image, x, y, const_list)
                avg += hidden_image
    
    avg = avg / loop
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int)
    
    return avg

def denoising_2(noisy_img_arr, hidden_image, const_list):
    loop = 100
    avg = np.zeros_like(hidden_image).astype(np.float64)
    for sim_round in range(loop):
        for x in range(hidden_image.shape[0]):
            for y in range(hidden_image.shape[1]):
                hidden_image = gibbs_move(noisy_img_arr, hidden_image, x, y, const_list)
                avg += hidden_image
    
    avg = avg / loop
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int)
    
    return avg

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1

im = imageio.imread('/Users/linfeng/workspace/MachineLearning/cat.png')
im = im/255
fig = plt.figure()
# ax = fig.add_subplot(321)
# ax.imshow(im,cmap='gray')

im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(221)
ax2.imshow(im2,cmap='gray')
# im3 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(222)
ax3.imshow(im2,cmap='gray')

img_gray_arr_2 = np.asarray(im2,int)
img_mean_2 = np.mean(img_gray_arr_2)
img_arr_2 = np.copy(img_gray_arr_2)
img_arr_2[img_gray_arr_2<img_mean_2] = -1
img_arr_2[img_gray_arr_2>=img_mean_2] = 1
noisy_img_arr_2 = np.copy(img_arr_2)

img_gray_arr_3 = np.asarray(im2,int)
img_mean_3 = np.mean(img_gray_arr_3)
img_arr_3 = np.copy(img_gray_arr_3)
img_arr_3[img_gray_arr_3<img_mean_3] = -1
img_arr_3[img_gray_arr_3>=img_mean_3] = 1
noisy_img_arr_3 = np.copy(img_arr_3)

beta = 2
hidden_image_2 = np.copy(noisy_img_arr_2)
hidden_image_3 = np.copy(noisy_img_arr_3)

avg_2 = denoising_1(noisy_img_arr_2, hidden_image_2, beta)
avg_3 = denoising_2(noisy_img_arr_3, hidden_image_3, beta)

ax4 = fig.add_subplot(223)
ax4.imshow(avg_2,cmap='gray')
ax4 = fig.add_subplot(224)
ax4.imshow(avg_3,cmap='gray')

plt.show()