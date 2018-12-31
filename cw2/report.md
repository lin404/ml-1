# 1. Implement ICM for a binary image that you have corrupted with noise, show the result for a set of images with different noise levels. How many passes through the nodes do you need until you are starting to get descent results?

The top row shows image with Gaussian noise on the left and the image with ’salt-and-pepper’ noise on the right. The bottom row shows the restored images obtained using iterated conditional models (ICM) with 2 different noise levels respectively.(Figure_1)
Around 10 passes through the all nodes until we get descent results.

(Figure_1.png)

# 2. Implement the Gibbs sampler for the image denoising task. Generate images with different amount of noise and see where it fails and where it works. My result is shown in Figure 2.

The top row shows image with Gaussian noise on the left and the image with ’salt-and-pepper’ noise on the right. The bottom row shows the restored images obtained using Gibbs Sampling with 2 different noise levels respectively.(Figure_2)
Gibbs sampler works for both of these noises. It will fail if the noise is big enough that the probability is too small.

(Figure_2.png)

# 3. There is nothing saying that you should cycle through the nodes in the graph index by index, you can pick any different order and you do not have to visit each node equally many times either. Alter your sampler so that it picks and updates a random node each iteration. Are the results different? Do you need more or less iterations? To get reproduceable results fix the random seed in your code with np.random.seed(42).

In the second row, the left image is the denoise image by visiting each node equally 10 times, the right image is the denoise image by picking and updating a random node 500000 times.(Figure_3_1)

In the second row, the left image is the denoise image by visiting each node equally 10 times, the right image is the denoise image by picking and updating the same random node 500000 times.(Figure_3_2)

As in general there will be spacial correlations, it makes more sense to randomly select the node each time. Therefore, the result of randomly selecting the node is the same as the results of visiting each node equally many times (Figure_3_1.png). However, by using np.random.seed(42), pick and update the same random node, the result is not the same, it will be worse (Figure_3_2.png).
The size of the image is 225*225, if it visits each node equally time (eg: 10 times in total), it needs 10*225*225 times of iterations, while picking and updating a random node each time, to get the same result, it needs less than 10*225*225 times of iterations.

(Figure_3_1.png)(Figure_3_2.png)

# 4. What effect does the number of iterations we run the sampler have on the results? Try to run it for different times, does the result always get better?

In the second row, the left image is the denoise image after 5 times of iterations, the right image is the denoise image after 100 times of iterations.(Figure_4)

It is getting better after flipping more times. However, it does not always get better, when the mean of the energy does not change, the result will not change with more iterations.

(Figure_4.png)