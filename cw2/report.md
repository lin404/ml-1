# 1. Implement ICM for a binary image that you have corrupted with noise, show the result for a set of images with different noise levels. How many passes through the nodes do you need until you are starting to get descent results?

The top row shows image with Gaussian noise on the left and the image with ’salt-and-pepper’ noise on the right. The bottom row shows the restored images obtained using iterated conditional models (ICM) with 2 different noise levels respectively.(Figure_1)
Around 10 passes through the all nodes until we get descent results.

# 2. Implement the Gibbs sampler for the image denoising task. Generate images with different amount of noise and see where it fails and where it works. My result is shown in Figure 2.

The top row shows image with Gaussian noise on the left and the image with ’salt-and-pepper’ noise on the right. The bottom row shows the restored images obtained using Gibbs Sampling with 2 different noise levels respectively.(Figure_2)
It works for the image with Gaussian nois, but fails to restore the image with ’salt-and-pepper’ noise.

# 3. There is nothing saying that you should cycle through the nodes in the graph index by index, you can pick any different order and you do not have to visit each node equally many times either. Alter your sampler so that it picks and updates a random node each iteration. Are the results different? Do you need more or less iterations? To get reproduceable results fix the random seed in your code with np.random.seed(42).

The Gibbs model is to loop over each n, and with this probability set xn=1, otherwise setting it to −1. As there will be spacial correlations, it makes more sense to randomly select the n each time. The results are the same, and we would need less iterations in general.

# 4. What effect does the number of iterations we run the sampler have on the results? Try to run it for different times, does the result always get better?

It does not always get better. If changing xn from it's current state decreases the energy, then we accept the flip, do more iterations; otherwise doing more iterations may get worse results. So we should stop iteration when the current energy is smaller than the energy after flipping.
