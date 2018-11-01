# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np

# # x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
# # plt.plot(x, np.sin(x))       # Plot the sine of each x point
#                  # Display the plot

# def gkern(l=10, sig=1):
#     """
#     creates gaussian kernel with side length l and a sigma of sig
#     """

#     ax = np.random.random_integers(10, size=(3,2))
#     xx, yy = np.meshgrid(ax, ax)

#     kernel = sig * np.exp(-(xx - yy)**2 / (2. * l**2))

#     return kernel / np.sum(kernel)

# plt.plot(gkern(5,1))
# plt.show()  

print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np
import math

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))]

for fig_index, kernel in enumerate(kernels):
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)

    # Plot prior
    plt.figure(fig_index, figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

plt.show()