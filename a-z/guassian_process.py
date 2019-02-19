import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# # create data
# x = np.linspace(-5,5,200)
# x = x.reshape(-1,1)
# mu = np.zeros(x.shape)
# # compute covariance matrix
# K = np.exp(-cdist(x,x)**2/5)
# # now we have the mean function and the covariance function 
# # the GP is fully described
# f = np.random.multivariate_normal(mu.flatten(),K,10)
# # plot the data
# plt.plot(x,f.T)
# plt.show()

rate = 1
apot_1 = 9325
apot_2 = 37950
apot_3 = 91900
apot_4 = 191650
apot_5 = 416700
apot_6 = 418400
apot_max = apot_1+apot_6

x = np.linspace(0, apot_1*rate, 1000)
plt.plot(x, x*0.1, color='red')
x = np.linspace(apot_1*rate, apot_2*rate, 1000)
plt.plot(x, 932.50*rate + (x-apot_1*rate)*0.15, color='red')
x = np.linspace(apot_2*rate, apot_3*rate, 100)
plt.plot(x, 5226.25*rate + (x-apot_2*rate)*0.25, color='red')
x = np.linspace(apot_3*rate, apot_4*rate, 100)
plt.plot(x, 18713.75*rate + (x-apot_3*rate)*0.28, color='red')
# x = np.linspace(apot_4*rate, apot_5*rate, 100)
# plt.plot(x, 46643.75*rate + (x-apot_4*rate)*0.33, color='red')
# x = np.linspace(apot_5*rate, apot_6*rate, 100)
# plt.plot(x, 120910.25*rate + (x-apot_5*rate)*0.35, color='red')
# x = np.linspace(apot_6*rate, apot_max*rate, 100)
# plt.plot(x, 121505.25*rate + (x-apot_6*rate)*0.396, color='red')

rate = 1/110
jpot_1 = 1950000
jpot_2 = 3300000
jpot_3 = 6950000
jpot_4 = 9000000
jpot_5 = 18000000
jpot_6 = 40000000
jpot_max = jpot_1+jpot_6

x = np.linspace(0, jpot_1*rate, 100)
plt.plot(x, x*0.05, color='green')
x = np.linspace(jpot_1*rate, jpot_2*rate, 100)
plt.plot(x, x*0.1, color='green')
x = np.linspace(jpot_2*rate, jpot_3*rate, 100)
plt.plot(x, x*0.2, color='green')
x = np.linspace(jpot_3*rate, jpot_4*rate, 100)
plt.plot(x, x*0.23, color='green')
x = np.linspace(jpot_4*rate, jpot_5*rate, 100)
plt.plot(x, x*0.33, color='green')
# x = np.linspace(jpot_5*rate, jpot_6*rate, 100)
# plt.plot(x, x*0.4, color='green')
# x = np.linspace(jpot_6*rate, jpot_max*rate, 100)
# plt.plot(x, x*0.45, color='green')

rate = 1.3
upot_1 = 11850
upot_2 = 46350
upot_3_range = 100000
upot_3 = 150000
upot_max = 50000000

x = np.linspace(0, upot_1*rate, 100)
plt.plot(x, x*0, color='blue')
x = np.linspace(upot_1*rate, upot_2*rate, 100)
plt.plot(x, x*0.2, color='blue')
x = np.linspace(upot_2*rate, upot_3_range*rate, 100)
plt.plot(x, x*0.4, color='blue')
# x = np.linspace(upot_3*rate, upot_max*rate, 100)
# plt.plot(x, x*0.45, color='blue')

plt.show()