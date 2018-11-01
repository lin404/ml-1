from pylab import *
from numpy import *
from numpy.random import normal
from scipy.optimize import fmin

e = lambda p, x, y: (abs((fp(p,x)-y))).sum()

fp = lambda c, x: c[0]+c[1]*x
real_p = rand(2)

# generating data with noise
n = 3
x = linspace(0,1,n)
y = fp(real_p,x) + normal(0,0.05,n)

# fitting the data with fmin
p0 = rand(2)
p = fmin(e, p0, args=(x,y))

xx = linspace(0,1,n*3)
plot(x,y,'bo', xx,fp(real_p,xx),'g', xx, fp(p,xx),'r')

show()