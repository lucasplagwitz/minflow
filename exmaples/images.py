import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from minflow.helper import create_map
from minflow.interfaces import MinCostFlow

digits = load_digits(return_X_y=False)['data']
nu = np.reshape(digits[0,:], (8, 8))
mu = np.reshape(digits[1,:], (8, 8))
mu2 = np.reshape(digits[10,:], (8, 8))

mu = mu / np.sum(mu)
nu = nu / np.sum(nu)



cf = MinCostFlow(mu, nu, 0.1)
a0 = cf.solve()

cf = MinCostFlow(mu2, nu, 0.1)
a1 = cf.solve()

graph0 = create_map(a0, m=mu.shape[0], n=mu.shape[1])
graph1 = create_map(a1, m=mu.shape[0], n=mu.shape[1])


f, axarr = plt.subplots(2,3)
axarr[0, 0].imshow(nu)
axarr[0, 1].imshow(mu)
axarr[0, 2].imshow(np.abs(graph0))
axarr[1, 0].imshow(nu)
axarr[1, 1].imshow(mu2)
axarr[1, 2].imshow(np.abs(graph1))
#plt.colorbar()
plt.savefig("mnist_map.png")
