from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

zero_ind = np.where(y == '0')
five_ind =np.where(y == '5')

from minflow.helper import create_map
from minflow.interfaces import MinCostFlow

shape = (28, 28)

nu = np.reshape(X[five_ind[0][0], :], shape)
mu = np.reshape(X[zero_ind[0][0], :], shape)
mu2 = np.reshape(X[five_ind[0][1], :], shape)

mu = mu / np.sum(mu) * 200
nu = nu / np.sum(nu) * 200
mu2 = mu2 / np.sum(mu2) * 200



cf = MinCostFlow(mu, nu, 0.001)
a0 = cf.solve()

cf = MinCostFlow(mu2, nu, 0.001)
a1 = cf.solve()

graph0 = create_map(a0, m=mu.shape[0], n=mu.shape[1])
graph1 = create_map(a1, m=mu.shape[0], n=mu.shape[1])

graph0[np.isnan(graph0)] = 0
graph1[np.isnan(graph1)] = 0

f, axarr = plt.subplots(2,3)
axarr[0, 0].imshow(nu)
axarr[0, 1].imshow(mu)
axarr[0, 2].imshow(np.abs(graph0))
axarr[1, 0].imshow(nu)
axarr[1, 1].imshow(mu2)
axarr[1, 2].imshow(np.abs(graph1))
plt.savefig("big_mnist_map.png")