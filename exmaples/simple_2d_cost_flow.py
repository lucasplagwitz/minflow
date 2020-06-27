import numpy as np
import matplotlib.pyplot as plt

from clusterot.helper.connection_matrix import create_map
from clusterot.interfaces.min_cost_flow import MinCostFlow

nu = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mu = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])

cf = MinCostFlow(mu, nu, 0.1)

a = cf.solve()

graph = create_map(a,m=mu.shape[0], n=mu.shape[1])

plt.imshow(graph)
plt.colorbar()
plt.savefig("map.png")
