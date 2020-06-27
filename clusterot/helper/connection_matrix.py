import pylops
from scipy import sparse
import numpy as np

def direct_connection(m, n):
    #a = np.ones(m*n-1) - np.array((([0]*(m-1)+[1])*n)[:-1])
    #b = np.ones(m*(n-1)) #- np.array([0]*((n-1)*m) + [1]*m)
    #D = sparse.diags(a, offsets=1) + sparse.diags(b, offsets=m)
    #return D - D.T
    a0 = np.array([1]*(n-1))
    b0 = np.array([1] * (n))
    a = sparse.diags([-a0, a0], [0,1], shape=(n-1, n))
    a1 = sparse.hstack((a, sparse.csr_matrix((n-1, n), dtype=int)))
    b = sparse.diags([-b0, b0], [0,n], shape=(n, 2*n))

    c = sparse.vstack((a1,b))

    result = sparse.vstack([sparse.hstack((sparse.csr_matrix((2*n-1, n*(i)), dtype=int), c, sparse.csr_matrix((2*n-1, n*m-2*n-n*i), dtype=int))) for i in range(m-1)])
    result = sparse.vstack((result, sparse.hstack((sparse.csr_matrix((n-1, m*n-n), dtype=int), a))))
    return result.T

def create_map(x, m, n):
    strategie = [0, 1] * (m - 1) + [0]
    plot_graph = np.zeros((((m - 1) * 2 + 1), (n - 1) * 2 + 1))
    index = 0
    for i in range(len(strategie)):
        if strategie[i] % 2 == 0:
            edges = list(x[index:index + n - 1])
            zeros = [np.nan] * n
            tmp = [None] * (len(zeros) + len(edges))
            tmp[::2] = zeros
            tmp[1::2] = edges
            plot_graph[i, :] = np.array(tmp)
            index += n - 1

        else:
            edges = list(x[index:index + n])
            zeros = [np.nan] * (n - 1)
            tmp = [None] * (len(zeros) + len(edges))
            tmp[::2] = edges
            tmp[1::2] = zeros
            plot_graph[i, :] = np.array(tmp)
            index += n

    return plot_graph