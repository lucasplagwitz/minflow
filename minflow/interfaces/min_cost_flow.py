from minflow.terms import CostFlow, Indicator
from minflow.solver.pd_hgm import PdHgm
from minflow.helper import direct_connection

import numpy as np

class MinCostFlow(object):

    def __init__(self, mu, nu, tau=0.1):
        self.mu = mu
        self.nu = nu
        if mu.shape == nu.shape:
            self.domain_shape = mu.shape
        else:
            raise ValueError("Dimension missmatch!")
        self.tau = tau
        self.G = CostFlow()
        self.F_star = Indicator(self.mu, self.nu)
        self.K = direct_connection(self.domain_shape[0], self.domain_shape[1])

        self.G.set_proxparam(self.tau)
        self.F_star.set_proxparam(self.tau)

    def solve(self, max_iter: int = 1450, tol: float = 5*10**(-6)):

        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return self.solver.var['x']