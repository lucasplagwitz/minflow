from minflow.terms import BaseTerm

class CostFlow(BaseTerm):

    def __init__(self):
        super(CostFlow, self).__init__()

    def prox_star(self, edges):
        edges[edges > 1] = 1
        edges[edges < -1] = -1
        return edges

    def prox(self, edges):
        return edges - self.tau * self.prox_star(edges/self.tau)