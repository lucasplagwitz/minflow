from minflow.terms import BaseTerm

class Indicator(BaseTerm):

    def __init__(self, mu, nu):
        super(Indicator, self).__init__()
        self.mu = mu.ravel()
        self.nu = nu.ravel()

    def prox(self, nodes):
        return nodes-self.tau*(self.mu - self.nu)