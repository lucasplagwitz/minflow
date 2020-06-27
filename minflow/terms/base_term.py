class BaseTerm(object):

    def __init__(self):
        pass

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def prox(self, f):
        """
        proximal operator of term
        """
        pass