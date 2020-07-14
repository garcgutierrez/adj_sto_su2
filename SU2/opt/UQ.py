from pylab import *
import chaospy as cp

class UQ(object):
    def __init__(self):
        self.alpha_dist = cp.Uniform(-0.5,0.5)
        self.Ma_dist = cp.Uniform(0.1,0.2)
        self.T = cp.Uniform(273, 274)
        self.distribution = cp.J(self.alpha_dist, self.Ma_dist)
        self.computeQuadrature()
        
    def computeQuadrature(self, nOrder=2, ruleN='C'):
        
        self.absissas, self.weights = cp.generate_quadrature(
            order = nOrder, dist=self.distribution, rule=ruleN)
        self.Machs = around(array(self.absissas)[1,:],2)
        self.AOAs = around(array(self.absissas)[0,:],3)
        self.Nquadrature = len(self.Machs)
        self.polynomial_expansion = cp.orth_ttr(nOrder, self.distribution)

        
    def computeProperties(self, numArray, debug=True):
        if(debug):
            print('shape: {}'.format(shape(numArray)))
            print('Nq:{}'.format(self.Nquadrature))
            print('Variables:{}'.format(numArray))
        self.poly_approx = cp.fit_quadrature(
            self.polynomial_expansion, self.absissas,
            self.weights, numArray)

        mean = cp.E(self.poly_approx, self.distribution)
        sigma = cp.Std(self.poly_approx, self.distribution)
        return mean, sigma 
