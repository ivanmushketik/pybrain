__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
from pybrain.optimization.optimizer import ContinuousOptimizer
from pybrain.auxiliary.gradientdescent import GradientDescent
    
class GradientOptimizer(ContinuousOptimizer):
    """
    Derivaties based function optimization
    """
    def __init__(self, evaluator, initEvaluable, gradientsCalculator, gradient = GradientDescent(), minChange = 1e-6):
        """
        evaluator - the same as in superclass
        initEvaluable - the same as in superclass
        gradientsCalculator - function that calculates partial derivatives for each variable for specified coordinates. 
        It should accept the only parameter - list of current values of parameters and should return an array of partial derivatives values.
        Value of partial derivative for parameter X should have the same position in the result array as parameter X has in parameters array.
        gradient (GradientDescent) - class that changes values of parameters using current gradient
        minChange -  
        """
        ContinuousOptimizer.__init__(self, evaluator, initEvaluable)
        self.gradientsCalculator = gradientsCalculator
        self.gradient = gradient
        self.minChange = minChange
        self.params = array(initEvaluable)
        
        self.gradient.init(self.params)
        
    #TODO: use minChange
    def _learnStep(self):
        gradients = self._calculateGradient()
        assert len(gradients) == len(self.params)
        
        self.params = self.gradient(gradients)
        self._oneEvaluation(self.params)
        
    def _calculateGradient(self):
        gradients = array(self.gradientsCalculator(self.params))
        
        if self.minimize:
            return -gradients
        else:
            return gradients
        