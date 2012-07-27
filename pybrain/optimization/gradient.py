__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import sys
from numpy import *
from pybrain.optimization.optimizer import ContinuousOptimizer
from pybrain.auxiliary.gradientdescent import GradientDescent
    
class GradientOptimizer(ContinuousOptimizer):
    """
    Derivaties based function optimization
    """
    def __init__(self, evaluator=None, initEvaluable=None, gradientsCalculator=None, gradient = GradientDescent(), minChange = 1e-6, minimize = True):
        """
        evaluator - the same as in superclass
        initEvaluable - the same as in superclass
        gradientsCalculator - function that calculates partial derivatives for each variable for specified coordinates. 
        It should accept the only parameter - list of current values of parameters and should return an array of partial derivatives values.
        Value of partial derivative for parameter X should have the same position in the result array as parameter X has in parameters array.
        gradient (GradientDescent) - class that changes values of parameters using current gradient
        minChange -  
        """
        
        self.gradientsCalculator = gradientsCalculator
        self.gradient = gradient
        self.minChange = minChange        
        
        self.prevFitness = sys.float_info.max
        self.currentFitness = sys.float_info.min     
        
        initEvaluable = array(initEvaluable)
        ContinuousOptimizer.__init__(self, evaluator, initEvaluable)
        self.minimize = minimize
    
    def _setInitEvaluable(self, evaluable):
        evaluable = array(evaluable)
        ContinuousOptimizer._setInitEvaluable(self, evaluable)
        
        self.gradient.init(evaluable)
        self.prevFitness = sys.float_info.max
        self.currentFitness = sys.float_info.min     
        
    def _learnStep(self):
        gradients = self._calculateGradient()
        assert len(gradients) == len(self.bestEvaluable)
        
        params = self.gradient(gradients)
        self.currentFitness = self._oneEvaluation(params)
        
    def _calculateGradient(self):
        gradients = array(self.gradientsCalculator(self.bestEvaluable))
        
        # If we minimize the function we should substract gradients from the current parameter values
        if self.minimize:
            return -gradients
        else:
            return gradients
        
    def _stoppingCriterion(self):
        result = False
        
        if ContinuousOptimizer._stoppingCriterion(self):
            result =  True
        # Check if during the last step we made a significant progress
        elif abs(self.prevFitness - self.currentFitness) < self.minChange:   
            result =  True
        
        self.prevFitness = self.currentFitness
        
        return result