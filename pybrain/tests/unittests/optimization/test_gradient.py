from pybrain.optimization.gradient import GradientOptimizer
from pybrain.tests.helpers import assertListAlmostEqual
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest

# Paraboloid function
def positiveParabaloidEvaluator(values):
    x = values[0]
    y = values[1]
    
    return x * x + y * y

# Calculating derivatives for paraboloid function
def positiveDerivativesCalculator(values):
    currX = values[0]
    currY = values[1]
    
    dx = 2 * currX
    dy = 2 * currY
    
    return [dx, dy]

# Parabaloid function with negation
def negativeParabaloidEvaluator(values):
    x = values[0]
    y = values[1]
    
    return -(x * x + y * y)

# Derivatives of parabaloid function
def negativeDerivativesCalculator(values):
    currX = values[0]
    currY = values[1]
    
    dx = - 2 * currX
    dy = - 2 * currY
    
    return [dx, dy]

class Test(unittest.TestCase):
    
    def testParaboloidOptimizationMinimize(self):
        optimizer = GradientOptimizer(positiveParabaloidEvaluator, [5., 5.], positiveDerivativesCalculator)
        optimizer.maxLearningSteps = 200
        optimizer.verbose = False
        optimizer.minimize = True
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMaximize(self):
        optimizer = GradientOptimizer(negativeParabaloidEvaluator, [5., 5.], negativeDerivativesCalculator)
        optimizer.maxLearningSteps = 200
        optimizer.verbose = False
        optimizer.minimize = False
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)


from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
