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
    
    def _createPositiveOptimizer(self, startParams):
        optimizer = GradientOptimizer(positiveParabaloidEvaluator, startParams, positiveDerivativesCalculator)
        optimizer.maxLearningSteps = 200
        optimizer.minimize = True
        
        return optimizer
    
    ### Testing that minimization will work correctly from any starting point
    
    def testParaboloidOptimizationMinimize1(self):
        optimizer = self._createPositiveOptimizer([5., 5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMinimize2(self):
        optimizer = self._createPositiveOptimizer([5., -5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMinimize3(self):
        optimizer = self._createPositiveOptimizer([-5., 5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMinimize4(self):
        optimizer = self._createPositiveOptimizer([-5., -5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def _createNegativeOptimizer(self, startParams):
        optimizer = GradientOptimizer(negativeParabaloidEvaluator, startParams, negativeDerivativesCalculator)
        optimizer.maxLearningSteps = 200
        optimizer.minimize = False
        
        return optimizer
    
    ### Testing that maximization will work correctly from any starting point
    
    def testParaboloidOptimizationMaximize1(self):
        optimizer = self._createNegativeOptimizer([5., 5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMaximize2(self):
        optimizer = self._createNegativeOptimizer([5., -5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMaximize3(self):
        optimizer = self._createNegativeOptimizer([-5., 5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        
    def testParaboloidOptimizationMaximize4(self):
        optimizer = self._createNegativeOptimizer([-5., -5.])
        result, f = optimizer.learn()
        
        assertListAlmostEqual(self, result, [0., 0.], 0.001)

    def testMinChange(self):
        optimizer = self._createPositiveOptimizer([5., 5.])
        # This is more than enough to optimize the function
        optimizer.maxLearningSteps = 10000
        optimizer.minChange = 1e-6
        result, f = optimizer.learn()

        assertListAlmostEqual(self, result, [0., 0.], 0.001)
        self.assertLess(optimizer.numLearningSteps, optimizer.maxLearningSteps)

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
