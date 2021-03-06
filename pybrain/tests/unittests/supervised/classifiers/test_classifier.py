__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
import unittest
from pybrain.supervised.classifiers.classifier import Classifier

class TestClassifier(Classifier):
    def __init__(self, distribution):
        self._distribution = array(distribution)
    
    def getDistribution(self, input):
        return self._distribution

class ClassifiersTest(unittest.TestCase):
    def testGetPrediction(self):
        tc = TestClassifier([1, 0, 0])
        result = tc.getPrediction([])
        expected = 0
        self.assertEqual(result, expected)


from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))