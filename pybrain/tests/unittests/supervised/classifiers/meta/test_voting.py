
import unittest
from numpy import array

from pybrain.tests.helpers import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.classifiers.classifier import ClassfierFactory, Classifier
from pybrain.supervised.classifiers.meta.voting import VotingFactory, MajorVoting, SumRule, MedianRule, MaximumProbabilityRule, MinimumProbabilityRule, ProductRule

class MockFactory(ClassfierFactory):

    def __init__(self, classifier):
        self.classifier = classifier

    def _build(self, dataset):
        return self.classifier


class ConstantClassifier(Classifier):
    def __init__(self, returnValues):
        self.returnValues = array(returnValues)

    def getDistribution(self, values):
        return self.returnValues

class VotingTestCase(unittest.TestCase):
    def testMajorVoting(self):
        classifier = self.createVoting(MajorVoting(), [1, 0, 0], [1, 0, 0], [0, 0, 1])
         
        result = classifier.getPrediction([])
        expected = 0

        self.assertEqual(result, expected)
        
    def createVoting(self, combinationRule, *distributions):
        classifiersFactory = [MockFactory(ConstantClassifier(distribution)) for distribution in distributions]
        voteFactory = VotingFactory(classifiersFactory, combinationRule)
        classifeir = voteFactory.buildClassifeir(SupervisedDataSet(1, 3))
        return classifeir
        
    def testSumRule(self):
        classifier = self.createVoting(SumRule(), [0.2, 0.5, 0.3], [0, 0.6, 0.4], [0.4, 0.4, 0.2])
        
        result = classifier.getDistribution([])
        expected = array([0.2, 0.5, 0.3])
        
        assertListAlmostEqual(self, result, expected, 0.00001)
        
    def testMedianRule(self):
        classifier = self.createVoting(MedianRule(), [0.2, 0.5, 0.3], [0, 0.6, 0.4], [0.4, 0.4, 0.2])
        
        result = classifier.getDistribution([])
        expected = array([0.2, 0.5, 0.3])
        
        assertListAlmostEqual(self, result, expected, 0.00001)
        
    def testMaximumProbabilityRule(self):
        classifier = self.createVoting(MaximumProbabilityRule(), [0.2, 0.5, 0.3], [0, 0.6, 0.4], [0.4, 0.4, 0.2])
        
        result = classifier.getDistribution([])
        expected = array([0.4, 0.6, 0.4])
        
        assertListAlmostEqual(self, result, expected, 0.00001)
        
    def testMinimumProbabilityRule(self):
        classifier = self.createVoting(MinimumProbabilityRule(), [0.2, 0.5, 0.3], [0, 0.6, 0.4], [0.4, 0.4, 0.2])
        
        result = classifier.getDistribution([])
        expected = array([0., 0.4, 0.2])
        
        assertListAlmostEqual(self, result, expected, 0.00001)
        
    def testProductRule(self):
        classifier = self.createVoting(ProductRule(), [0.2, 0.5, 0.3], [0, 0.6, 0.4], [0.4, 0.4, 0.2])
        
        result = classifier.getDistribution([])
        expected = array([0., 0.12, 0.024])
        
        assertListAlmostEqual(self, result, expected, 0.00001)

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
