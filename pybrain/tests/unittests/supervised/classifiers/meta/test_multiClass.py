from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tests.unittests.supervised.classifiers.utils import createFromGrid,\
    ConstantClassifier

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest
from pybrain.supervised.classifiers.meta.multiClass import OneVsAllFactory, _OneVsAllVoting,\
    AllVsAllFactory, _AllVsAllVoting
from pybrain.optimization.gradient import GradientOptimizer
from pybrain.supervised.classifiers.logisticRegression import LogisticRegressionFactory

class TestOveVsAllClassifier(unittest.TestCase):
    def testClassifier(self):
        classifiers = [ConstantClassifier([1, 0]),
                       ConstantClassifier([1, 0]),
                       ConstantClassifier([0, 1])
                       ]
        classifier = _OneVsAllVoting(classifiers)
        self.assertEqual(classifier.getPrediction([]), 2)

class TestOneVsAllFactory(unittest.TestCase):
    def testLogisticWithMulticlass(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0           ', #0
                ' 0   0       2  2', #1
                '                 ', #2
                '             2  2', #3
                '                 ', #4
                '    1   1        ', #5
                '                 ', #6
                '    1   1        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=3, class_labels= ['0', '1', '2'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        oneVsAll = OneVsAllFactory(lrf)
        classifier = oneVsAll.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  0)
        self.assertEqual(classifier.getPrediction([4, 1]),  0)
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        self.assertEqual(classifier.getPrediction([15, 2]), 2)
        self.assertEqual(classifier.getPrediction([14, 2]), 2)
        
class TestAllVsAllClassifier(unittest.TestCase):
    def testClassifier(self):
        classifiers = [
                       ConstantClassifier([1, 0]), # (0, 1)
                       ConstantClassifier([1, 0]), # (0, 2)
                       ConstantClassifier([0, 1]), # (1, 2)
                       ]
        pairs = [(0, 1), (0,2), (1, 2)]
        classifier = _AllVsAllVoting(classifiers, pairs, 3)
        self.assertEqual(classifier.getPrediction([]), 0)
        
class TestAllVsAllFactory(unittest.TestCase):
    def testLogisticWithMulticlass(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0           ', #0
                ' 0   0       2  2', #1
                '                 ', #2
                '             2  2', #3
                '                 ', #4
                '    1   1        ', #5
                '                 ', #6
                '    1   1        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=3, class_labels= ['0', '1', '2'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        allVsAll = AllVsAllFactory(lrf)
        classifier = allVsAll.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  0)
        self.assertEqual(classifier.getPrediction([4, 1]),  0)
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        self.assertEqual(classifier.getPrediction([15, 2]), 2)
        self.assertEqual(classifier.getPrediction([14, 2]), 2)  

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLogisticWithMulticlass']
    unittest.main()