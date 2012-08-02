
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest
from pybrain.supervised.classifiers.meta.stacking import StackingFactory
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tests.unittests.supervised.classifiers.utils import createFromGrid
from pybrain.optimization.gradient import GradientOptimizer
from pybrain.supervised.classifiers.logisticRegression import LogisticRegressionFactory

class Test(unittest.TestCase):

    # Test stacking using distributions of level-0 classifiers
    def testUsingDistributions(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0     11  11', #0
                ' 0   0     1 1  1', #1
                '   0           1 ', #2
                '   0  0      1  1', #3
                '  0           1 1', #4
                ' 0  0   0     111', #5
                '      0   0    11', #6
                '  0 0   0  0   11', #7
                
                ]

        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        
        stackingFactory = StackingFactory([lrf, lrf], lrf)        
        stackingClassifier = stackingFactory.buildClassifier(dataset)
                
        self.assertEqual(stackingClassifier.getPrediction([2, 6]), 0)
        self.assertEqual(stackingClassifier.getPrediction([1, 8]), 0)
        self.assertEqual(stackingClassifier.getPrediction([3, 7]), 0)
        self.assertEqual(stackingClassifier.getPrediction([14, 2]), 1)
        self.assertEqual(stackingClassifier.getPrediction([13, 0]), 1)
        self.assertEqual(stackingClassifier.getPrediction([16, 4]), 1)
        
    # Test stacking using predictions of level-0 classifiers
    def testUsingPredictions(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0     11  11', #0
                ' 0   0     1 1  1', #1
                '   0           1 ', #2
                '   0  0      1  1', #3
                '  0           1 1', #4
                ' 0  0   0     111', #5
                '      0   0    11', #6
                '  0 0   0  0   11', #7
                
                ]

        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        
        stackingFactory = StackingFactory([lrf, lrf], lrf, useDistributions=True)
        stackingClassifier = stackingFactory.buildClassifier(dataset)
                
        self.assertEqual(stackingClassifier.getPrediction([2, 6]), 0)
        self.assertEqual(stackingClassifier.getPrediction([1, 8]), 0)
        self.assertEqual(stackingClassifier.getPrediction([3, 7]), 0)
        self.assertEqual(stackingClassifier.getPrediction([14, 2]), 1)
        self.assertEqual(stackingClassifier.getPrediction([13, 0]), 1)
        self.assertEqual(stackingClassifier.getPrediction([16, 4]), 1)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLogisticClassification']
    unittest.main()