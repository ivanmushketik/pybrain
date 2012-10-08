from pybrain.supervised.classifiers.meta.voting import _Voting, MajorVoting
from pybrain.supervised.classifiers.meta.grading import _Grading, GradingFactory
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.classifiers.logisticRegression import LogisticRegressionFactory
from pybrain.optimization.gradient import GradientOptimizer
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest
from pybrain.tests.unittests.supervised.classifiers.utils import ConstantClassifier,\
    createFromGrid
from numpy import *


class TestGrading(unittest.TestCase):


    def testGrading(self):        
        # Test if all classifiers were graded as incorrect
        gradingClassifier = _Grading(
                        [ConstantClassifier([0, 1]), ConstantClassifier([0, 1]), ConstantClassifier([1, 0]) ],
                        [ConstantClassifier([0]), ConstantClassifier([0]), ConstantClassifier([0])], 
                        MajorVoting())
        
        prediction = gradingClassifier.getPrediction(array([1, 2, 3]))
        self.assertEqual(prediction, 1)       
        
    def testGrading2(self):        
        # Test if all classifiers were graded as correct
        gradingClassifier = _Grading(
                        [ConstantClassifier([0, 1]), ConstantClassifier([0, 1]), ConstantClassifier([1, 0]) ],
                        [ConstantClassifier([1]), ConstantClassifier([1]), ConstantClassifier([1])], 
                        MajorVoting())
        
        prediction = gradingClassifier.getPrediction(array([1, 2, 3]))
        self.assertEqual(prediction, 1)        
        
    def testGrading3(self):        
        # Test if only some classifiers were graded as correct
        gradingClassifier = _Grading(
                        [ConstantClassifier([0, 1]), ConstantClassifier([0, 1]), ConstantClassifier([1, 0]),  ConstantClassifier([1, 0]), ConstantClassifier([1, 0])],
                        [ConstantClassifier([1]),    ConstantClassifier([1]),    ConstantClassifier([1]),     ConstantClassifier([0]),    ConstantClassifier([0])], 
                        MajorVoting())
        
        prediction = gradingClassifier.getPrediction(array([1, 2, 3]))
        self.assertEqual(prediction, 1)                                     
        
    def testGradingTraining(self):
        
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 1 1 1 1 1 1    0', #0
                ' 1 1 1   1      0', #1
                ' 1 1  1         0', #2
                '  1   1      0  0', #3
                ' 1 1       0 0 0 ', #4
                ' 1  1      0 0  0', #5
                '  1        0 0 00', #6
                ' 1     0   0 00 0', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        gradingFactory = GradingFactory([lrf, lrf, lrf], lrf, 0.7, MajorVoting)
        
        gradingClassifier = gradingFactory.buildClassifier(dataset)
        
        self.assertEqual(gradingClassifier.getPrediction([0, 0]), 1)
        self.assertEqual(gradingClassifier.getPrediction([16, 7]), 0)
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGrading']
    unittest.main()