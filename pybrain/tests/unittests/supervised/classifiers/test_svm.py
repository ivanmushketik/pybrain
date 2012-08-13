from pybrain.supervised.classifiers.svm import SVMFactory, SVMType
from pybrain.tests.unittests.supervised.classifiers.utils import createFromGrid
from pybrain.datasets.classification import ClassificationDataSet
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest


class Test(unittest.TestCase):


    def testLinearSeparable(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 1   1           ', #0
                ' 1   1       0  0', #1
                '                 ', #2
                '             0  0', #3
                '                 ', #4
                '    0   0        ', #5
                '                 ', #6
                '    0   0        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        svmf = SVMFactory(estimateProbability=False)
        classifier = svmf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  1)
        self.assertEqual(classifier.getPrediction([4, 1]),  1)
        self.assertEqual(classifier.getPrediction([6, 5]),  0)
        self.assertEqual(classifier.getPrediction([6, 6]),  0)
        self.assertEqual(classifier.getPrediction([15, 2]), 0)
        self.assertEqual(classifier.getPrediction([14, 2]), 0)
        
    def testLinearSeparableWithProbabilityEstimation(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 1   1           ', #0
                ' 1   1       0  0', #1
                '                 ', #2
                '             0  0', #3
                '                 ', #4
                '    0   0        ', #5
                '                 ', #6
                '    0   0        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        svmf = SVMFactory(C=100, estimateProbability=True)
        classifier = svmf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  1)
        self.assertEqual(classifier.getPrediction([4, 1]),  1)
        self.assertEqual(classifier.getPrediction([6, 5]),  0)
        self.assertEqual(classifier.getPrediction([6, 6]),  0)
        self.assertEqual(classifier.getPrediction([15, 2]), 0)
        self.assertEqual(classifier.getPrediction([14, 2]), 0)
        
    def testMulticlassTask(self):
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
        
        svmf = SVMFactory(C=100, estimateProbability=False)
        classifier = svmf.buildClassifier(dataset)
      
        self.assertEqual(classifier.getPrediction([3, 0]),  0)
        self.assertEqual(classifier.getPrediction([4, 1]),  0)
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        self.assertEqual(classifier.getPrediction([15, 2]), 2)
        self.assertEqual(classifier.getPrediction([14, 2]), 2)
        
    def testMulticlassTaskWithProbabilityEstimation(self):
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
        
        svmf = SVMFactory(C=100, estimateProbability=True)
        classifier = svmf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  0)
        self.assertEqual(classifier.getPrediction([4, 1]),  0)
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        self.assertEqual(classifier.getPrediction([15, 2]), 2)
        self.assertEqual(classifier.getPrediction([14, 2]), 2)
        
    def testLinearSeparableNuSVCType(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 1   1           ', #0
                ' 1   1       0  0', #1
                '                 ', #2
                '             0  0', #3
                '                 ', #4
                '    0   0        ', #5
                '                 ', #6
                '    0   0        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        svmf = SVMFactory(estimateProbability=False, svmType=SVMType.NU_SVC)
        classifier = svmf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  1)
        self.assertEqual(classifier.getPrediction([4, 1]),  1)
        self.assertEqual(classifier.getPrediction([6, 5]),  0)
        self.assertEqual(classifier.getPrediction([6, 6]),  0)
        self.assertEqual(classifier.getPrediction([15, 2]), 0)
        self.assertEqual(classifier.getPrediction([14, 2]), 0)
        
    def testLinearSeparableNuSVCTypeEstimateProbability(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 1   1           ', #0
                ' 1   1       0  0', #1
                '                 ', #2
                '             0  0', #3
                '                 ', #4
                '    0   0        ', #5
                '                 ', #6
                '    0   0        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        svmf = SVMFactory(nu = 0.3, estimateProbability=True, svmType=SVMType.NU_SVC)
        classifier = svmf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  1)
        self.assertEqual(classifier.getPrediction([4, 1]),  1)
        self.assertEqual(classifier.getPrediction([6, 5]),  0)
        self.assertEqual(classifier.getPrediction([6, 6]),  0)
        self.assertEqual(classifier.getPrediction([15, 2]), 0)
        self.assertEqual(classifier.getPrediction([14, 2]), 0)
       


        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLinearSeparable']
    unittest.main()