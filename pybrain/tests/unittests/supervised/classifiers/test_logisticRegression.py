from pybrain.supervised.classifiers.logisticRegression import LogisticRegressionFactory,\
    _LogisticRegression
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tests.helpers import assertListAlmostEqual
from pybrain.tools.functions import sigmoid
from pybrain.optimization.gradient import GradientOptimizer
from pybrain.auxiliary.gradientdescent import GradientDescent
from pybrain.tests.unittests.supervised.classifiers.utils import createFromGrid
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

import unittest

class LogisticRegressionTest(unittest.TestCase):
    def testDistribution(self):
        input = [1, 2]
        thetas = [0.5, 0.4, 0.3]
        classifier = _LogisticRegression(thetas)
        
        distribution = classifier.getDistribution(input)
        posProbability = sigmoid(0.5 * 1 + 0.4 * 1 + 0.3 * 2)
        expectedDistribution = [1 - posProbability, posProbability]
        
        assertListAlmostEqual(self, distribution, expectedDistribution, 0.0001)

        

class LogisticRegressionFactoryTest(unittest.TestCase):

    def testOneDimensionalClassificationTest(self):
        # Imaginable medical dataset. If tumor is small it is benigh, otherwise it is malignant.
        tumorDataset = ClassificationDataSet(1, nb_classes=2, class_labels=['Benign', 'Malignant'])
        tumorDataset.appendLinked([ 0.1  ] , [0])
        tumorDataset.appendLinked([ 0.15 ] , [0])
        tumorDataset.appendLinked([ 0.2  ] , [0])
        tumorDataset.appendLinked([ 0.33 ] , [0])
        tumorDataset.appendLinked([ 0.23 ] , [0])
        tumorDataset.appendLinked([ 0.4 ]  , [0])
        tumorDataset.appendLinked([ 0.8 ] , [1])
        tumorDataset.appendLinked([ 1.4 ] , [1])
        tumorDataset.appendLinked([ 2.3 ] , [1])
        tumorDataset.appendLinked([ 0.9 ] , [1])
        tumorDataset.appendLinked([ 1.9 ] , [1])
        tumorDataset.appendLinked([ 2.9 ] , [1])
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(tumorDataset)
        
        self.assertEqual(classifier.getPrediction([0.2]),   0)
        self.assertEqual(classifier.getPrediction([0.1]),   0)
        self.assertEqual(classifier.getPrediction([0.3]),   0)
        self.assertEqual(classifier.getPrediction([0.001]), 0)
        
        self.assertEqual(classifier.getPrediction([1.2]),   1)
        self.assertEqual(classifier.getPrediction([2.2]),   1)
        self.assertEqual(classifier.getPrediction([3.2]),   1)
        self.assertEqual(classifier.getPrediction([1.9]),   1)
        
    
    def testTwoDimensionalLinearClassificationTest(self):
        # Class number one - points from I quadrant of Cartesian coordinates
        # Class number two - points from III quadrant of Cartesian coordinates
        testDataset = ClassificationDataSet(2, nb_classes=2, class_labels=['I-Quadrant', 'III-Quadrant'])
        testDataset.appendLinked([ 2, 2 ]  , [0])
        testDataset.appendLinked([ 4, 2 ]  , [0])
        testDataset.appendLinked([ 5, 0 ]  , [0])
        testDataset.appendLinked([ 0, 5 ]  , [0])
        testDataset.appendLinked([ 3, 2 ]  , [0])
        testDataset.appendLinked([ 8, 1 ]  , [0])
        testDataset.appendLinked([ 1, 8 ]  , [0])
        testDataset.appendLinked([ -4, -2 ] , [1])
        testDataset.appendLinked([ -3, -2 ] , [1])
        testDataset.appendLinked([ -8, -1 ] , [1])
        testDataset.appendLinked([ -1, -5 ] , [1])
        testDataset.appendLinked([ -2, -2 ] , [1])
        testDataset.appendLinked([ -5, -5 ] , [1])
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(testDataset)
        
        self.assertEqual(classifier.getPrediction([3, 3]),   0)
        self.assertEqual(classifier.getPrediction([2, 4]),   0)
        self.assertEqual(classifier.getPrediction([10, 10]), 0)
        self.assertEqual(classifier.getPrediction([9, 5]),   0)
        
        self.assertEqual(classifier.getPrediction([-4, -4]),   1)
        self.assertEqual(classifier.getPrediction([-20, -20]),   1)
        self.assertEqual(classifier.getPrediction([-8, -3]),   1)
        self.assertEqual(classifier.getPrediction([-9, -9]),   1)        

    def testTwoDimensionalClassification1(self):
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
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  1)
        self.assertEqual(classifier.getPrediction([4, 1]),  1)
        
    def testTwoDimensionalClassification2(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0           ', #0
                ' 0   0       0  0', #1
                '                 ', #2
                '             0  0', #3
                '                 ', #4
                '    1   1        ', #5
                '                 ', #6
                '    1   1        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        
    def testTwoDimensionalClassification3(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0           ', #0
                ' 0   0       1  1', #1
                '                 ', #2
                '             1  1', #3
                '                 ', #4
                '    0   0        ', #5
                '                 ', #6
                '    0   0        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(dataset)
                 
        self.assertEqual(classifier.getPrediction([15, 2]), 1)
        self.assertEqual(classifier.getPrediction([14, 2]), 1)
        
    def testTwoDimensionalClassification4(self):
        # Coordinates plane for classification task
        #                  1111111
        # X      01234567890123456   #Y
        grid = [' 0   0           ', #0
                ' 0   0           ', #1
                '                 ', #2
                '                 ', #3
                '                 ', #4
                '    1   1        ', #5
                '                 ', #6
                '    1   1        ', #7
                
                ]
        
        dataset = ClassificationDataSet(2, nb_classes=2, class_labels= ['0', '1'])
        
        createFromGrid(grid, dataset)
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction([3, 0]),  0)
        self.assertEqual(classifier.getPrediction([4, 1]),  0)
        self.assertEqual(classifier.getPrediction([6, 5]),  1)
        self.assertEqual(classifier.getPrediction([6, 6]),  1)
        
    def testNonLinearClassificaion(self):
        # This tests checks if logistic regression classifier will able
        # to classify non linear separable data
        # Data set consists of two classes were elements of class 0
        # are inside the circle of radius 0.5 and elements with class 1 
        # are outside this circle
        dataset = ClassificationDataSet(4, nb_classes=2, class_labels= ['0', '1'])
        
        dataset.appendLinked(self._getSquaredTerms([0.5,  0.5]),  [0])
        dataset.appendLinked(self._getSquaredTerms([0,    0.5]),  [0])
        dataset.appendLinked(self._getSquaredTerms([0.5,  0]),    [0])
        dataset.appendLinked(self._getSquaredTerms([0,    0]),    [0])
        dataset.appendLinked(self._getSquaredTerms([0.3,  0.2]),  [0])
        dataset.appendLinked(self._getSquaredTerms([0.3,  -0.5]), [0])
        dataset.appendLinked(self._getSquaredTerms([-0.3, 0.5]),  [0])
        dataset.appendLinked(self._getSquaredTerms([-0.4, 0]),    [0])
        dataset.appendLinked(self._getSquaredTerms([0.6,  -0.3]), [0])
        dataset.appendLinked(self._getSquaredTerms([-0.3, -0.5]), [0])
        
        dataset.appendLinked(self._getSquaredTerms([2,  4]),  [1])
        dataset.appendLinked(self._getSquaredTerms([4,  -5]), [1])
        dataset.appendLinked(self._getSquaredTerms([-3, 2]),  [1])
        dataset.appendLinked(self._getSquaredTerms([4,  4]),  [1])
        dataset.appendLinked(self._getSquaredTerms([-3, 5]),  [1])
        dataset.appendLinked(self._getSquaredTerms([-2, -4]), [1])
        dataset.appendLinked(self._getSquaredTerms([-5, 0]),  [1])
        dataset.appendLinked(self._getSquaredTerms([5,  0]),  [1])
        dataset.appendLinked(self._getSquaredTerms([4,  3]),  [1])
        dataset.appendLinked(self._getSquaredTerms([-5, 1]),  [1])
        
        optimizer = GradientOptimizer(minChange=1e-6)
        optimizer.maxLearningSteps = 1000
        optimizer.verbose = False
        lrf = LogisticRegressionFactory(optimizer)
        classifier = lrf.buildClassifier(dataset)
        
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([0.1,  0.1])),  0)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([0.1,  -0.1])), 0)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([-0.1, 0.1])),  0)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([-0.1, -0.1])), 0)
        
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([4,  4])),  1)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([4,  -4])), 1)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([-4, 4])),  1)
        self.assertEqual(classifier.getPrediction(self._getSquaredTerms([-4, -4])), 1)
        
        
    def _getSquaredTerms(self, vector):
        """
        Get values in vector and squared values concatenated in a single result vector
        """
        result = list(vector)
        
        for el in vector:
            result.append(el ** 2)
        return result

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))