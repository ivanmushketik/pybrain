
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com' 

import unittest
from pybrain.supervised.classifiers.neuralnetwork import NeuralNetworkFactory
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.datasets.classification import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

class Test(unittest.TestCase):

    #FIXME: NN does not learn how to classify from this simple array. Must be wrong parameters for NN trainer
    def _testTrainingOnClassificationDataset(self):
        DS = ClassificationDataSet(2, class_labels=['Zero', 'One'])
        DS.appendLinked([ 0, 0 ] , [0])
        DS.appendLinked([ 0, 1 ] , [0])
        DS.appendLinked([ 1, 0 ] , [0])
        DS.appendLinked([ 1, 1 ] , [1])
        
        network = buildNetwork(DS.indim, 5, 2, outclass=SoftmaxLayer)
        trainer = BackpropTrainer( network, momentum=0.1, verbose=True, weightdecay=0.01)
        
        nnf = NeuralNetworkFactory(network, trainer, seed=2, iterationsNum=20)
        nnClassifier = nnf.buildClassifier(DS)
        
        self.assertEqual(nnClassifier.getPrediction([0, 0]), 0) 
        self.assertEqual(nnClassifier.getPrediction([0, 1]), 0)
        self.assertEqual(nnClassifier.getPrediction([1, 0]), 0)
        self.assertEqual(nnClassifier.getPrediction([1, 1]), 1) 
        
    # FIXME: This test fails sometimes because of the generating random values somewhere in the code.
    # This need to be fixed to make tests/training repeatable
    def testTrainingOnSepervisedDataset(self):
        DS = SupervisedDataSet(2, 1)
        DS.addSample([ 0, 0 ] , [0])
        DS.addSample([ 0, 1 ] , [1])
        DS.addSample([ 1, 0 ] , [1])
        DS.addSample([ 1, 1 ] , [0])
        
        network = N = buildNetwork(2, 4, 1)
        trainer = BackpropTrainer(N, learningrate = 0.01, momentum = 0.99)
        trainer.verbose = True
        
        nnf = NeuralNetworkFactory(network, trainer, seed=2, iterationsNum=500)
        nnClassifier = nnf.buildClassifier(DS)
        
        self.assertAlmostEqual(nnClassifier.getPrediction([0, 0]), 0, delta=0.01) 
        self.assertAlmostEqual(nnClassifier.getPrediction([0, 1]), 1, delta=0.01)
        self.assertAlmostEqual(nnClassifier.getPrediction([1, 0]), 1, delta=0.01)
        self.assertAlmostEqual(nnClassifier.getPrediction([1, 1]), 0, delta=0.01)  


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))