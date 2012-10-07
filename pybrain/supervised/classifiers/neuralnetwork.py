
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from random import Random
from pybrain.supervised.classifiers.classifier import ClassifierFactory, Classifier
from pybrain.datasets.classification import ClassificationDataSet

class NeuralNetworkFactory(ClassifierFactory):
    #TODO: Add documentation
    """
    classdocs
    """

    def __init__(self, module, trainer, iterationsNum=10, seed=1):
        self._module = module
        self._trainer = trainer
        self._iterationsNum = iterationsNum
        self._seed = 1
        
    def _build(self, dataset):
        if isinstance(dataset, ClassificationDataSet):
            dataset._convertToOneOfMany()
        
        random = Random()
        random.seed(self._seed)
        #self._module.randomize(random)
        self._module.randomize()
        
        self._trainer.setData(dataset)
        self._trainer.trainOnDataset(dataset, self._iterationsNum)
            
        return _NeuralNetworkClassifier(self._module)
    
    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module):
        if not module:
            raise ValueError("Module should not be None")
        
        self._module = module
        
    @property
    def trainer(self):
        return self._trainer
    
    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer
        self._trainer.module = self._module
        
        
class _NeuralNetworkClassifier(Classifier):
    def __init__(self, module):
        self._module = module
    
    def getDistribution(self, input):
        return self._module.activate(input)
        