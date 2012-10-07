from numpy import *
from pybrain.supervised.classifiers.classifier import ClassifierFactory,\
    Classifier
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

class MockFactory(ClassifierFactory):

    def __init__(self, classifier):
        self.classifier = classifier

    def _build(self, dataset):
        return self.classifier

    

class ConstantClassifier(Classifier):
    def __init__(self, returnValues):
        Classifier.__init__(self, len(returnValues))

        self.returnValues = array(returnValues)

    def getDistribution(self, values):
        return self.returnValues
    
    def distributionLength(self):
        return len(self.returnValues)

def createFromGrid(grid, dataset):
    for rNum, row in enumerate(grid):
        for cNum, value in enumerate(row):
            if value.isdigit():
                classValue = int(value)
                dataset.appendLinked([cNum, rNum], [value])
                
    
        