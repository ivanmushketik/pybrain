

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.classifiers.classifier import ClassifierFactory,\
    Classifier
from pybrain.supervised.classifiers.meta.voting import _Voting, MajorVoting

class OneVsAll(ClassifierFactory):
    '''
    classdocs
    '''

    def __init__(self, classifierFactory):
        '''
        Constructor
        '''
        self.classifierFactory = classifierFactory
        
    def _build(self, dataset):        
        classifiers = []
        for classValue, className in enumerate(dataset.class_labels):
            singleClassDataset = self._createDatasetForClass(dataset, classValue)
            classifier = self.classifierFactory.buildClassifier(singleClassDataset)
            
            classifiers.append(classifier)
            
        return _OneVsAll(classifiers)
            
    def _createDatasetForClass(self, dataset, classValue):
        datasetForClass = ClassificationDataSet(dataset.getDimension('input'), nb_classes=2)
        
        for instance in dataset:
            input = instance[0]
            target = instance[1]
            
            if target[0] == classValue:
                datasetForClass.appendLinked(input, [1])
            else:
                datasetForClass.appendLinked(input, [0])
                
        return datasetForClass 
    
class _OneVsAll(Classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        
    def getDistribution(self, values):
        count = array([0.] * len(self.classifiers))
        for indx, classifier in enumerate(self.classifiers):
            classDistribution = classifier.getDistribution(values)
            if classDistribution[1] > classDistribution[0]:
                count[indx] += 1
            else:
                for i in range(len(count)):
                    if i != indx:
                        count[i] += 1
                        
        distribution = count / count.sum()
        
        return distribution
            