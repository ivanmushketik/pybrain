

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.classifiers.classifier import ClassifierFactory,\
    Classifier
from pybrain.supervised.classifiers.meta.voting import _Voting, MajorVoting

class OneVsAll(ClassifierFactory):
    '''
    Create a multi-class classifier using several binary classifiers. For each class in a multiclass dataset
    it creates a binary classifier that learns to distinguish this class. This method therefore builds N
    classifiers, where N - number of classes in a dataset.
    '''

    def __init__(self, classifierFactory):
        '''
        classifierFactory - factory that will create a binary classifier that will be used to create 
        multiclass classifier
        '''
        self.classifierFactory = classifierFactory
        
    def _build(self, dataset):        
        classifiers = []
        for classValue in range(len(dataset.class_labels)):
            singleClassDataset = self._createDatasetForClass(dataset, classValue)
            classifier = self.classifierFactory.buildClassifier(singleClassDataset)
            
            classifiers.append(classifier)
            
        return _OneVsAllVoting(classifiers)
            
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
    
class _OneVsAllVoting(Classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        
    def getDistribution(self, values):
        # Implementation of variation of voting algorithm
        # If classifier N confident that current instance 
        # belongs to its class, this class receives a vote
        # otherwise all classes except class N receives 
        # a vote
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
    
    def distributionLength(self):
        return len(self.classifiers)

class _AllVsAllVoting(Classifier):    
    def __init__(self, classifiers, pairs, numClasses):
        self.classifiers = classifiers
        self.pairs = pairs
        self.numClasses = numClasses
        
    def getDistribution(self, values):
        votingCount = array([0.] * self.numClasses)
        
        for i, classifier in enumerate(self.classifiers):
            prediction = classifier.getPrediction(values)
            
            # Get class for which current classifier voted
            classValue = self.pairs[i][prediction]
            votingCount[classValue] += 1
            
        distribution = votingCount / votingCount.sum()
        
        return distribution
        
    def distributionLength(self):
        return self.numClasses
    
            
class AllVsAllFactory(ClassifierFactory):
    """
    Multiclass classification algorithm. It traines m*(m - 1)/2 classifiers for each pair of classes.
    To predict result class value every trained classifers vote for one of two classes. The result is 
    selected by majority voting.
    """
    def __init__(self, classifierFactory):
        '''
        classifierFactory - factory that will create a binary classifier that will be used to create 
        multiclass classifier
        '''
        self.classifierFactory = classifierFactory
        
    def _build(self, dataset):
        classifiers = []
        
        classPairs = self._generateClassPairs(dataset)
        
        for pair in classPairs:
            # Create dataset with instances that have only one of two classes in a pair
            pairDataset = self._getFilteredDataset(dataset, pair)
            
            classifier = self.classifierFactory.buildClassifier(pairDataset)
            classifiers.append(classifier)
        
        return _AllVsAllVoting(classifiers, classPairs, dataset.nClasses)
    
    def _generateClassPairs(self, dataset):
        pairs = []
        for i in range(dataset.nClasses):
            for j in range(i, dataset.nClasses):
                if i != j:
                    pairs.append((i, j))
                
        return pairs
    
    def _getFilteredDataset(self, dataset, pair):
        datasetForPair = ClassificationDataSet(dataset.getDimension('input'), nb_classes=2)
        
        for instance in dataset:
            input = instance[0]
            target = instance[1]
            
            classValue = target[0]
            
            # First class in pair is negative class and the second one is a positive class
            if classValue == pair[0]:
                datasetForPair.appendLinked(input, [0])
            elif classValue == pair[1]:
                datasetForPair.appendLinked(input, [1])
                
        return datasetForPair
    