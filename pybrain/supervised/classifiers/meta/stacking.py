
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
import itertools
from pybrain.supervised.classifiers.classifier import ClassifierFactory
from pybrain.supervised.classifiers.meta.voting import CombinationRule, _Voting
from pybrain.datasets.classification import ClassificationDataSet

class StackingFactory(ClassifierFactory):
    '''
    Generalization of voting method method. Several classifiers (called level-0 classifiers) are produce 
    prediction of classification and a separate classifier (level-1 classifier) is making 
    the final prediction using predictions of level-1 classifiers
    '''
    def __init__(self, level0classifiersFactories, level1classifierFactory, useDistributions=True, splitProporition = 0.25):
        '''
        level0ClassifiersFactories(iter(ClassifierFactory)) - iterable of classifiers factories that will be used to produce level-0 classifiers
        level1ClassifierFactory(ClassifierFactory) - factory of the level-1 classifier
        useDistribution(bool) - if True array of concatenated distributions will be used as input for level-1 classifier
        splitProportion(float) (0, 1) - percent of train instances that will be used to train level-1 classifier  
        '''
        self.l0factories = level0classifiersFactories
        self.l1factory = level1classifierFactory
        self.useDistributions = useDistributions
        self.splitProportion = splitProporition
        
    def _build(self, dataset):
        l1PreDataset, l0Dataset = dataset.splitWithProportion( self.splitProportion )
        
        # Train level-0 classifiers
        classifiers = [factory.buildClassifier(l0Dataset) for factory in self.l0factories]
        # Convert initial data to level-1 dataset
        l1Dataset = self._createL1Dataset(classifiers, l1PreDataset)
        
        l1Classifier = self.l1factory.buildClassifier(l1Dataset)
        
        return _Voting(classifiers, StackingCombinationRule(l1Classifier, self.useDistributions))
        
    def _createL1Dataset(self, classifiers, l1PreDataset):
        l1DatasetDimensions = classifiers[0].distributionLength() * len(classifiers)
        l1Dataset = ClassificationDataSet(l1DatasetDimensions, nb_classes=2)
        
        for instance in l1PreDataset:
            input = instance[0]
            target = instance[1]            
            
            l1Input = _getLevel1Input(classifiers, input, self.useDistributions)
            l1Dataset.appendLinked(l1Input, target)
            
        return l1Dataset

def _getLevel1Input(classifiers, input, useDistributions):
    """
    Get feature vector for level-1 training set.
    classifiers - list of level-0 classifiers that will make a prediction
    input - input vector from the original dataset
    useDistribution - if True use distributions instead of predictions 
    """
    inputsList = []
    
    for classifier in classifiers:
        if useDistributions:
            inputsList.append(classifier.getDistribution(input))
        else:
            inputsList.append(array([classifier.getPrediction(input)]))
    
    return list(itertools.chain.from_iterable(inputsList))
    
class StackingCombinationRule(CombinationRule):
    
    def __init__(self, l1Classifier, useDistribution):
        self.l1Classifier = l1Classifier
        self.useDistribution = useDistribution
    
    def combine(self, classifiers, input):
        l1Input = _getLevel1Input(classifiers, input, self.useDistribution)
        
        return self.l1Classifier.getDistribution(l1Input)
    
        