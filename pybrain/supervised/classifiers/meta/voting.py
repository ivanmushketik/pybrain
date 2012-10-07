from pybrain.supervised.classifiers.classifier import Classifier, ClassifierFactory
from pybrain.utilities import abstractMethod
from numpy import *

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

class VotingFactory(ClassifierFactory):
    def __init__(self, factories, combinationRule):
        self._factories = factories
        self._combinationRule = combinationRule
    
    def _build(self, dataset):
        classifiers = [factory._build(dataset) for factory in self._factories]
        return _Voting(classifiers, self._combinationRule)

    @property
    def classifierFactories(self):
        return self._factories

    @classifierFactories.setter
    def classifierFactories(self, factories):
        self._factories = factories

    @property
    def combinationRule(self):
        return self._combinationRule

    @combinationRule.setter
    def combinationRule(self, combinationRule):
        self._combinationRule = combinationRule

class CombinationRule:
    """Base class for combiners that produce prediction by combining 
       predictions of individual classifiers in voting.       
        """
    def combine(self, classifiers, value):
        """Receives list of trained classifers """
        abstractMethod()

class MajorVoting(CombinationRule):
    
    def combine(self, classifiers, value):
        numberOfClasses = classifiers[0].distributionLength()
        
        votes ={}
        for classifier in classifiers:
            prediction = classifier.getPrediction(value)
            votes[prediction] = votes.get(prediction, 0) + 1
    
        votesArr = array([0] * numberOfClasses)
        for i in range(numberOfClasses):
            votesArr[i] = votes.get(i, 0)
           
                    
        distribution = [0] * numberOfClasses
        distribution[votesArr.argmax()] = 1
        
        return array(distribution)
    
class DistributionBasedRule(CombinationRule):
    def combine(self, classifiers, value):
        distributionMatrix = array([classifier.getDistribution(value) for classifier in classifiers])
        
        combinedDistribution = self._getCombinedDistribution(distributionMatrix, len(classifiers))
        
        # Because voting can be used as a part of other ensamble method, we should normalize result distribution,
        # because:
        #  * not all combination rules return normalized distribution
        #  * non-normalized distribution can give too small or too big confidence/weight to classifier with non-normalized distribution 
        normalizedDistribution = combinedDistribution / combinedDistribution.sum()
        return combinedDistribution
    
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        abstractMethod()

class SumRule(DistributionBasedRule):
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        nonNormalizedDistribution = distributionMatrix.sum(axis = 0)
        normalizedDistribution = nonNormalizedDistribution / numClassifiers
        
        return normalizedDistribution
    
class WeightedSumRule(DistributionBasedRule):
    def __init__(self, weights):
        self._weights = vstack(weights)
    
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        assert len(self._weights) == numClassifiers        
        
        weightedDistribution = distributionMatrix * self._weights
        distribution = weightedDistribution.sum(axis = 0)
        
        return distribution
    
class MedianRule(DistributionBasedRule):
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        distribution = ma.extras.median(distributionMatrix, axis = 0)
        
        return distribution
    
class MaximumProbabilityRule(DistributionBasedRule):
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        distribution = distributionMatrix.max(axis = 0)
        return distribution
    
class MinimumProbabilityRule(DistributionBasedRule):
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        distribution = distributionMatrix.min(axis = 0)
        return distribution
    
class ProductRule(DistributionBasedRule):
    def _getCombinedDistribution(self, distributionMatrix, numClassifiers):
        distribution = distributionMatrix.prod(axis = 0)
        return distribution

class _Voting(Classifier):
    def __init__(self, classifiers, combinationRule):
        self._classifiers = classifiers
        self._combinationRule = combinationRule
    
    def getDistribution(self, values):
        return self._combinationRule.combine(self._classifiers, values) 
