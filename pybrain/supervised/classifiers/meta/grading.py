from pybrain.supervised.classifiers.classifier import ClassifierFactory
from pybrain.supervised.classifiers.classifier import Classifier
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.classifiers.meta.voting import MajorVoting

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

CorrectPrediction = 1
IncorrectPrediction = 0

class GradingFactory(ClassifierFactory):
    '''
    Grading method for each base classifier trains grading classifier that predicts if the base classifier will correctly classify
    an instance or will err. During the classification process predicitions of those classifiers that were "graded" as correct are used 
    to produce classification result. If all classifiers are predicted to be wrong, the predictions of all classifiers are combined.
    '''
    
    def __init__(self, baseFactories, gradingFactory, splitingProportion, combinationRule=MajorVoting):
        '''
        baseFactories(iter(ClassifierFactory)) - collection of factories that will be used to train base classifiers
        gradingFactory(ClassifierFactory) - factory that will be used to train grading classifier
        splitingProportion(float) (0, 1) - proportion of instances that will be used to train base classifiers
        combinationRule(CombinationRule) - rule that will be used to combine prediction of base classifiers
        '''
        self.baseFactories = baseFactories
        self.gradingFactory = gradingFactory
        self.splitingProportion = splitingProportion
        self.combinationRule = combinationRule
        
    def _build(self, dataset):
        baseDataset, gradingSet = dataset.splitWithProportion(self.splitingProportion)
        
        baseClassifiers = self._trainBaseClassifiers(baseDataset)
        gradingClassifiers = self._trainGradingClassifiers(baseClassifiers, gradingSet)
        
        return _Grading(baseClassifiers, gradingClassifiers, self.combinationRule())
        
    def _trainBaseClassifiers(self, baseDataset):
        baseClassifiers = [ baseFactory.buildClassifier(baseDataset) for baseFactory in self.baseFactories]
        
        return baseClassifiers
    
    def _trainGradingClassifiers(self, baseClassifiers, gradingSet):   
        # Get number of attributes in the dataset     
        numOfAttirubes = gradingSet.data['input'].shape[1]
        
        gradingClassifiers = []
        for baseClassifier in baseClassifiers:
            gradingDataset = self._createGradingDataset(baseClassifier, gradingSet, numOfAttirubes)
            gradingClassifier = self.gradingFactory.buildClassifier(gradingDataset)
            
            gradingClassifiers.append(gradingClassifier)
            
                 
        return gradingClassifiers

    def _createGradingDataset(self, baseClassifier, gradingSet, numOfAttirubes):
        gradingDataset = ClassificationDataSet(numOfAttirubes, nb_classes=2, class_labels=["Incorrect", "Correct"])
        
        for instance in gradingSet:
            # Get attributes from the instances
            attributes = instance[0]
            # Get class from the instance
            cls = instance[0][0]
            
            prediction = baseClassifier.getPrediction(attributes)
            
            if prediction == cls:
                gradingDataset.appendLinked(attributes, [CorrectPrediction])
            else:
                gradingDataset.appendLinked(attributes, [IncorrectPrediction])
        
        return gradingDataset

class _Grading(Classifier):
    def __init__(self, baseClassifiers, gradingClassifiers, combinationRule):
        assert len(baseClassifiers) == len(gradingClassifiers)
        
        self.baseClassifiers = baseClassifiers
        self.gradingClassifiers = gradingClassifiers
        self.combinationRule = combinationRule
        
    def getDistribution(self, values):
        correctClassifiers = []
        
        for i in range(len(self.baseClassifiers)):
            gradingClassifier = self.gradingClassifiers[i]
            grade = gradingClassifier.getPrediction(values)
            
            if grade == CorrectPrediction:
                correctClassifiers.append(self.baseClassifiers[i])
                
        if not len(correctClassifiers):
            correctClassifiers = self.baseClassifiers
            
        return self.combinationRule.combine(correctClassifiers, values)
    
    