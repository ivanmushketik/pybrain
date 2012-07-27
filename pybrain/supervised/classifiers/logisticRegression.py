
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

from numpy import *
from pybrain.supervised.classifiers.classifier import ClassifierFactory,\
    Classifier
from pybrain.tools.functions import sigmoid

class LogisticRegressionFactory(ClassifierFactory):
    """
    Logistic regression is a classifier that is using gradient descent to find the best parameters for classification boundary.
    """
    def __init__(self, optimizer):
        '''
        optimizer(GradientOptimizer) - specifies an optimizer that will be used to find global optimum of logistic regression parameters
        '''
        self.optimizer = optimizer
        
    def _build(self, dataset):
        if dataset.nClasses == 2:
            # Binary classification problem
            return self._createLogisticRegression(dataset)
        else:
            raise ValueError("Logistic regression can handle only binary classification task. Please use multi class classification instead")
        
    def _createLogisticRegression(self, dataset):
        X, Y = self._getMatricies(dataset)
        thetas = self._generateInitialThetas(X)
        thetas = self._getOptimalThetas(X, Y, thetas)
        
        return _LogisticRegression(thetas)
      
    def _generateInitialThetas(self, X):
        return zeros(X.shape[1])
                
    def _getOptimalThetas(self, X, Y, initThetas):
        # TODO: this implementation is VERY simple and VERY-VERY inefficient. Convert to vectorized form.
        def costFunction(thetas):
            totalCost = 0
            for i in range(Y.shape[0]):
                y = Y[i][0]                    
                x = X[i]
                
                mul = thetas * x
                posProb = sigmoid(mul.sum())
                
                assert y == 1 or y == 0
                
                if y == 1:
                    cost = -log(posProb)
                else:
                    cost = -log(1 - posProb)
                    
                totalCost += cost
                
            regLambda = 0.0001
            regularizationSum = regLambda * (thetas ** 2).sum() / (2 * Y.shape[0])
            
            return totalCost / Y.shape[0] + regularizationSum
        
        def calculateDerivatives(thetas):
            derivatives = []
            for j in range(len(thetas)):
                sum = 0
                
                for i in range(Y.shape[0]):
                    y = Y[i][0]                    
                    x = X[i]
                    
                    mul = thetas * x
                    posProb = sigmoid(mul.sum())
                    
                    sum += (posProb - y) * X[i][j]
                    
                jDerivative = sum / Y.shape[0]
                derivatives.append(jDerivative)
                
            return derivatives
        
        initParams = initThetas
        self.optimizer.setEvaluator(costFunction, initParams)
        self.optimizer.gradientsCalculator = calculateDerivatives
        optimalThetas, minError = self.optimizer.learn()
        
        return optimalThetas
        
    def _getMatricies(self, dataset):
        input = dataset.data['input']
        input = input[0:dataset.getLength()]
        biasUnit = ones((dataset.getLength(), 1))
        X = hstack( (biasUnit, input) )
        
        output = dataset.data['target']
        Y = output[0:dataset.getLength()]
        
        return X, Y
    
class _LogisticRegression(Classifier):
    """
    This class is created by LogisticRegressionFactory for binary classification problems
    """
    def __init__(self, thetas):
        """
        thetas(list, array) - parameters for decision boundary for classification
        """
        self.thetas = array(thetas)
        
    def getDistribution(self, values):
        values = self._appendBiasTerm(values)
        mul = self.thetas * values
        posProb = sigmoid(mul.sum())
        negProb = 1 - posProb
        
        return array([negProb, posProb])
    
    def _appendBiasTerm(self, values):
        return hstack( ([1], values) )
    
    def distributionLength(self):
        # Logistic regression 
        return 2
