
__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

try:
    import svmutil
    from svmutil import *
except ImportError:
    raise ImportError("Cannot find LIBSVM installation. Make sure svm.py and svmc.* are in the PYTHONPATH!")
    
from numpy import *
import logging
from pybrain.supervised.classifiers.classifier import ClassifierFactory,\
    Classifier

class KernelType:
    LINEAR = svmutil.LINEAR
    POLY = svmutil.POLY
    RBF = svmutil.RBF
    SIGMOID = svmutil.SIGMOID
    PRECOMPUTED = svmutil.PRECOMPUTED

class SVMType:
    C_SVC = svmutil.C_SVC
    NU_SVC = svmutil.NU_SVC
    # TODO: Implement One Class SVM
    #ONE_CLASS = svmutil.ONE_CLASS
    # TODO: Implement SVM for regression
    #EPSILON_SVR = svmutil.EPSILON_SVR
    #NU_SVR = svmutil.NU_SVC

PrintFuncType = CFUNCTYPE(None, c_char_p)


class SVMFactory(ClassifierFactory):
    """
    Factory for SVM classifier. Uses python-libsvm bindings to implement SVM
    """


    def __init__(self, C = 1, kernel = KernelType.POLY, estimateProbability = False, svmType = SVMType.C_SVC, degree = 3, 
                 gamma = 0, coef0 = 0, nu = 0.5, cacheSize = 100, eps = 0.001, p = 0.1, shrinking = True, crossValidation = False,
                 verbose = False, numFolds = 10):
        """
        C(float) - cost parameter
        kernel(one of KernelType) - type of SVM kernel
        estimateProbability(bool) - if True estimate probability of an unknonw instance
        svmType(one of SVMType) - type of SVM classifier
        degree - degree in kernel function 
        gamma - gamma in kernel function
        coef0 - coef0 in kernel function
        nu(float) - nu parameter for NU_SVC, NU_SVR and one class SVM
        cacheSize(float) - cache size in MB
        eps(float) - epsilon in loss function for epsilon-SVR
        shrinking(bool) - if True use shrinking heuristicc
        crossValidation(bool) - perform n-fold cross-validaion 
        verbose(bool) - if True output libsvm debug output on stdout
        """
        
        self.C = C
        self.kernelType = kernel
        self.estimateProbability = estimateProbability
        self.svmType = svmType
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.nu = nu
        self.cacheSize = cacheSize
        self.eps = eps
        self.p = p
        self.shrinking = shrinking
        self.crossValidation = crossValidation
        self.verbose = verbose
        self.numFolds = numFolds
    
    def _build(self, dataset):
        input = dataset.data['input']
        X = input[0:dataset.getLength()]
        
        output = dataset.data['target']
        Y = output[0:dataset.getLength()]
        
        # Converty numpy arrays to python lists 
        y = self._getYList(Y)
        x = self._getXList(X)
        
        # Create libsvm dataset
        prob = svm_problem(y, x)
        
        # Fill parameters for libsvm
        param = self._createSVMParameters()
        # Train libsvm classifier
        model = svm_train(prob, param)
       
        # return classifier
        return self._createClassifier(model, dataset, y)
   
    def _getYList(self, Y):
        result = []
        for i in range(Y.shape[0]):
            result.append(Y[i][0])
            
        return result
    
    def _getXList(self, X):
        result = []
        for r in range(X.shape[0]):
            featureVector = []
            for c in range(X.shape[1]):
                featureVector.append(X[r][c])
            result.append(featureVector)
            
        return result
        
    def _createSVMParameters(self):
        params = svm_parameter()
        params.C = self.C
        params.kernel_type = self.kernelType
                
        params.svm_type = self.svmType;
        params.degree = self.degree
        params.gamma = self.gamma
        params.coef0 = self.coef0
        params.nu = self.nu
        params.cache_size = self.cacheSize
        params.eps = self.eps
        params.p = self.p
        
        params.shrinking = self._getIntValue(self.shrinking)
        params.probability = self._getIntValue(self.estimateProbability)
        params.cross_validation = self.crossValidation
        params.nr_fold = self.numFolds
        
        #params.nr_weight = 0
        
        # Funciton to print libsvm output
        def printFunc(msg):
            logging.debug("[LIBSVM] %s", msg)
            if self.verbose:
                print msg
        
        params.print_func = PrintFuncType(printFunc)
        
        return params
    
    # Convert False/True to 0/1 values
    def _getIntValue(self, boolean):
        if boolean:
            return 1
        else:
            return 0
    
    def _createClassifier(self, model, dataset, y):
        
        if self.estimateProbability:
            return _SVMWithProbability(model, self._getDistributionIndex(dataset, y))
        else:
            return _SVMWithoutProbability(model, dataset.nClasses)
      
    # When libsvm returns probability distribution it returns not
    # in the numeric order {0, 1, 2, ...} but in the order
    # in which classes apear in the dataset. This function
    # creates index of which class was in which order in the original
    # dataset to sort result probability distribution
    def _getDistributionIndex(self, dataset, y):
        foundClasses = set()
        numFoundClasses = 0
        
        index = []
        
        for klass in y:
            if klass not in foundClasses:
                index.append(klass)
                foundClasses.add(klass)
                if len(foundClasses) == dataset.nClasses:
                    break
                
        return index 
        
        
class _SVMWithProbability(Classifier):
    def __init__(self, model, index):
        self.model = model
        self.index = index
        
    def getDistribution(self, values):
        svmResult =  svm_predict([0], [values], self.model, options = '-b 1')
        shuffledProbabilities = svmResult[2][0]
        
        probabilityDistribution = [0] * len(self.index)
        # Sorting probability distribution
        for i, prob in enumerate(shuffledProbabilities):
            probabilityDistribution[self.index[i]] = prob

        return array(probabilityDistribution)
    
        
class _SVMWithoutProbability(Classifier):
    def __init__(self, model, numClasses):
        self.model = model
        self.numClasses = numClasses
        
    def getDistribution(self, values):        
        svmResult =  svm_predict([0], [values], self.model)
        predictedClass =  svmResult[0][0]
        
        distribution = [0] * self.numClasses
        distribution[int(predictedClass)] = 1
        return array(distribution)
        
    
class _OneClassSVM(Classifier):
    def __init__(self, model):
        self.model = model
        
    def getDistribution(self, values):   
        pass