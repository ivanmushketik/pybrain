from pybrain.utilities import abstractMethod
from pybrain.datasets import SupervisedDataSet

__author__ = 'Ivan Mushketyk, ivan.mushketik@gmail.com'

class ClassfierFactory:
    def buildClassifeir(self, dataset):
        if not isinstance(dataset, SupervisedDataSet):
            raise TypeError("Only SupervisedDataSet can be used to build a classifier")

        return self._build(dataset)

    def _build(self, dataset):
        abstractMethod()

class Classifier:
    def getPrediction(self, values):
        distribution = self.getDistribution(values)

        if len(distribution) == 1:
            return distribution[0]

        maxIndx = distribution.argmax()

        return maxIndx

    def getDistribution(self, values):
        """Returns NumPy array of posterior distributions for each class."""
        abstractMethod()


