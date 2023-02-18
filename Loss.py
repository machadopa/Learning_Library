
""" Measures how good our predictions are.
    can be used to adjust parameters of our network
    """
import numpy as np
from LearningLibrary.Tensor import Tensor
class Loss:
    def loss(self, predicted: Tensor, actual: Tensor)-> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor)-> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean squared error
    """
    def loss(self, predicted: Tensor, actual : Tensor)-> Tensor:
        return np.sum((predicted - actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor)-> Tensor:
        return 2 * (predicted - actual)