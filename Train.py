"""
Function that can train a neural net
"""
from LearningLibrary.Tensor import Tensor
from LearningLibrary.neuralnet import NeuralNet
from LearningLibrary.Loss import Loss,MSE
from LearningLibrary.optim import Optimizer, SGD
from LearningLibrary.data import DataIterator,BatchIterator

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD())-> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch,epoch_loss)