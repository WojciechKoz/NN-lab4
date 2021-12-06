import numpy as np
from sklearn.utils import shuffle
from utils import RANDOM_SEED
from optimizers import SGD, MomentumOptimizer

np.random.seed(RANDOM_SEED)


class MLP:
    def __init__(self, layers, optimizer=SGD(eta=0.05), batch_size=100,
                 verbose=True, target_accuracy=1.):
        self.layers = layers

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.target_accuracy = target_accuracy

    def forward(self, X):
        a = X.T
        for layer in self.layers:
            a = layer.forward(a)
        return a.T

    def backward(self, y_pred, y):
        self.layers[-1].error = y - y_pred
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].backward(self.layers[i + 1])

    def fit(self, X, y, valid_X, valid_y, epochs=25):
        losses = []
        scores = []
        for epoch in range(epochs):
            train_X, train_y = shuffle(X, y)
            X_batches = np.array_split(train_X, int(len(train_X) / self.batch_size))
            y_batches = np.array_split(train_y, int(len(train_X) / self.batch_size))
            for Xi, yi in zip(X_batches, y_batches):
                y_pred = self.forward(Xi)
                self.backward(y_pred, yi)
                self.optimizer.update(self.layers, self.batch_size)

            y_valid_pred = self.forward(valid_X)
            valid_accuracy = self.accuracy(valid_y, self.to_hot_ones(y_valid_pred))
            loss = self.loss(valid_y, y_valid_pred).mean()

            scores.append(valid_accuracy)
            losses.append(loss)

            if valid_accuracy > self.target_accuracy:
                if self.verbose: print(f'Got accuracy = {self.target_accuracy}')
                break

            if self.verbose:
                train_accuracy = self.accuracy(train_y, self.predict(train_X))
                self.log(epoch + 1, train_accuracy, valid_accuracy, loss)

        return scores, losses

    def predict(self, X):
        y_pred = self.forward(X)
        return self.to_hot_ones(y_pred)

    def to_hot_ones(self, y_pred):
        from utils import hot_ones
        return hot_ones(np.argmax(y_pred, axis=1), 10)

    def loss(self, y, y_pred):
        return -(np.log(y_pred) * y).sum(axis=1)

    @staticmethod
    def accuracy(y, y_pred):
        return np.where(y == y_pred, y, 0).sum() / len(y)

    def log(self, epoch, train_acc, test_acc, loss):
        print(f"Epoch: {epoch:3} | "
              f"train acc: {train_acc:.4f} | "
              f"valid acc: {test_acc:.4f} | "
              f"loss: {loss:.4f} | ")
