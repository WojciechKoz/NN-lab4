from initializers import HeInitializer, StandardInitializer
from activations import relu, relu_prime, softmax, sigmoid_prime, sigmoid, tanh_prime, tanh
import numpy as np
from numpy.lib.stride_tricks import as_strided


class Dense:
    def __init__(self, neurons, activation='relu', weights_initializer=HeInitializer()):
        self.neurons = neurons
        self.weights_initializer = weights_initializer
        self.weight = None
        self.bias = None

        if activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_prime = None

    def forward(self, a):
        if self.weight is None:
            self.weight, self.bias = self.weights_initializer.init_weights(a.shape[0], self.neurons)

        self.z = self.weight.dot(a) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def backward(self, next_layer):
        self.error = next_layer.weight.T.dot(next_layer.error.T).T * self.activation_prime(self.z).T


class Input:
    def forward(self, a):
        self.a = a
        return a

    def backward(self, next_layer):
        pass


class Conv2D:
    def __init__(self, filters=12, kernel_shape=(5, 5), weights_initializer=StandardInitializer()):
        self.filters = filters
        self.kernel_shape = kernel_shape
        self.weight = weights_initializer.init_weights((filters, 1, *kernel_shape))
        # self.biases = weights_initializer.init_weights((filters, 1))

    def iterate_regions(self, a):
        _, _, h, w = a.shape

        for i in range(h - (self.kernel_shape[1] - 1)):
            for j in range(w - (self.kernel_shape[0] - 1)):
                im_region = a[0, :, i:(i + self.kernel_shape[0]), j:(j + self.kernel_shape[1])]
                yield im_region, i, j

    def forward(self, a):
        a = a.transpose(0, 3, 1, 2)
        self.a = a
        _, batch_size, h, w = a.shape
        self.batch_size = batch_size

        output = np.zeros((h - (self.kernel_shape[1] - 1), w - (self.kernel_shape[0] - 1), self.filters, batch_size))

        for im_regions, i, j in self.iterate_regions(a):
            output[i, j] = np.sum(im_regions * self.weight, axis=(2, 3))

        return output  #  + self.biases

    def backward(self, next_layer):
        error = np.zeros((self.filters, self.batch_size, *self.kernel_shape))

        for im_region, i, j in self.iterate_regions(self.a):
            for f in range(self.filters):
                error[f] += next_layer.error[i, j, f].reshape(self.batch_size, 1, 1) * im_region

        self.error = error.sum(axis=1, keepdims=True)


class MaxPooling2D:
    def __init__(self, pool_shape=(2, 2)):
        self.pool_shape = pool_shape

    def iterate_regions(self, image):
        h, w, _, _ = image.shape

        new_h = h // self.pool_shape[1]
        new_w = w // self.pool_shape[0]

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.pool_shape[1]):((i + 1) * self.pool_shape[1]),
                            (j * self.pool_shape[0]):((j + 1) * self.pool_shape[0])]
                yield im_region, i, j

    def forward(self, a):
        self.a = a
        h, w, num_filters, batch_size = a.shape
        output = np.zeros((h // self.pool_shape[1], w // self.pool_shape[0], num_filters, batch_size))

        for im_region, i, j in self.iterate_regions(a):
            output[i, j] = np.max(im_region, axis=(0, 1))

        return output

    def backward(self, last_layer):
        self.error = np.zeros_like(self.a)
        for im_region, i, j in self.iterate_regions(self.a):
            h, w, f, b = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for bi in range(b):
                for hi in range(h):
                    for wi in range(w):
                        for fi in range(f):
                            if im_region[hi, wi, fi, bi] == amax[fi, bi]:
                                self.error[i * self.pool_shape[0] + hi, j * self.pool_shape[1] + wi, fi, bi] = \
                                    last_layer.error[i, j, fi, bi]
                                break

class Flatten:
    def forward(self, a):
        self.shape = a.shape
        self.a = a.reshape(-1, a.shape[-1])
        return self.a

    def backward(self, last_layer):
        error = last_layer.weight.T.dot(last_layer.error.T).T
        self.error = error.reshape(self.shape)
