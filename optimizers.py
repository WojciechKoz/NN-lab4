import numpy as np
from layers import Dense, Conv2D


class SGD:
    def __init__(self, eta=0.05):
        self.eta = eta

    def update(self, layers, batch_size):
        for i in range(1, len(layers)):
            layers[i].weight += (self.eta / batch_size) * layers[i].error.T.dot(layers[i - 1].a.T)
            layers[i].bias += (self.eta / batch_size) * np.array([layers[i].error.sum(axis=0)]).T


class MomentumOptimizer:
    def __init__(self, eta=0.05, gamma=0.8):
        self.previous_grads = []
        self.gamma = gamma
        self.eta = eta

    def update(self, layers, batch_size):
        for i in range(1, len(layers)):
            gradient_w = layers[i].error.T.dot(layers[i - 1].a.T)
            gradient_b = np.array([layers[i].error.sum(axis=0)]).T

            if len(self.previous_grads)+1 == i:
                self.previous_grads.append({"w": np.zeros_like(gradient_w), "b": np.zeros_like(gradient_b)})

            gradient_w += self.gamma * self.previous_grads[i-1]['w']
            gradient_b += self.gamma * self.previous_grads[i-1]['b']

            layers[i].weight += (self.eta / batch_size) * gradient_w
            layers[i].bias += (self.eta / batch_size) * gradient_b


class AdamOptimizer:
    def __init__(self, eta=0.07, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.epsilon = epsilon
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = []
        self.v = []
        self.eta = eta

    def init_params(self, layers):
        self.m = [{} for _ in layers]
        self.v = [{} for _ in layers]

        for i, layer in enumerate(layers):
            if isinstance(layer, Dense):
                self.m[i] = {"w": np.zeros_like(layers[i].weight), "b": np.zeros_like(layers[i].bias)}
                self.v[i] = {"w": np.zeros_like(layers[i].weight), "b": np.zeros_like(layers[i].bias)}
            elif isinstance(layer, Conv2D):
                self.m[i] = {"w": np.zeros_like(layers[i].weight)}
                self.v[i] = {"w": np.zeros_like(layers[i].weight)}


    def update(self, layers, batch_size):
        if not(self.m and self.v):
            self.init_params(layers)

        for i in range(1, len(layers)):
            if isinstance(layers[i], Dense) or isinstance(layers[i], Conv2D):
                if isinstance(layers[i], Dense):
                    gradient_w = layers[i].error.T.dot(layers[i - 1].a.T)
                    gradient_b = np.array([layers[i].error.sum(axis=0)]).T
                else:
                    gradient_w = layers[i].error

                self.m[i]['w'] = self.beta1 * self.m[i]['w'] + (1 - self.beta1) * gradient_w
                self.v[i]['w'] = self.beta2 * self.v[i]['w'] + (1 - self.beta2) * (gradient_w ** 2)
                m_corr_w = self.m[i]['w'] / (1 - self.beta1 ** self.t)
                v_corr_w = self.v[i]['w'] / (1 - self.beta2 ** self.t)
                gradient_w = m_corr_w / (np.sqrt(v_corr_w) + self.epsilon)
                layers[i].weight += (self.eta / batch_size) * gradient_w

                if isinstance(layers[i], Dense):
                    self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * gradient_b
                    self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (gradient_b ** 2)
                    m_corr_b = self.m[i]['b'] / (1 - self.beta1 ** self.t)
                    v_corr_b = self.v[i]['b'] / (1 - self.beta2 ** self.t)
                    gradient_b = m_corr_b / (np.sqrt(v_corr_b) + self.epsilon)
                    layers[i].bias += (self.eta / batch_size) * gradient_b
        self.t += 1


