from utils import get_data
from mlp import MLP
from optimizers import SGD, MomentumOptimizer, AdamOptimizer
from layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

train_X, valid_X, train_y, valid_y, test_X, test_y = get_data()


def my_conv_test(filters=8, filter_size=5):
    layers = [
                  Input(),
                  Conv2D(filters, kernel_shape=(filter_size, filter_size)),
                  MaxPooling2D(pool_shape=(3,3)),
                  Flatten(),
                  Dense(100),
                  Dense(40),
                  Dense(10, activation='softmax')
                     ]
    mlp = MLP(layers, optimizer=AdamOptimizer())
    mlp.fit(train_X[:4000], train_y[:4000], valid_X[:500], valid_y[:500], epochs=4)
    print(MLP.accuracy(test_y, mlp.predict(test_X)))


def keras_conv_test(filters=8, filter_size=5, pooling=True, pooling_size=4):
    print('start keras')
    model = models.Sequential()
    model.add(layers.Conv2D(filters, (filter_size, filter_size), activation='relu', input_shape=(28, 28, 1)))
    if pooling:
        model.add(layers.MaxPooling2D((pooling_size, pooling_size)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_X, train_y, epochs=4, validation_data=(valid_X, valid_y), verbose=False)
    score, acc = model.evaluate(test_X, test_y, verbose=0)
    return acc

def my_dense_test():
    layers = [
        Input(),
        Flatten(),
        Dense(100),
        Dense(40),
        Dense(10, activation='softmax')
    ]
    mlp = MLP(layers, optimizer=AdamOptimizer())
    mlp.fit(train_X, train_y, valid_X, valid_y, epochs=10)
    print(MLP.accuracy(test_y, mlp.predict(test_X)))


accs = [my_conv_test() for _ in range(10)]
print('z:', np.array(accs).mean())

accs = [keras_conv_test(pooling=False) for _ in range(10)]
print('bez:', np.array(accs).mean())


