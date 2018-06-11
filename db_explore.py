#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import tensorflow as tf


def load_mnist(path='./mnist_db/'):
    x_train = np.load(path+'x_train.npy')
    y_train = np.load(path+'y_train.npy')
    x_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')
    return (x_train, y_train), (x_test, y_test)


def main():
    train, test = load_mnist()
    x_train, y_train = train
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    print(ds)


if __name__ == '__main__':
    main()
