#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# load the mnist database from the *.npy files
def load_mnist(path='./mnist_db/'):
    # numpy.load for .npy file
    # return numpy.array, (train_db, test_db)
    # x: image
    # y: label
    x_train = np.load(path + 'x_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'x_test.npy')
    y_test = np.load(path + 'y_test.npy')
    return (x_train, y_train), (x_test, y_test)


# show image by random
def __rnd_show(images, labels):
    # get a random index
    # plt.show for img in np.array type
    index = np.random.randint(len(images))
    img = images[index]
    label = labels[index]
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()


def __input_fn(features, labels, batch_size, count):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1024).repeat(count=count).batch(batch_size)
    dataset = dataset.map(__label_one_hot)  # convert labels to one hot array
    return dataset  # .make_one_shot_iterator().get_next()


# construct input dataset for estimator training
def train_input_fn(features, labels, batch_size):
    count = None
    return __input_fn(features, labels, batch_size, count)


# construct input dataset for estimator evaluating
def eval_input_fn(features, labels, batch_size):
    count = 1
    return __input_fn(features, labels, batch_size, count)


def predict_input_fn(features, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(buffer_size=1024).repeat(count=1).batch(batch_size)
    return dataset  # .make_one_shot_iterator().get_next()


# convert labels to one_hot
def __label_one_hot(features, labels):
    features = tf.cast(features, dtype=tf.float32)
    labels = tf.one_hot(labels, 10, dtype=tf.int32)
    return features, labels


def main():
    train, test = load_mnist()
    x_train, y_train = train
    a = train_input_fn(x_train, y_train, 3)
    with tf.Session() as sess:
        b = sess.run(a)


if __name__ == '__main__':
    main()
