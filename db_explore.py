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
    x_train = np.load(path+'x_train.npy')
    y_train = np.load(path+'y_train.npy')
    x_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')
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


# construct input dataset for estimator training
def train_input_fn(features, labels, batch_size):
    # features: all features in numpy array type
    # labels: all labels in numpy array type
    # batch_size: size of training dataset in each training step
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1024).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def main():
    train, test = load_mnist()
    x_train, y_train = train
    a = train_input_fn(x_train, y_train, 2)
    with tf.Session() as sess:
        b = sess.run(a)
    print(b)


if __name__ == '__main__':
    main()
