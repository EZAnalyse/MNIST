#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow import layers
import db_explore


def cnn_model(features, mode):
    """
    cnn model structure
    :param features: images
    :return: predicts
    """
    input_layer = tf.reshape(features, shape=[-1, 28, 28, 1])
    # conv1
    conv1 = layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                          name='conv1')
    pool1 = layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    # conv2
    conv2 = layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                          name='conv2')
    pool2 = layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    # fully connected layer
    pool2_flat = layers.flatten(pool2, name='flatten')
    dense = layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu, name='dense_layer')
    dropout = layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # output layer
    logits = tf.layers.dense(inputs=dropout, units=10, name='logits')
    return logits


def Loss(labels, logits, mode):
    """
    loss of the model in training and evaluating
    :param labels:
    :param logits: predictions of cnn model
    :param mode: Train, Evaluate, Predict
    :return: loss
    """
    if mode != tf.estimator.ModeKeys.PREDICT:
        return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


def Train_op(loss, mode):
    """
    set the optimizer of the train step
    :param loss:
    :param mode: Train, Evaluate, Predict
    :return: trainer
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return train_op


# build mode
def cnn_model_fn(features, labels, mode):
    """
    build the hole cnn model for tf.estimator
    :param features: image arrays of shape (batch_size, 28, 28, 1)
    :param labels: one hot label of shape (batch_size, 10)
    :param mode: passed by tf.estimator
    :return: tf.estimator.EstimatorSpec
    """

    logits = cnn_model(features=features, mode=mode)

    # predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    loss = Loss(labels=labels, logits=logits, mode=mode)

    train_op = Train_op(loss=loss, mode=mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["classes"], name='acc')
        eval_metric_ops = {'accuracy': accuracy, }
    else:
        eval_metric_ops = None

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def main():
    # set logging verbosity high enough
    tf.logging.set_verbosity(tf.logging.INFO)

    # load mnist data
    train, test = db_explore.load_mnist()
    x_train, y_train = train

    run_config = tf.estimator.RunConfig(model_dir='./model/cnn', keep_checkpoint_max=2, save_checkpoints_steps=100)
    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, config=run_config)

    train_input_fn = lambda: db_explore.train_input_fn(x_train, y_train, 32)
    x_test, y_test = test
    eval_input_fn = lambda: db_explore.eval_input_fn(x_test, y_test, 32)

    # for _ in range(10):
    #     estimator.train(input_fn=train_input_fn, steps=2000, )
    #
    #     eval = estimator.evaluate(input_fn=eval_input_fn, steps=2000)
    #     print(eval)

    eval = estimator.evaluate(input_fn=eval_input_fn)
    print(eval)


if __name__ == '__main__':
    main()
