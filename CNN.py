#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow import layers


# build mode
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], shape=[-1, 28, 28, 1])
    conv1 = layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = layers.flatten(pool2)
    dense = layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    pass


if __name__ == '__main__':
    main()
