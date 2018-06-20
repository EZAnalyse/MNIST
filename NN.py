#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf


# build mode
def nn_model_fn(features, labels, mode):
    # input layer
    input_layer = features
    # nn layer
    hidden_layer = tf.layers.dense(inputs=input_layer, units=32, activation=tf.nn.relu)
    # output_layer
    output_layer = tf.layers.dense(inputs=hidden_layer, units=10)
    predictions = {"classes": tf.argmax(input=output_layer, axis=1), "probabilities": output_layer}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(onehot_labels=labels, logits=output_layer)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=tf.arg_max(labels), predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    pass


if __name__ == '__main__':
    main()
