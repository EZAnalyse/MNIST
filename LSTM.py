#!/usr/bin/env python
# _*_ coding:utf-8 _*_


import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib import rnn
import db_explore


def lstm_model(features, mode):
    """
    cnn model structure
    :param features: images
    :return: predicts
    """
    input_layer = tf.unstack(value=features, num=28, axis=1, name='input')
    lstm_cell = rnn.BasicLSTMCell(num_units=128, name='lstm')
    lstm_out, _ = rnn.static_rnn(lstm_cell, input_layer, dtype=tf.float32, )
    flatten_layer = layers.flatten(lstm_out[-1], name='flatten')
    dense_layer = layers.dense(inputs=flatten_layer, units=512, name='dense')
    dropout = layers.dropout(inputs=dense_layer, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN),
                             name='dropout')
    logits = layers.dense(inputs=dropout, units=10, name='logits')
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return train_op


# build mode
def lstm_model_fn(features, labels, mode):
    """
    build the hole cnn model for tf.estimator
    :param features: image arrays of shape (batch_size, 28, 28, 1)
    :param labels: one hot label of shape (batch_size, 10)
    :param mode: passed by tf.estimator
    :return: tf.estimator.EstimatorSpec
    """
    logits = lstm_model(features=features, mode=mode)

    # predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1, name='y_pred'),
                   "probabilities": tf.nn.softmax(logits, name="y_prob")}

    loss = Loss(labels=labels, logits=logits, mode=mode)

    train_op = Train_op(loss=loss, mode=mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["classes"],
                                       name='accuracy')
        eval_metric_ops = {'accuracy': accuracy, }
    else:
        eval_metric_ops = None

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def main(argv):
    # set logging verbosity high enough
    tf.logging.set_verbosity(tf.logging.INFO)

    # build estimator
    run_config = tf.estimator.RunConfig(model_dir='./model/lstm', keep_checkpoint_max=2, save_checkpoints_steps=500,
                                        tf_random_seed=2018)
    estimator = tf.estimator.Estimator(model_fn=lstm_model_fn, config=run_config)

    # load mnist data
    train, test = db_explore.load_mnist()

    x_train, y_train = train
    train_input_fn = lambda: db_explore.train_input_fn(x_train, y_train, 32)
    x_test, y_test = test
    eval_input_fn = lambda: db_explore.eval_input_fn(x_test, y_test, 32)

    for _ in range(10):
        estimator.train(input_fn=train_input_fn, steps=2000, )
        eval = estimator.evaluate(input_fn=eval_input_fn, steps=500)
        print(eval)
    eval = estimator.evaluate(input_fn=eval_input_fn, )
    print(eval)


if __name__ == '__main__':
    tf.app.run()
