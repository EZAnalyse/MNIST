#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from db_explore import *


# nn model structure
def nn_model(features):
    # input layer
    input_layer = tf.layers.flatten(inputs=features, name='input')
    # nn layer
    hidden_layer = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.sigmoid, name='hidden')
    # output_layer
    logits = tf.layers.dense(inputs=hidden_layer, units=10, name='logits')
    return logits


# predict
def Predict(logits, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"classes": tf.argmax(input=logits, axis=1), }
        return predictions


# loss of the model
def Loss(labels, logits, mode):
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        return loss


# train op of the model
def Train(loss, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return train_op


# evaluate matric
def Eval_matric(labels, logits, mode):
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
        eval_metric_ops = {'accuracy': accuracy, }
        return eval_metric_ops


# build mode for estimator
def nn_model_fn(features, labels, mode):
    logits = nn_model(features)

    predictions = Predict(logits, mode)
    loss = Loss(labels, logits, mode)
    train_op = Train(loss, mode)
    eval_metric_ops = Eval_matric(labels, logits, mode)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def main(argv):
    # set logging verbosity high enough
    tf.logging.set_verbosity(tf.logging.INFO)

    # build config of ckpt files
    save_config = tf.estimator.RunConfig(model_dir='./model/nn', keep_checkpoint_max=3, save_checkpoints_steps= 500)

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=nn_model_fn, config=save_config)

    # build dataset
    train, test = load_mnist()
    batch_size = 32

    x_train, y_train = train
    train_input = lambda: train_input_fn(x_train, y_train, batch_size)
    x_test, y_test = test
    test_input_fn = lambda: eval_input_fn(x_test, y_test, batch_size)

    # # Set up logging for predictions
    # tensors_to_log = {"probabilities": tensor's name}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    #
    # estimator.train(input_fn=train_input, steps=100, hooks=[logging_hook, ])

    # no logging hooks
    for _ in range(10):
        estimator.train(input_fn=train_input, steps=2000, )
        ev = estimator.evaluate(input_fn=test_input_fn, steps=100)
        print(ev)


if __name__ == '__main__':
    tf.app.run()
