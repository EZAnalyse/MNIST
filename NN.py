#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf


# build mode
def nn_model_fn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features, shape=[-1, 28 * 28])
    # nn layer
    hidden_layer = tf.layers.dense(inputs=input_layer, units=32, activation=tf.nn.sigmoid)
    # output_layer
    output_layer = tf.layers.dense(inputs=hidden_layer, units=10)

    predictions = {"classes": tf.argmax(input=output_layer, axis=1),
                   "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # set logging verbosity high enough
    tf.logging.set_verbosity(tf.logging.INFO)

    import db_explore
    train, test = db_explore.load_mnist()
    x_train, y_train = train
    batch_size = 32
    # build config of ckpt files
    save_config = tf.estimator.RunConfig(model_dir='./model/nn', keep_checkpoint_max=3)

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=nn_model_fn, config=save_config)

    # # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    #
    # estimator.train(input_fn=lambda: db_explore.train_input_fn(x_train, y_train, batch_size), steps=100,
    #                 hooks=[logging_hook, ])

    # no logging hooks
    estimator.train(input_fn=lambda: db_explore.train_input_fn(x_train, y_train, batch_size), steps=200, )

    x_test, y_test = test
    ev = estimator.evaluate(input_fn=lambda: db_explore.eval_input_fn(x_test, y_test, batch_size))
    print(ev)


if __name__ == '__main__':
    main()
