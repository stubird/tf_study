"""
   Copyright 2018 (c) Jinxin Xie

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from mnist_data import *
import tensorflow.contrib.slim as slim
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
IMAGE_SIZE = 28

FLAGS = None

def my_input_fn(train_data, train_labels, batch_size = 1,repeat_count=None, shuffle_count=1):

    print("log",len(train_data))
    print("log2", len(train_labels))
    #data = data.reshape(num_images, 1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

    dataset = (tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        #.skip(1)  # Skip header row
        #.map(decode_csv, num_parallel_calls=4)  # Decode each line
        #.cache() # Warning: Caches entire dataset, can cause out of memory
        .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
        .repeat(repeat_count)    # Repeats dataset this # times
        .batch(batch_size)
        #.prefetch(1)  # Make sure you always have 1 batch ready to serve
    )

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    print(batch_features)
    print(batch_labels)
    return batch_features, batch_labels

def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,
    ):    # And instance of tf.estimator.ModeKeys, see below

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    features_reshape = tf.reshape(features,[-1,28,28,1])
    print(features)
    print(features.get_shape().ndims)

    input_layer = features_reshape

    with slim.arg_scope([slim.conv2d], padding='SAME',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(0.0005)):

        net = slim.conv2d(input_layer, 32, 5, scope='conv1_1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.fully_connected(net, 512, scope='fc1')
        net = slim.dropout(net, 0.5, scope='dropout1')
        #net = slim.flatten(net,scope='flt1')
        net = tf.layers.Flatten()(net)
        print('flat:',net.get_shape())
        net = slim.fully_connected(net, 10, scope='fc2')#activation_fn=None,
        #net = tf.argmax(net,axis=1)
        print(net)
        # Define the loss functions and get the total loss.
        # classification_loss = tf.losses.softmax_cross_entropy(net, labels)

        predictions = {'class_ids': tf.argmax(input=net, axis=1)}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=net)
        accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=losses,
                eval_metric_ops={'my_accuracy': accuracy})

        total_loss1 = losses# + sum_of_squares_loss + pose_loss + regularization_loss
        learning_rate = 0.001
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.9).minimize(losses,
                                                             global_step=tf.train.get_global_step())

        #predictions = {'class_ids': tf.argmax(input=net, axis=1)}
        #accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])
    return tf.estimator.EstimatorSpec(
            net,
            loss=losses,
            train_op = optimizer
        )#,eval_metric_ops={'my_accuracy': accuracy})

def main(_):
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
           model_dir='./model')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    print(FLAGS.self_test,FLAGS.train,FLAGS.predict)
    if FLAGS.self_test:
        # print('Running self-test.')
        # train_data, train_labels = fake_data(256)
        # validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        # test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        # num_epochs = 1
        pass
    elif FLAGS.predict:
        _, _,test_data, test_labels = get_data(32,80)
        predict_results = classifier.predict(input_fn=lambda: my_input_fn(test_data, test_labels, repeat_count = 1))
        print(predict_results)

        for i,predict_result in enumerate(predict_results):
            tf.logging.info("...{} true:{}".format(predict_result["class_ids"],test_labels[i]))
    elif FLAGS.train:

        train_data, train_labels, test_data, test_labels = get_data(20000, 32)
        classifier.train(input_fn=lambda: my_input_fn(train_data, train_labels),steps=10000)
        # my_feature_columns = []
        #
        # for key in train_data.keys():
        #     my_feature_columns.append(tf.feature_column.numeric_column(key=key))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train',
      default=False,
      help='True if running a train.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')
  parser.add_argument(
      '--predict',
      default=False,
      action='store_true',
      help='True if running a predict.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)