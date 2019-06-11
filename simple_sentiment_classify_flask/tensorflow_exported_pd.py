#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@File    :  ${saved_filename}.py
#@Date    :  2019-05-11 17:34:25
#@Author  :  H.wei
#@Version :  1.0
#@License :  Copyright (C) 2019 YPLSEC, Inc.
#
#@Desc    :  None
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

#! /usr/bin/env python
"""
训练并导出Softmax回归模型，使用SaveModel导出训练模型并添加签名。
"""

from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import basic.mnist_input_data as mnist_input_data

# 定义模型参数
tf.app.flags.DEFINE_integer('training_iteration', 10, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 2, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', './tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):

    # 参数校验
    # if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    #     print('Usage: mnist_saved_model.py [--training_iteration=x] '
    #           '[--model_version=y] export_dir')
    #     sys.exit(-1)
    # if FLAGS.training_iteration <= 0:
    #     print('Please specify a positive value for training iteration.')
    #     sys.exit(-1)
    # if FLAGS.model_version <= 0:
    #     print('Please specify a positive value for version number.')
    #     sys.exit(-1)

    # Train model
    print('Training model...')

    mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)

    sess = tf.InteractiveSession()

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('training accuracy %g' % sess.run(
        accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        }))
    print('Done training!')

    # Export model
    # WARNING(break-tutorial-inline-code): The following code snippet is
    # in-lined in tutorials, please update tutorial documents accordingly
    # whenever code changes.

    # export_path_base = sys.argv[-1]
    export_path_base = "/Users/xingoo/PycharmProjects/ml-in-action/实践-tensorflow/01-官方文档-学习和使用ML/save_model"
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    # 配置导出地址，创建SaveModel
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.

    # 创建TensorInfo，包含type,shape,name
    classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    # 分类签名：算法类型+输入+输出（概率和名字）
    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    # 预测签名：输入的x和输出的y
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # 构建图和变量的信息：
    """
    sess                会话
    tags                标签，默认提供serving、train、eval、gpu、tpu
    signature_def_map   签名
    main_op             初始化？
    strip_default_attrs strip?
    """
    # predict_images就是服务调用的方法
    # serving_default是没有输入签名时，使用的方法
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    # 保存
    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()