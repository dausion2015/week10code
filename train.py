#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import pydensecrf.densecrf as dcrf
import vgg
from dataset import inputs
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian, unary_from_softmax)
from utils import (bilinear_upsample_weights, grayscale_to_voc_impl)

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset_train', type=str)
    parser.add_argument('--dataset_val', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()

slim = tf.contrib.slim


tf.reset_default_graph()
is_training_placeholder = tf.placeholder(tf.bool)
batch_size = FLAGS.batch_size

image_tensor_train, orig_img_tensor_train, annotation_tensor_train = inputs(FLAGS.dataset_train, train=True, batch_size=batch_size, num_epochs=1e4)
image_tensor_val, orig_img_tensor_val, annotation_tensor_val = inputs(FLAGS.dataset_val, train=False, num_epochs=1e4)

image_tensor, orig_img_tensor, annotation_tensor = tf.cond(is_training_placeholder,
                                                           true_fn=lambda: (image_tensor_train, orig_img_tensor_train, annotation_tensor_train),
                                                           false_fn=lambda: (image_tensor_val, orig_img_tensor_val, annotation_tensor_val))

feed_dict_to_use = {is_training_placeholder: True}

upsample_factor = 16
number_of_classes = 21

log_folder = os.path.join(FLAGS.output_dir, 'train')

vgg_checkpoint_path = FLAGS.checkpoint_path

# Creates a variable to hold the global_step.
global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)


# Define the model that we want to use -- specify to use only two classes at the last layer
'''
with slim.arg_scope(vgg.vgg_arg_scope()):
    the code in vgg.py
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc
    '''
with slim.arg_scope(vgg.vgg_arg_scope()):
    # in this case ,output shape is not 1*1*channel,so,spatial_squeeze=False,fc_conv_padding='SAME'
    logits, end_points = vgg.vgg_16(image_tensor,
                                    num_classes=number_of_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')

downsampled_logits_shape = tf.shape(logits)

img_shape = tf.shape(image_tensor)

# Calculate the ouput size of the upsampled tensor
# The shape should be batch_size X width X height X num_classes
upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  img_shape[1],
                                  img_shape[2],
                                  downsampled_logits_shape[3]
                                  ])
'''
主要分两步：
1、由于fc_conv_padding='SAME'，最后的输出的logist的shape和pool5的shape是一样的，对logist进行2s上采样反卷积得到结果B1，此时的结果和pool4输出的sapce是一样的，
同时直接对pool4输出进行卷积分类输出B2，得到的B1和B2shape一样，对B1和B2进行对应元素相加得到结果C，此时直接对C进行16倍上采样得到的结果的space和input一样，是16supsampling
2、对pool3的输出直接进行logist分类，得到C1。同时对C进行2s upsampling,得到结果C2,C1和C2的shape一样，C2和pool4的输出的space一样，此时对C1和C2进行对应元素相加得到结果D，
对D进行8S upsampling得到和输入space一样的结果 ，最后把输入onehot 两者进行交叉熵对比，这就是8s上采样
'''

pool4_feature = end_points['vgg_16/pool4']
with tf.variable_scope('vgg_16/fc8'):
    #classification from pool4 drectly here
    aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool4')

# Perform the upsampling 
upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                  number_of_classes)
#2 times upsampling kernel tensor
upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/logist_t_conv_x2')
#transpose logits with 2s upsampling kernel tensor
upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2,
                                          output_shape=tf.shape(aux_logits_16s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')

#the result's space get 2 times
#add element with  upsampled_logits and aux_logits_16s

upsampled_logits = upsampled_logits + aux_logits_16s
#the result just upsampling 16s .then,having the same space with original image
#apsample arrary
#upsample_filter_np_x16 = bilinear_upsample_weights(upsample_factor,number_of_classes)

#classify in pool3 drectly
pool3_feature = end_points['vgg_16/pool3']
with tf.variable_scope('vgg_16/fc8'):
    aux_logits_8s = slim.conv2d(pool3_feature,number_of_classes,[1,1],
                                activation_fn=None,
                                weights_initializer=tf.zeros_initializer,
                                scope='conv_pool3')
#first,create a upsampling numpy arrary,scecond,create a upsamling tensor by the arrary

#create upsampling arrary
upsample_filter_np_x2_2 = bilinear_upsample_weights(2,number_of_classes)
#create upsampling tensor
upsample_filter_tensor_x2_2 = tf.variable(upsample_filter_np_x2_2,name='vgg_16/fc8/16s_t_conv_x2')
#transpose converlution 2s upsampled_logits by upsample_filter_tensor_x2_2 
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits,upsample_filter_tensor_x2_2,
                                          output_shape=tf.shape(aux_logits_8s),
                                          strides=[1,2,2,1],
                                          padding='SAME')
#element add with  aux_logits_8s and upsampled_logits
upsampled_logits = upsampled_logits + aux_logits_8s
#upsampling arrary 8s 
upsample_filter_np_x8 = bilinear_upsample_weights(8,number_of_classes)
#upsampling tensor 8s
upsample_filter_tensor_x8 = tf.variable(upsample_filter_np_x8,name='vgg_16/fc8/8s_t_conv_x8')

#upsample tensor 16s
# upsample_filter_tensor_x16 = tf.Variable(upsample_filter_np_x16, name='vgg_16/fc8/t_conv_x16')
#transpose converlution 8s
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1,8,8,1],
                                          padding='SAME')
#此时的upsampled_logits他的shape大小和输入一样
#label's channel is not num_of_class ,so,transfer the channel to num_of_class with tf,one_hot
lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
#计算交叉熵
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                          labels=lbl_onehot)
#求和平均
cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))


# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
#找出image不同像素点在channel方向上21个分类上的最大值位置
pred = tf.argmax(upsampled_logits, axis=3)

probabilities = tf.nn.softmax(upsampled_logits)

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_loss)
#使用adam优化器
with tf.variable_scope("adam_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    gradients = optimizer.compute_gradients(loss=cross_entropy_loss)

    for grad_var_pair in gradients:

        current_variable = grad_var_pair[1]
        current_gradient = grad_var_pair[0]

        # Relace some characters from the original variable name
        # tensorboard doesn't accept ':' symbol
        gradient_name_to_save = current_variable.name.replace(":", "_")

        # Let's get histogram of gradients for each layer and
        # visualize them later in tensorboard
        tf.summary.histogram(gradient_name_to_save, current_gradient)

    train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.

#排除不需要从checkpoint恢复的点
vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])

# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
#在pool4计算直接计算logist增加的点
vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])
#优化器增加的点
adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

checkpoint_path = tf.train.latest_checkpoint(log_folder)
continue_train = False
if checkpoint_path:
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % log_folder)
    variables_to_restore = slim.get_model_variables()

    continue_train = True

else:

    # Create an OP that performs the initialization of
    # values of variables to the values from VGG.
    read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
        vgg_checkpoint_path,
        vgg_except_fc8_weights)

    # Initializer for new fc8 weights -- for two classes.
    vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

    # Initializer for adam variables
    optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)


def perform_crf(image, probabilities):

    image = image.squeeze()
    softmax = probabilities.squeeze().transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], number_of_classes)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return res


with sess:
    # Run the initializers.
    sess.run(init_op)
    sess.run(init_local_op)
    if continue_train:
        saver.restore(sess, checkpoint_path)

        logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))
    else:
        sess.run(vgg_fc8_weights_initializer)
        sess.run(optimization_variables_initializer)

        read_vgg_weights_except_fc8_func(sess)
        logging.debug('value initialized...')

    # start data reader
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start = time.time()
    for i in range(FLAGS.max_steps):
        feed_dict_to_use[is_training_placeholder] = True

        gs, _ = sess.run([global_step, train_step], feed_dict=feed_dict_to_use)
        if gs % 10 == 0:
            gs, loss, summary_string = sess.run([global_step, cross_entropy_loss, merged_summary_op], feed_dict=feed_dict_to_use)
            logging.debug("step {0} Current Loss: {1} ".format(gs, loss))
            end = time.time()
            logging.debug("[{0:.2f}] imgs/s".format(10 * batch_size / (end - start)))
            start = end

            summary_string_writer.add_summary(summary_string, i)

            if gs % 100 == 0:
                save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
                logging.debug("Model saved in file: %s" % save_path)

            if gs % 200 == 0:
                eval_folder = os.path.join(FLAGS.output_dir, 'eval')
                if not os.path.exists(eval_folder):
                    os.makedirs(eval_folder)

                logging.debug("validation generated at step [{0}]".format(gs))
                feed_dict_to_use[is_training_placeholder] = False
                val_pred, val_orig_image, val_annot, val_poss = sess.run([pred, orig_img_tensor, annotation_tensor, probabilities],
                                                                         feed_dict=feed_dict_to_use)

                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_img.jpg'.format(gs)), cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_annotation.jpg'.format(gs)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_annot)), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_prediction.jpg'.format(gs)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_pred)), cv2.COLOR_RGB2BGR))

                crf_ed = perform_crf(val_orig_image, val_poss)
                cv2.imwrite(os.path.join(FLAGS.output_dir, 'eval', 'val_{0}_prediction_crfed.jpg'.format(gs)), cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR))

                overlay = cv2.addWeighted(cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR), 1, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR), 0.8, 0)
                cv2.imwrite(os.path.join(FLAGS.output_dir, 'eval', 'val_{0}_overlay.jpg'.format(gs)), overlay)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
    logging.debug("Model saved in file: %s" % save_path)

summary_string_writer.close()
