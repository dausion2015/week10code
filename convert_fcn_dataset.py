#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

#generate 256*256*256 longth RGB 0-vector
cm2lbl = np.zeros(256**3) 
#GRB是256进制，把colormap的每个颜色对应的类别编号存入RGB 0-vector
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')   #very pixel as int32
    # cv2.imread. default channel layout is BGR
    #find the num of class of label pic's per pixel by RGB 0-vector 
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])  #return bp.array


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()    #read data
    img_label = cv2.imread(label)    #read label by opencv
    img_mask = image2label(img_label) #calculate the class of label pic 's pixels  type np.arrary
    encoded_label = img_mask.astype(np.uint8).tobytes()  #change the datatype to uint8 , then ,to the bytes

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inf.name.encode('utf8')])),     #unicode to utf8
        'image/encoded': tf.train.Feature(float_list=tf.train.FloatList(value=[encoded_data])),
        'image/label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
        'image/format': tf.train.Feature(float_list=tf.train.BytesList(value=['JPEG'.encode()])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    ''' file_pars: train_files/val_files  return zip(data, label) train_set or val_set input_data and label_data dir  
        output_filename: train_output_path('fcn_train.record')/val_output_path( 'fcn_val.record')
    '''

    writer = tf.python_io.TFRecordWriter(output_filename) # generate writer to write example data and output specify filename
    for data,label in file_pars:                          # form data and label list get the pic dir to generate example
        example = dict_to_tf_example(data,label)
        writer.write(example.SerializeToString())          # write to tf.record
    writer.close()
        

def read_images_names(root, train=True): # root dir  train_obj name or val_obj name
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))   # input data
        label.append('%s/SegmentationClass/%s.png' % (root, fname))  #ground_truh
    return zip(data, label)  #generate train_data and label_data dir tuple


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
