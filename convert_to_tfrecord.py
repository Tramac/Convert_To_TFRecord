import os
import random
import numpy as np
import tensorflow as tf

from PIL import Image
from configs.config import args as config


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def convert_to_example(image, label, height, width):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': bytes_feature(label),
        'image_raw': bytes_feature(image),
        'height': int64_feature(height),
        'width': int64_feature(width)
    }))
    return example


def convert_to_tfrecord(data_dir, output_dir, shuffling=False, max_samples=None):
    tfrecord_filename = os.path.join(output_dir, '{}_{}.tfrecord'.format(config.dataset, config.set))
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    examples_path = os.path.join(data_dir, 'ImageSets', 'Segmentation', config.set + '.txt')
    examples_list = read_examples_list(examples_path)
    if max_samples:
        examples_list = examples_list[:max_samples]
    if shuffling:
        random.shuffle(examples_list)
    for idx, example in enumerate(examples_list):
        image_path = os.path.join(data_dir, "JPEGImages", example + '.jpg')
        label_path = os.path.join(data_dir, "SegmentationClass", example + '.png')

        # method 1
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        height, width = image.shape[0], image.shape[1]
        image_raw = image.tobytes()
        label_raw = label.tobytes()

        # method 2
        # image = np.array(Image.open(image_path))
        # height, width = image.shape[0], image.shape[1]
        # image_raw = tf.gfile.FastGFile(image_path, 'r').read()
        # label_raw = tf.gfile.FastGFile(label_path, 'r').read()

        example = convert_to_example(image_raw, label_raw, height, width)
        writer.write(example.SerializeToString())

    writer.close()


def main():
    data_dir = os.path.join(config.data_dir, config.dataset)
    output_dir = os.path.join(config.data_dir, "tfrecord_data")

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    convert_to_tfrecord(data_dir, output_dir)

    print('\nFinished converting the {} dataset!'.format(config.dataset))


if __name__ == '__main__':
    main()
