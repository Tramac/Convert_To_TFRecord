import time
import os
import tensorflow as tf

from utils.preprocess import preprocess_image


class DataGenerator(object):
    def __init__(self, config, split_name):
        self.config = config
        self.split_name = split_name
        self.tfrecord_filename = '{}_{}.tfrecord'.format(self.config.dataset, self.split_name)
        self.tfrecord_path = os.path.join(self.config.data_dir, 'tfrecord_data', self.tfrecord_filename)
        self.image, self.label = self.read_and_decode(self.tfrecord_path)

    def read_and_decode(self, filename):
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label_raw': tf.FixedLenFeature([], tf.string),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                           })
        # method 1
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [height, width, 3])
        label = tf.decode_raw(features['label_raw'], tf.uint8)
        label = tf.reshape(label, [height, width, 1])

        # method 2
        # image = tf.image.decode_jpeg(features['image_raw'])
        # label = tf.image.decode_png(features['label_raw'])

        image, label = preprocess_image(image, label, is_training=self.config.mode)

        return image, label

    def next_batch(self, batch_size):
        image_batch, label_batch = tf.train.shuffle_batch([self.image, self.label],
                                                          batch_size=batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=5)
        return image_batch, label_batch
