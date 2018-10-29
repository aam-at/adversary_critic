from __future__ import absolute_import, division, print_function

import os
import time
from collections import OrderedDict
from logging import FileHandler, getLogger

import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets import mnist

from carlini import from_carlini_images, to_carlini_images
from carlini.l2_attack import CarliniL2
from models import create_model
from utils import (AttributeDict, batch_iterator, get_sha,
                   load_training_params, print_results_str, register_metrics,
                   save_images, select_balanced_subset)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_dir", "data", "path to data")
flags.DEFINE_string("load_dir", None, "path to load checkpoint from")
flags.DEFINE_string("working_dir", "test", "path to working dir")
flags.DEFINE_string("adv_data_dir", "adv_data", "path to generated adv data")
flags.DEFINE_string("samples_dir", "samples", "path to samples dir")
flags.DEFINE_string("git_revision", None, "git revision")

# model parameters
flags.DEFINE_string("model_name", None, "name of the model")
flags.DEFINE_string("model", None, "model name (mlp or lenet5)")
flags.DEFINE_string("w_init", "msra", "weights initializer")
flags.DEFINE_string("activation_fn", None, "activation function")
flags.DEFINE_integer("num_classes", -1, "number of classes")
flags.DEFINE_string("layer_dims", None, "dimensions of fully-connected layers")

# test parameters
flags.DEFINE_integer("restore_epoch_index", None, "epoch for which restore model")
flags.DEFINE_integer("seed", 1, "seed for sampling")
flags.DEFINE_integer("batch_size", 100, "batch_index size (default: 100)")
flags.DEFINE_integer("num_examples", 10000, "number of examples to generate perturbation")
flags.DEFINE_string("dataset", "test", "dataset to use (train, validation, test)")

# attack parameters
flags.DEFINE_integer("carlini_batch_size", 100, "batch size (default: 100)")
flags.DEFINE_integer("carlini_max_iter", 10000, "max iterations (default: 1000)")
flags.DEFINE_integer("carlini_binary_steps", 9, "number of binary steps")
flags.DEFINE_float("carlini_confidence", 0, "margin confidence of adversarial examples")

flags.DEFINE_bool("sort_labels", True, "sort labels")
flags.DEFINE_boolean("generate_summary", True, "generate summary images")
flags.DEFINE_integer("print_frequency", 10, "summarize frequency")

FLAGS = tf.app.flags.FLAGS


def setup_experiment():
    np.random.seed(FLAGS.seed)

    if not tf.gfile.Exists(FLAGS.data_dir) or not tf.gfile.IsDirectory(
            FLAGS.data_dir):
        raise ValueError("Could not find folder %s" % FLAGS.data_dir)
    assert FLAGS.batch_size % FLAGS.carlini_batch_size == 0

    if not tf.gfile.Exists(FLAGS.load_dir) or not tf.gfile.IsDirectory(
            FLAGS.load_dir):
        raise ValueError("Could not find folder %s" % FLAGS.load_dir)
    FLAGS.working_dir = os.path.join(FLAGS.working_dir, os.path.basename(os.path.normpath(FLAGS.load_dir)))
    FLAGS.adv_data_dir = os.path.join(FLAGS.working_dir, FLAGS.adv_data_dir)
    FLAGS.samples_dir = os.path.join(FLAGS.working_dir, FLAGS.samples_dir)
    FLAGS.git_revision = get_sha()
    if tf.gfile.Exists(FLAGS.working_dir):
        tf.gfile.DeleteRecursively(FLAGS.working_dir)
    tf.gfile.MakeDirs(FLAGS.working_dir)
    tf.gfile.MakeDirs(FLAGS.adv_data_dir)
    tf.gfile.MakeDirs(FLAGS.samples_dir)

    train_params = load_training_params(FLAGS.load_dir)
    FLAGS.model = train_params['model']
    FLAGS.model_name = train_params['model_name']
    FLAGS.activation_fn = train_params['activation_fn']
    FLAGS.num_classes = train_params['num_classes']
    FLAGS.layer_dims = train_params['layer_dims']

    # configure logging
    logger = getLogger('tensorflow')
    tf.logging.set_verbosity(tf.logging.INFO)
    file_hndl = FileHandler(os.path.join(FLAGS.working_dir, 'tensorflow.log'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)

    # print config
    logging.info({k: v.value for k, v in FLAGS._flags().items()})
    logging.info("Training params: %s", train_params)


def non_converged_indices(x):
    indices = []
    for i, x_i in enumerate(x):
        if not np.allclose(x_i, 0.5 * np.ones_like(x_i)):
            indices.append(i)
    return np.asarray(indices, dtype=np.int32)


def filter_non_coverged(x, y):
    # For some examples carlini method not always converges
    i = 0
    filtered_x = []
    filtered_y = []
    for x_i, y_i in zip(x, y):
        if not np.allclose(x_i, 0.5 * np.ones_like(x_i)):
            filtered_x.append(x_i)
            filtered_y.append(y_i)
        else:
            i += 1
    logging.warn("Failed to converged for {} images".format(i))
    filtered_x = np.asanyarray(filtered_x)
    filtered_y = np.asanyarray(filtered_y)
    return filtered_x, filtered_y


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    setup_experiment()

    mnist_ds = mnist.read_data_sets(
        FLAGS.data_dir, dtype=tf.float32, reshape=False)
    test_ds = getattr(mnist_ds, FLAGS.dataset)

    images = test_ds.images
    labels = test_ds.labels
    if FLAGS.sort_labels:
        ys_indices = np.argsort(labels)
        images = images[ys_indices]
        labels = labels[ys_indices]

    # loaded discriminator number of classes and dims
    img_shape = [None, 1, 28, 28]
    num_classes = FLAGS.num_classes

    X = tf.placeholder(tf.float32, shape=img_shape, name='X')
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    y_onehot = tf.one_hot(y, num_classes)

    # model
    model = create_model(FLAGS, name=FLAGS.model_name)

    def test_model(x, **kwargs):
        return model(x, train=False, **kwargs)

    # wrap model for carlini method
    def carlini_predict(x):
        # carlini requires inputs in [-0.5, 0.5] but network trained on
        # [0, 1] inputs
        x = (2 * x + 1) / 2
        x = tf.transpose(x, [0, 3, 1, 2])
        return test_model(x)['logits']
    carlini_model = AttributeDict({'num_channels': 1,
                                   'image_size': 28,
                                   'num_labels': num_classes,
                                   'predict': carlini_predict})

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # carlini l2 attack
        carlini_l2 = CarliniL2(sess, carlini_model,
                               batch_size=FLAGS.carlini_batch_size,
                               max_iterations=FLAGS.carlini_max_iter,
                               confidence=FLAGS.carlini_confidence,
                               binary_search_steps=FLAGS.carlini_binary_steps,
                               targeted=False)

        def generate_carlini_l2(images, onehot_labels):
            return from_carlini_images(
                carlini_l2.attack(
                    to_carlini_images(images), onehot_labels))
        X_ca_l2 = tf.py_func(generate_carlini_l2, [X, y_onehot], tf.float32)
        X_ca_l2 = tf.reshape(X_ca_l2, tf.shape(X))

        filter_index_l2 = tf.py_func(non_converged_indices, [X_ca_l2], tf.int32)
        filter_index_l2.set_shape([FLAGS.batch_size])
        X_f_l2 = tf.gather(X, filter_index_l2)
        X_ca_f_l2 = tf.gather(X_ca_l2, filter_index_l2)

        # outputs
        outs_x = test_model(X)
        outs_x_ca_l2 = test_model(X_ca_l2)

        # l2 carlini results
        l2_ca = tf.sqrt(tf.reduce_sum((X_ca_l2 - X)**2, axis=(1, 2, 3)))
        l2_ca_norm = l2_ca / tf.sqrt(tf.reduce_sum(X**2, axis=(1, 2, 3)))
        conf_ca = tf.reduce_mean(tf.reduce_max(outs_x_ca_l2['prob'], axis=1))
        l2_ca_f = tf.sqrt(tf.reduce_sum((X_ca_f_l2 - X_f_l2)**2, axis=(1, 2, 3)))
        l2_ca_f_norm = l2_ca_f / tf.sqrt(tf.reduce_sum(X_f_l2**2, axis=(1, 2, 3)))
        smoothness_ca_f = tf.reduce_mean(tf.image.total_variation(X_ca_f_l2))

        nll = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_onehot, outs_x['logits']))
        err = 1 - slim.metrics.accuracy(outs_x['pred'], y)
        err_ca_l2 = 1 - slim.metrics.accuracy(outs_x_ca_l2['pred'], y)
        total_processed_l2 = tf.shape(X_f_l2)[0]

        metrics = OrderedDict([('nll', nll),
                               ('err', err),
                               ('err_ca_l2', err_ca_l2),
                               ('l2_ca', tf.reduce_mean(l2_ca)),
                               ('l2_ca_norm', tf.reduce_mean(l2_ca_norm)),
                               ('conf_ca', conf_ca),
                               ('l2_ca_f', tf.reduce_mean(l2_ca_f)),
                               ('l2_ca_f_norm', tf.reduce_mean(l2_ca_f_norm)),
                               ('smoothness_ca', smoothness_ca_f),
                               ('total_processed_l2', total_processed_l2)])
        metrics_mean, metrics_upd = register_metrics(metrics)
        tf.summary.histogram('y_data', y)
        tf.summary.histogram('y_hat', outs_x['pred'])
        tf.summary.histogram('y_adv', outs_x_ca_l2['pred'])

        # start
        tf.local_variables_initializer().run()
        model_loader = tf.train.Saver(tf.model_variables())
        model_filename = ('model' if FLAGS.restore_epoch_index is None else
                          'model-%d' % FLAGS.restore_epoch_index)
        model_path = os.path.join(FLAGS.load_dir, 'chks', model_filename)
        model_loader.restore(sess, model_path)

        summary_writer = tf.summary.FileWriter(FLAGS.working_dir, sess.graph)
        summaries = tf.summary.merge_all()

        if FLAGS.generate_summary:
            logging.info("Generating samples...")
            summary_images, summary_labels = select_balanced_subset(
                images, labels, num_classes, num_classes)
            summary_images = summary_images.transpose((0, 3, 1, 2))
            err_l2, summary_ca_l2_imgs = (
                sess.run([err_ca_l2, X_ca_l2],
                         {X: summary_images, y: summary_labels}))
            if not np.allclose(err_l2, 1):
                logging.warn("Generated samples are not all mistakes: %f", err_l2)
            save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
            save_images(summary_images, save_path)
            save_path = os.path.join(FLAGS.samples_dir, 'carlini_l2.png')
            save_images(summary_ca_l2_imgs, save_path)
        else:
            logging.debug("Skipping summary...")

        logging.info("Starting...")
        # Carlini is slow. Sample random subset
        if FLAGS.num_examples > 0 and FLAGS.num_examples < images.shape[0]:
            indices = np.arange(images.shape[0])
            np.random.shuffle(indices)
            images = images[indices[:FLAGS.num_examples]]
            labels = labels[indices[:FLAGS.num_examples]]

        X_hat_np = []
        test_iterator = batch_iterator(images, labels, FLAGS.batch_size, shuffle=False)
        start_time = time.time()
        for batch_index, (images, labels) in enumerate(test_iterator, 1):
            ca_l2_imgs, summary = sess.run(
                [X_ca_l2, summaries, metrics_upd],
                {X: images, y: labels})[:2]
            X_hat_np.extend(ca_l2_imgs)
            summary_writer.add_summary(summary, batch_index)

            save_path = os.path.join(FLAGS.samples_dir, 'b%d-ca_l2.png' % batch_index)
            save_images(ca_l2_imgs, save_path)
            save_path = os.path.join(FLAGS.samples_dir, 'b%d-orig.png' % batch_index)
            save_images(images, save_path)

            if batch_index % FLAGS.print_frequency == 0:
                str_bfr = six.StringIO()
                str_bfr.write("Batch {} [{:.2f}s]:".format(batch_index, time.time() - start_time))
                print_results_str(str_bfr, metrics.keys(), sess.run(metrics_mean))
                logging.info(str_bfr.getvalue()[:-1])

        X_hat_np = np.asarray(X_hat_np)
        save_path = os.path.join(FLAGS.adv_data_dir, 'mnist_%s.npz' % FLAGS.dataset)
        np.savez(save_path, X_hat_np)
        logging.info("Saved adv_data to %s", save_path)
        str_bfr = six.StringIO()
        str_bfr.write("Test results [{:.2f}s]:".format(time.time() - start_time))
        print_results_str(str_bfr, metrics.keys(), sess.run(metrics_mean))
        logging.info(str_bfr.getvalue()[:-1])


if __name__ == "__main__":
    tf.app.run(main)
