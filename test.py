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

from attack import deepfool, high_confidence_attack
from models import create_model
from utils import (batch_compute_psnr, batch_compute_ssim, batch_iterator,
                   get_sha, load_training_params, print_results_str,
                   register_metrics, save_images)

flags = tf.flags

flags.DEFINE_string("data_dir", "data", "path to data")
flags.DEFINE_string("load_dir", None, "path to load checkpoint from")
flags.DEFINE_string("working_dir", "test", "path to working dir")
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
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")
flags.DEFINE_string("dataset", "test", "dataset to use (train, validation, test)")
flags.DEFINE_bool("sort_labels", True, "sort labels")

# attack parameters
flags.DEFINE_integer("attack_iter", 100, "maximum number iterations for the attacks")
flags.DEFINE_float("attack_clip", 0.1, "perturbation clip during search")
flags.DEFINE_bool("attack_box_clip", False, "add box clipping for the attacks")
flags.DEFINE_float("attack_overshoot", 0.02, "multiplier for final perturbation")

flags.DEFINE_boolean("bound_random", False, "use random targets for boundary attack")
flags.DEFINE_boolean("hc_random", False, "use random targets for high-confidence attack")
flags.DEFINE_string("hc_confidence", "same", "target mistake confidence")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")

FLAGS = tf.app.flags.FLAGS
logging = tf.logging


def setup_experiment():
    np.random.seed(1)
    tf.set_random_seed(1)
    if not tf.gfile.Exists(FLAGS.data_dir) or not tf.gfile.IsDirectory(
            FLAGS.data_dir):
        raise ValueError("Could not find folder %s" % FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.load_dir) or not tf.gfile.IsDirectory(
            FLAGS.load_dir):
        raise ValueError("Could not find folder %s" % FLAGS.load_dir)
    FLAGS.working_dir = os.path.join(FLAGS.working_dir, os.path.basename(os.path.normpath(FLAGS.load_dir)))
    FLAGS.samples_dir = os.path.join(FLAGS.working_dir, FLAGS.samples_dir)
    FLAGS.git_revision = get_sha()
    if tf.gfile.Exists(FLAGS.working_dir):
        tf.gfile.DeleteRecursively(FLAGS.working_dir)
    tf.gfile.MakeDirs(FLAGS.working_dir)
    tf.gfile.MakeDirs(FLAGS.samples_dir)

    train_params = load_training_params(FLAGS.load_dir)
    FLAGS.model = train_params['model']
    FLAGS.model_name = train_params['model_name']
    FLAGS.activation_fn = train_params['activation_fn']
    FLAGS.num_classes = train_params['num_classes']
    FLAGS.layer_dims = train_params['layer_dims']
    FLAGS.validation_size = train_params['validation_size']

    # configure logging
    logger = getLogger('tensorflow')
    file_hndl = FileHandler(os.path.join(FLAGS.working_dir, 'tensorflow.log'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    logging.set_verbosity(logging.INFO)

    # print config
    logging.info({k: v.value for k, v in FLAGS._flags().items()})
    logging.info("Training params: %s", train_params)


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    setup_experiment()

    mnist_ds = mnist.read_data_sets(
        FLAGS.data_dir, dtype=tf.float32, reshape=False,
        validation_size=FLAGS.validation_size)
    test_ds = getattr(mnist_ds, FLAGS.dataset)

    test_images, test_labels = test_ds.images, test_ds.labels
    if FLAGS.sort_labels:
        ys_indices = np.argsort(test_labels)
        test_images = test_images[ys_indices]
        test_labels = test_labels[ys_indices]

    img_shape = [None, 1, 28, 28]
    X = tf.placeholder(tf.float32, shape=img_shape, name='X')
    y = tf.placeholder(tf.int32, shape=[None])
    y_onehot = tf.one_hot(y, FLAGS.num_classes)

    # model
    model = create_model(FLAGS, name=FLAGS.model_name)

    def test_model(x, **kwargs):
        return model(x, train=False, **kwargs)

    out_x = test_model(X)
    attack_clip = FLAGS.attack_clip if FLAGS.attack_clip > 0 else None
    if FLAGS.attack_box_clip:
        boxmin, boxmax = 0.0, 1.0
    else:
        boxmin, boxmax = None, None
    X_df = deepfool(
        lambda x: test_model(x)['logits'],
        X, labels=y, max_iter=FLAGS.attack_iter, clip_dist=attack_clip,
        over_shoot=FLAGS.attack_overshoot, boxmin=boxmin, boxmax=boxmax)
    X_df_all = deepfool(
        lambda x: test_model(x)['logits'],
        X, max_iter=FLAGS.attack_iter, clip_dist=attack_clip,
        over_shoot=FLAGS.attack_overshoot, boxmin=boxmin, boxmax=boxmax)
    if FLAGS.hc_confidence == 'same':
        confidence = out_x['conf']
    else:
        confidence = float(FLAGS.hc_confidence)
    X_hc = high_confidence_attack(
        lambda x: test_model(x)['logits'],
        X, labels=y, random=FLAGS.hc_random, max_iter=FLAGS.attack_iter,
        clip_dist=attack_clip, confidence=confidence,
        boxmin=boxmin, boxmax=boxmax)
    X_hcd = tf.stop_gradient(X_hc)
    X_rec = high_confidence_attack(
        lambda x: model(x)['logits'], X_hcd, targets=out_x['pred'],
        attack_topk=None, max_iter=FLAGS.attack_iter,
        clip_dist=attack_clip, confidence=out_x['conf'],
        boxmin=boxmin, boxmax=boxmax)

    out_x_df = test_model(X_df)
    out_x_hc = test_model(X_hc)

    reduce_ind = (1, 2, 3)
    X_norm = tf.sqrt(tf.reduce_sum(X**2, axis=reduce_ind))
    l2_df = tf.sqrt(tf.reduce_sum((X_df - X)**2, axis=reduce_ind))
    l2_df_norm = l2_df / X_norm
    smoothness_df = tf.reduce_mean(tf.image.total_variation(X_df))
    l2_df_all = tf.sqrt(tf.reduce_sum((X_df_all - X)**2, axis=reduce_ind))
    l2_df_all_norm = l2_df_all / X_norm
    l2_hc = tf.sqrt(tf.reduce_sum((X_hc - X)**2, axis=reduce_ind))
    l2_hc_norm = l2_hc / X_norm
    smoothness_hc = tf.reduce_mean(tf.image.total_variation(X_hc))
    l1_rec = tf.reduce_sum(tf.abs(X - X_rec), axis=reduce_ind)
    l2_rec = tf.sqrt(tf.reduce_sum((X - X_rec) ** 2, axis=reduce_ind))
    # image noise statistics
    psnr = tf.py_func(batch_compute_psnr, [X, X_df], tf.float32)
    ssim = tf.py_func(batch_compute_ssim, [X, X_df], tf.float32)

    nll = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_onehot, out_x['logits']))
    err = 1 - slim.metrics.accuracy(out_x['pred'], y)
    conf = tf.reduce_mean(out_x['conf'])
    err_df = 1 - slim.metrics.accuracy(out_x_df['pred'], y)
    conf_df = tf.reduce_mean(out_x_df['conf'])
    err_hc = 1 - slim.metrics.accuracy(out_x_hc['pred'], y)
    conf_hc = tf.reduce_mean(out_x_hc['conf'])

    metrics = OrderedDict([('nll', nll), ('err', err),
                           ('conf', conf),
                           ('err_df', err_df),
                           ('err_hc', err_hc),
                           ('l2_df', tf.reduce_mean(l2_df)),
                           ('l2_df_norm', tf.reduce_mean(l2_df_norm)),
                           ('l2_df_all', tf.reduce_mean(l2_df_all)),
                           ('l2_df_all_norm', tf.reduce_mean(l2_df_all_norm)),
                           ('conf_df', conf_df),
                           ('smoothness_df', smoothness_df),
                           ('l2_hc', tf.reduce_mean(l2_hc)),
                           ('l2_hc_norm', tf.reduce_mean(l2_hc_norm)),
                           ('conf_hc', conf_hc),
                           ('smoothness_hc', smoothness_hc),
                           ('l1_rec', tf.reduce_mean(l1_rec)),
                           ('l2_rec', tf.reduce_mean(l2_rec)),
                           ('psnr', tf.reduce_mean(psnr)),
                           ('ssim', tf.reduce_mean(ssim))])
    metrics_mean, metrics_upd = register_metrics(metrics)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.local_variables_initializer().run()
        model_loader = tf.train.Saver(tf.model_variables())
        model_filename = ('model' if FLAGS.restore_epoch_index is None else
                          'model-%d' % FLAGS.restore_epoch_index)
        model_path = os.path.join(FLAGS.load_dir, 'chks', model_filename)
        model_loader.restore(sess, model_path)

        summary_writer = tf.summary.FileWriter(FLAGS.working_dir, sess.graph)
        summaries = tf.summary.merge_all()

        test_iterator = batch_iterator(test_images, test_labels,
                                       FLAGS.batch_size, shuffle=False)
        start_time = time.time()
        for batch_index, (images, labels) in enumerate(test_iterator, 1):
            if batch_index % FLAGS.summary_frequency == 0:
                hc_images, df_images, rec_images, summary = sess.run(
                    [X_hc, X_df, X_rec, summaries, metrics_upd], feed_dict={X: images, y: labels})[:-1]
                save_path = os.path.join(FLAGS.samples_dir, 'epoch_orig-%d.png' % batch_index)
                save_images(images, save_path)
                save_path = os.path.join(FLAGS.samples_dir, 'epoch_hc-%d.png' % batch_index)
                save_images(hc_images, save_path)
                save_path = os.path.join(FLAGS.samples_dir, 'epoch_df-%d.png' % batch_index)
                save_images(df_images, save_path)
                save_path = os.path.join(FLAGS.samples_dir, 'epoch_rec-%d.png' % batch_index)
                save_images(rec_images, save_path)
            else:
                summary = sess.run([metrics_upd, summaries],
                                   feed_dict={X: images, y: labels})[-1]
            summary_writer.add_summary(summary, batch_index)
        str_bfr = six.StringIO()
        str_bfr.write("Test results [{:.2f}s]:".format(time.time() - start_time))
        print_results_str(str_bfr, metrics.keys(), sess.run(metrics_mean),
                          throw_on_nan=False)
        logging.info(str_bfr.getvalue()[:-1])


if __name__ == "__main__":
    tf.app.run(main)
