from __future__ import absolute_import, division, print_function

import ast
import functools
import json
import os
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage.measure import compare_psnr, compare_ssim

flags = tf.flags
logging = tf.logging

FLAGS = tf.app.flags.FLAGS


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            value = self.__getitem__(key)
        except KeyError as exc:
            return None
        if isinstance(value, dict):
            value = AttributeDict(value)
        return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


def prepare_dir(dir_path, subdir_name):
    base = os.path.join(dir_path, subdir_name)
    i = 0
    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except OSError:
            i += 1
    return name


def load_params(str):
    try:
        return ast.literal_eval(str)
    except:
        return json.loads(str)


def load_training_params(path):
    with open(os.path.join(path, 'tensorflow.log'), 'r') as f:
        return load_params(f.readline())


def register_metrics(metrics_dict, prefix="", collections=None):
    if collections is not None and not isinstance(collections, (tuple, list)):
        collections = [collections]
    for metric_name, metric_value in metrics_dict.items():
        tf.summary.scalar("%s%s" % (prefix, metric_name), metric_value,
                          collections=collections)
    metric_mean, metric_upd = slim.metrics.aggregate_metrics(
        * [
            tf.metrics.mean(value, name=name)
            for name, value in metrics_dict.items()
        ])
    return metric_mean, metric_upd


def save_checkpoint(sess, model_vars, name="model", epoch=None):
    model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=10)
    model_save_path = os.path.join(FLAGS.train_dir, 'chks', name)
    model_saver.save(sess, model_save_path, global_step=epoch)
    if epoch is not None:
        logging.info("Model `%s` saved for epoch %d", name, epoch)
    else:
        logging.info("Final model `%s` saved", name)


class NanError(BaseException):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "NanError: %s has nan value" % self.message


def print_results_str(str_bfr, names, values, throw_on_nan=True):
    for metric_name, metric_value in zip(names, values):
        if np.isnan(metric_value):
            if throw_on_nan:
                raise NanError(metric_name)
            else:
                metric_value = -1
        str_bfr.write(" {}: {:.6f},".format(metric_name, metric_value))


@contextmanager
def redirect_stderr(new_target):
    old_target, sys.stderr = sys.stderr, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


@contextmanager
def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


@contextmanager
def silence_stderr():
    with redirect_stderr(open(os.devnull, 'w')) as redirect:
        yield redirect


def save_images(images, path, **kwargs):
    import torch
    from torchvision.utils import save_image
    if "nrow" not in kwargs:
        kwargs["nrow"] = int(np.sqrt(images.shape[0]))
    save_image(torch.from_numpy(images), path, **kwargs)


def register_experiment_flags():
    # experiment parameters
    flags.DEFINE_string("name", None, "name of the experiment")
    flags.DEFINE_integer("seed", 1, "experiment seed")
    flags.DEFINE_string("data_dir", "data", "path to data")
    flags.DEFINE_string("train_dir", "runs", "path to working dir")
    flags.DEFINE_string("chks_dir", "chks", "path to checks dir")
    flags.DEFINE_string("samples_dir", "samples", "path to samples dir")
    flags.DEFINE_string("git_revision", None, "git revision")


def setup_experiment(logger, FLAGS, default_name):
    from logging import FileHandler, getLogger
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    dict_values = {k: v.value for k, v in FLAGS._flags().items()}
    if FLAGS.name is None:
        FLAGS.name = default_name % dict_values
    FLAGS.git_revision = get_sha()

    if not tf.gfile.Exists(FLAGS.data_dir) or not tf.gfile.IsDirectory(FLAGS.data_dir):
        raise ValueError("Could not find folder %s" % FLAGS.data_dir)
    FLAGS.train_dir = prepare_dir(FLAGS.train_dir, FLAGS.name)
    FLAGS.chks_dir = os.path.join(FLAGS.train_dir, FLAGS.chks_dir)
    FLAGS.samples_dir = os.path.join(FLAGS.train_dir, FLAGS.samples_dir)
    tf.gfile.MakeDirs(FLAGS.chks_dir)
    tf.gfile.MakeDirs(FLAGS.samples_dir)

    # configure logging
    logger = getLogger('tensorflow')
    file_hndl = FileHandler(os.path.join(FLAGS.train_dir, 'tensorflow.log'))
    file_hndl.setLevel(tf.logging.DEBUG)
    logger.addHandler(file_hndl)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # print config
    train_params = json.dumps({k: v.value
                               for k, v in FLAGS._flags().items()}, sort_keys=True)
    logger.info(train_params)


# data utils
def batch_iterator(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    # reshape for NCHW
    inputs = inputs.transpose((0, 3, 1, 2))
    n_samples = inputs.shape[0]
    if shuffle:
        # Shuffles indicies of training data, so we can draw batches
        # from random indicies instead of shuffling whole data
        indx = np.random.permutation(range(n_samples))
    else:
        indx = range(n_samples)
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = inputs[indx[sl]]
        y_batch = targets[indx[sl]]
        yield X_batch, y_batch


def select_balanced_subset(X, y, num_classes=10, samples_per_class=10, seed=1):
    total_samples = num_classes * samples_per_class
    X_subset = np.zeros([total_samples] + list(X.shape[1:]), dtype=X.dtype)
    y_subset = np.zeros((total_samples,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for i in range(num_classes):
        yi_indices = np.where(y == i)[0]
        rng.shuffle(yi_indices)
        X_subset[samples_per_class * i:(i + 1) * samples_per_class, ...] = X[yi_indices[:samples_per_class]]
        y_subset[samples_per_class * i:(i + 1) * samples_per_class] = i
    return X_subset, y_subset


def find_cluster_centers(X, y):
    classes = np.unique(y)
    class_centers = []
    for y_i in classes:
        X_i = X[np.where(y == y_i)]
        mu = np.mean(X_i, axis=0, keepdims=True)
        diff = np.sum((X_i - mu)**2, axis=(1, 2, 3))
        class_center = X_i[diff.argmin(axis=0)]
        class_centers.append(class_center)
    class_centers = np.asarray(class_centers)
    return class_centers, classes


def batch_compute_psnr(im_true, im_test):
    psnr = np.zeros(im_true.shape[0], dtype=im_true.dtype)
    for i in range(im_true.shape[0]):
        if not np.allclose(im_true[i], im_test[i]):
            psnr[i] = compare_psnr(im_true[i], im_test[i])
        else:
            psnr[i] = np.nan
    return psnr[~np.isnan(psnr)]


def batch_compute_ssim(im_true, im_test):
    ssim = np.zeros(im_true.shape[0], dtype=im_true.dtype)
    im_true = np.transpose(im_true, (0, 2, 3, 1))
    im_test = np.transpose(im_test, (0, 2, 3, 1))
    for i in range(im_true.shape[0]):
        with silence_stderr():
            ssim[i] = compare_ssim(im_true[i], im_test[i], dynamic_range=1.0,
                                   multichannel=True)
    return ssim


# tensorflow utils
def binary_accuracy(predictions, targets, threshold=0.5):
    if targets.dtype in [tf.int32, tf.int64]:
        targets = tf.cast(targets, tf.bool)
    is_one = tf.greater_equal(predictions, threshold)
    is_correct = tf.cast(tf.equal(is_one, targets), tf.float32)
    return tf.reduce_mean(is_correct)


def l2_normalize(x, dim, epsilon=1e-12, name=None):
    """Stable l2 normalization
    """
    with tf.name_scope(name, "l2_normalize", [x]) as name:
        x = tf.convert_to_tensor(x, name="x")
        x /= (epsilon + tf.reduce_max(tf.abs(x), dim, keepdims=True))
        square_sum = tf.reduce_sum(tf.square(x), dim, keepdims=True)
        x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
        return tf.multiply(x, x_inv_norm, name)


def jacobian(y, x, pack_axis=1):
    jac = [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y, axis=1)]
    return tf.stack(jac, axis=pack_axis)


def random_targets(logits, labels, uniform=False):
    batch_size = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]
    batch_indices = tf.range(batch_size)

    labels_onehot = tf.one_hot(labels, depth=num_classes)
    if uniform:
        masked_logits = tf.ones_like(logits)
        masked_logits = tf.where(tf.equal(labels_onehot, 1), -np.inf * tf.ones_like(logits), masked_logits)
    else:
        masked_logits = tf.where(tf.equal(labels_onehot, 1), -np.inf * tf.ones_like(logits), logits)

    targets = tf.cast(tf.multinomial(masked_logits, 1), batch_indices.dtype)
    targets = tf.reshape(targets, (-1, ))
    return targets


def lr_decay(lr, epoch, prefix=''):
    lr_decay = getattr(FLAGS, '%slr_decay' % prefix)
    update_op = None
    if lr_decay == 'no':
        pass
    else:
        lr_decay_factor = getattr(FLAGS, '%slr_decay_factor' % prefix)
        if lr_decay == 'exp':
            lr_decay_start = getattr(FLAGS, '%slr_decay_start' % prefix)
            if epoch > lr_decay_start:
                update_op = tf.assign(lr, lr.eval() * lr_decay_factor)
        elif lr_decay == 'step':
            # step learning rate decay
            lr_decay_step = getattr(FLAGS, '%slr_decay_step' % prefix)
            if epoch % lr_decay_step == 0:
                update_op = tf.assign(lr, lr.eval() * lr_decay_factor)
        elif lr_decay == 'schedule':
            # schedule learning rate decay
            lr_schedule = getattr(FLAGS, '%slr_schedule' % prefix)
            if epoch in map(int, lr_schedule.split('-')):
                update_op = tf.assign(lr, lr.eval() * lr_decay_factor)
        else:
            raise ValueError("Unknown lr decay option")
    return update_op


def prediction(prob, name='predictions'):
    return tf.cast(tf.argmax(prob, axis=1), tf.int32, name=name)


def lrelu(x, leak=0.2, scope=None):
    with tf.variable_scope(scope, 'lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def gaussian_noise(x, stddev, scope=None):
    if stddev != 0.0:
        with tf.variable_scope(scope, 'wnoise'):
            return x + tf.random_normal(shape=tf.shape(x), mean=0., stddev=stddev)
    else:
        return x


def norm_penalty(x, slope=1.0):
    reduce_ind = list(range(1, x.get_shape().ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(x), axis=reduce_ind))
    return tf.reduce_mean((slopes - slope)**2)


def with_end_points(model):

    @functools.wraps(model)
    def gather_end_points(*args, **kwargs):
        logits = model(*args, **kwargs)
        predictions = prediction(logits)
        prob = slim.softmax(logits, scope='prob')
        conf = tf.reduce_max(prob, axis=1)
        end_points = {
            'logits': logits,
            'pred': predictions,
            'prob': prob,
            'conf': conf
        }
        return end_points

    return gather_end_points


# git utils
def get_sha(repo='.'):
    """
    Grabs the current SHA-1 hash of the given directory's git HEAD-revision.
    The output of this is equivalent to calling git rev-parse HEAD.

    Be aware that a missing git repository will make this return an error message,
    which is not a valid hash.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.decode('ascii').strip()
