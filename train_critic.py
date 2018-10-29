from __future__ import absolute_import, division, print_function

import os
import time
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.training.moving_averages import assign_moving_average

from attack import deepfool, high_confidence_attack, high_confidence_attack_unrolled
from models import create_model, register_model_flags
from utils import (NanError, batch_iterator, binary_accuracy, lr_decay,
                   norm_penalty, print_results_str, register_experiment_flags,
                   register_metrics, save_checkpoint, save_images,
                   select_balanced_subset, setup_experiment)

flags = tf.flags
logging = tf.logging

register_experiment_flags()
register_model_flags(model='mlp')
register_model_flags(model="critic_mlp", activation_fn="lrelu", prefix='critic_')
# tensorflow
flags.DEFINE_float("gpu_memory", 1.0, "gpu memory for session")
# data parameters
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("test_batch_size", 100, "test batch size")
flags.DEFINE_boolean("validation", False, "if true, train model using whole train dataset")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("df_iter", 50, "maximum number of iterations for deepfool attack")
flags.DEFINE_float("df_clip", 0.5, "clip per iteration of deepfool")
flags.DEFINE_float("df_overshoot", 0.02, "overshoot for deepfool iteration")
flags.DEFINE_integer("attack_iter", 5, "number of iterations for the attack")
flags.DEFINE_string("attack_confidence", "class_running_mean", "attack target mistake confidence")
flags.DEFINE_float("attack_label_smoothing", 0.0, "attack label smoothing")
flags.DEFINE_float("attack_overshoot", 0.05, "constant to overshoot for attack")
flags.DEFINE_boolean("attack_random", False, "attack random target samples from output softmax")
flags.DEFINE_boolean("attack_uniform", False, "attack target uniformly")
flags.DEFINE_float("val_attack_confidence", 0.8, "attack target mistake confidence")

# training parameters
flags.DEFINE_integer("pretrain_niter", 1, "number of epochs to pretrain model for")
flags.DEFINE_integer("niter", 100, "number of epochs to train for")
flags.DEFINE_integer("critic_steps", 1, "number of updates per classifier update")

# lr parameters
flags.DEFINE_float("lr", 0.0005, "learning rate for classifier (default: 0.0005)")
flags.DEFINE_string("lr_decay", "step", "learning rate decay option (exp, step, schedule, no)")
flags.DEFINE_float("lr_decay_factor", 0.5, "learning rate decay factor")
flags.DEFINE_integer("lr_decay_step", 40, "learning rate decay step")

# critic lr parameters
flags.DEFINE_float("critic_lr", 0.001, "critic learning rate (default: 0.001)")
flags.DEFINE_string("critic_lr_decay", "step", "learning rate decay option (exp, step, schedule, no)")
flags.DEFINE_float("critic_lr_decay_factor", 0.5, "critic learning rate decay factor")
flags.DEFINE_integer("critic_lr_decay_step", 40, "learning rate decay step for critic")

# regularization parameters
flags.DEFINE_float("lmbd", 0.5, "gan weight for learning model")
flags.DEFINE_float("lmbd_rec_l1", 0, "weight for reconstruction penalty")
flags.DEFINE_float("lmbd_rec_l2", 0.01, "weight for reconstruction penalty")
flags.DEFINE_float("lmbd_grad", 10.0, "weight for gradient penalty")
flags.DEFINE_float("weight_decay", 0, "weight decay")

flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in epochs)")
flags.DEFINE_integer("checkpoint_frequency", -1, "frequency to save model (in epochs)")

FLAGS = tf.app.flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    setup_experiment(logging, FLAGS, "critic_model")

    if FLAGS.validation:
        mnist_ds = mnist.read_data_sets(
            FLAGS.data_dir, dtype=tf.float32, reshape=False, validation_size=0)
        val_ds = mnist_ds.test
    else:
        mnist_ds = mnist.read_data_sets(FLAGS.data_dir, dtype=tf.float32, reshape=False,
                                        validation_size=FLAGS.validation_size)
        val_ds = mnist_ds.validation
    train_ds = mnist_ds.train
    val_ds = mnist_ds.validation
    test_ds = mnist_ds.test
    num_classes = FLAGS.num_classes

    img_shape = [None, 1, 28, 28]
    X = tf.placeholder(tf.float32, shape=img_shape, name='X')
    # placeholder to avoid recomputation of adversarial images for critic
    X_hat_h = tf.placeholder(tf.float32, shape=img_shape, name='X_hat')
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    y_onehot = tf.one_hot(y, num_classes)
    reduce_ind = list(range(1, X.get_shape().ndims))
    # test/validation inputs
    X_v = tf.placeholder(tf.float32, shape=img_shape, name='X_v')
    y_v = tf.placeholder(tf.int32, shape=[None], name='y_v')
    y_v_onehot = tf.one_hot(y_v, num_classes)

    # classifier model
    model = create_model(FLAGS, name=FLAGS.model_name)

    def test_model(x, **kwargs):
        return model(x, train=False, **kwargs)

    # generator
    def generator(inputs, confidence, targets=None):
        return high_confidence_attack_unrolled(
            lambda x: model(x)['logits'], inputs, targets=targets,
            confidence=confidence, max_iter=FLAGS.attack_iter,
            over_shoot=FLAGS.attack_overshoot,
            attack_random=FLAGS.attack_random,
            attack_uniform=FLAGS.attack_uniform,
            attack_label_smoothing=FLAGS.attack_label_smoothing)

    def test_generator(inputs, confidence, targets=None):
        return high_confidence_attack(
            lambda x: test_model(x)['logits'], inputs, targets=targets,
            confidence=confidence, max_iter=FLAGS.df_iter,
            over_shoot=FLAGS.df_overshoot, random=FLAGS.attack_random,
            uniform=FLAGS.attack_uniform, clip_dist=FLAGS.df_clip)

    # discriminator
    critic = create_model(FLAGS, prefix='critic_', name='critic')

    # classifier outputs
    outs_x = model(X)
    outs_x_v = test_model(X_v)
    params = tf.trainable_variables()
    model_weights = [param for param in params if "weights" in param.name]
    vars = tf.model_variables()
    target_conf_v = [None]

    if FLAGS.attack_confidence == "same":
        # set the target confidence to the confidence of the original prediction
        target_confidence = outs_x['conf']
        target_conf_v[0] = target_confidence
    elif FLAGS.attack_confidence == "class_running_mean":
        # set the target confidence to the mean confidence of the specific target
        # use running mean estimate
        class_conf_mean = tf.Variable(np.ones(num_classes, dtype=np.float32))
        batch_conf_mean = tf.unsorted_segment_mean(
            outs_x['conf'], outs_x['pred'], num_classes)
        # if batch does not contain predictions for the specific target
        # (zeroes), replace zeroes with stored class mean (previous batch)
        batch_conf_mean = tf.where(
            tf.not_equal(batch_conf_mean, 0), batch_conf_mean, class_conf_mean)
        # update class confidence mean
        class_conf_mean = assign_moving_average(class_conf_mean, batch_conf_mean, 0.5)
        # init class confidence during pre-training
        tf.add_to_collection("PREINIT_OPS", class_conf_mean)

        def target_confidence(targets_onehot):
            targets = tf.argmax(targets_onehot, axis=1)
            check_conf = tf.Assert(tf.reduce_all(tf.not_equal(class_conf_mean, 0)), [class_conf_mean])
            with tf.control_dependencies([check_conf]):
                t = tf.gather(class_conf_mean, targets)
            target_conf_v[0] = t
            return tf.stop_gradient(t)
    else:
        target_confidence = float(FLAGS.attack_confidence)
        target_conf_v[0] = target_confidence

    X_hat = generator(X, target_confidence)
    outs_x_hat = model(X_hat)
    # select examples for which attack succeeded (changed the prediction)
    X_hat_filter = tf.not_equal(outs_x['pred'], outs_x_hat['pred'])
    X_hat_f = tf.boolean_mask(X_hat, X_hat_filter)
    X_f = tf.boolean_mask(X, X_hat_filter)

    outs_x_f = model(X_f)
    outs_x_hat_f = model(X_hat_f)
    X_hatd = tf.stop_gradient(X_hat)
    X_rec = generator(X_hatd, outs_x['conf'], outs_x['pred'])
    X_rec_f = tf.boolean_mask(X_rec, X_hat_filter)

    # validation/test adversarial examples
    X_v_hat = test_generator(X_v, FLAGS.val_attack_confidence)
    X_v_hatd = tf.stop_gradient(X_v_hat)
    X_v_rec = test_generator(X_v_hatd, outs_x_v['conf'], targets=outs_x_v['pred'])
    X_v_hat_df = deepfool(lambda x: test_model(x)['logits'], X_v, y_v,
                          max_iter=FLAGS.df_iter, clip_dist=FLAGS.df_clip)
    X_v_hat_df_all = deepfool(lambda x: test_model(x)['logits'], X_v,
                              max_iter=FLAGS.df_iter, clip_dist=FLAGS.df_clip)

    y_hat = outs_x['pred']
    y_adv = outs_x_hat['pred']
    y_adv_f = outs_x_hat_f['pred']
    tf.summary.histogram('y_data', y, collections=["model_summaries"])
    tf.summary.histogram('y_hat', y_hat, collections=["model_summaries"])
    tf.summary.histogram('y_adv', y_adv, collections=["model_summaries"])

    # critic outputs
    critic_outs_x = critic(X)
    critic_outs_x_hat = critic(X_hat_f)
    critic_params = list(set(tf.trainable_variables()) - set(params))
    critic_vars = list(set(tf.trainable_variables()) - set(vars))

    # binary logits for a specific target
    logits_data = critic_outs_x['logits']
    logits_data_flt = tf.reshape(logits_data, (-1,))
    z_data = tf.gather(logits_data_flt, tf.range(tf.shape(X)[0]) * num_classes + y)
    logits_adv = critic_outs_x_hat['logits']
    logits_adv_flt = tf.reshape(logits_adv, (-1,))
    z_adv = tf.gather(logits_adv_flt, tf.range(tf.shape(X_hat_f)[0]) * num_classes + y_adv_f)

    # classifier/generator losses
    nll = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_onehot, outs_x['logits']))
    nll_v = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_v_onehot, outs_x_v['logits']))
    # gan losses
    gan = tf.losses.sigmoid_cross_entropy(tf.ones_like(z_adv), z_adv)
    rec_l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(X_f - X_rec_f), axis=reduce_ind))
    rec_l2 = tf.reduce_mean(tf.reduce_sum((X_f - X_rec_f) ** 2, axis=reduce_ind))

    weight_decay = slim.apply_regularization(slim.l2_regularizer(1.0), model_weights[:-1])
    pretrain_loss = nll + 5e-6 * weight_decay
    loss = nll + FLAGS.lmbd * gan
    if FLAGS.lmbd_rec_l1 > 0:
        loss += FLAGS.lmbd_rec_l1 * rec_l1
    if FLAGS.lmbd_rec_l2 > 0:
        loss += FLAGS.lmbd_rec_l2 * rec_l2
    if FLAGS.weight_decay > 0:
        loss += FLAGS.weight_decay * weight_decay

    # critic loss
    critic_gan_data = tf.losses.sigmoid_cross_entropy(tf.ones_like(z_data), z_data)
    # use placeholder for X_hat to avoid recomputation of adversarial noise
    y_adv_h = model(X_hat_h)['pred']
    logits_adv_h = critic(X_hat_h)['logits']
    logits_adv_flt_h = tf.reshape(logits_adv_h, (-1,))
    z_adv_h = tf.gather(logits_adv_flt_h, tf.range(tf.shape(X_hat_h)[0]) * num_classes + y_adv_h)
    critic_gan_adv = tf.losses.sigmoid_cross_entropy(tf.zeros_like(z_adv_h), z_adv_h)
    critic_gan = critic_gan_data + critic_gan_adv

    # Gulrajani discriminator regularizer (we do not interpolate)
    critic_grad_data = tf.gradients(z_data, X)[0]
    critic_grad_adv = tf.gradients(z_adv_h, X_hat_h)[0]
    critic_grad_penalty = norm_penalty(critic_grad_adv) + norm_penalty(critic_grad_data)
    critic_loss = critic_gan + FLAGS.lmbd_grad * critic_grad_penalty

    # classifier model_metrics
    err = 1 - slim.metrics.accuracy(outs_x['pred'], y)
    conf = tf.reduce_mean(outs_x['conf'])
    err_hat = 1 - slim.metrics.accuracy(test_model(X_hat)['pred'], outs_x['pred'])
    err_hat_f = 1 - slim.metrics.accuracy(test_model(X_hat_f)['pred'], outs_x_f['pred'])
    err_rec = 1 - slim.metrics.accuracy(test_model(X_rec)['pred'], outs_x['pred'])
    conf_hat = tf.reduce_mean(test_model(X_hat)['conf'])
    conf_hat_f = tf.reduce_mean(test_model(X_hat_f)['conf'])
    conf_rec = tf.reduce_mean(test_model(X_rec)['conf'])
    err_v = 1 - slim.metrics.accuracy(outs_x_v['pred'], y_v)
    conf_v_hat = tf.reduce_mean(test_model(X_v_hat)['conf'])
    l2_hat = tf.sqrt(tf.reduce_sum((X_f - X_hat_f) ** 2, axis=reduce_ind))
    tf.summary.histogram('l2_hat', l2_hat, collections=["model_summaries"])

    # critic model_metrics
    critic_err_data = 1 - binary_accuracy(z_data, tf.ones(tf.shape(z_data), tf.bool), 0.0)
    critic_err_adv = 1 - binary_accuracy(z_adv, tf.zeros(tf.shape(z_adv), tf.bool), 0.0)

    # validation model_metrics
    err_df = 1 - slim.metrics.accuracy(test_model(X_v_hat_df)['pred'], y_v)
    err_df_all = 1 - slim.metrics.accuracy(test_model(X_v_hat_df_all)['pred'],
                                           outs_x_v['pred'])
    l2_v_hat = tf.sqrt(tf.reduce_sum((X_v - X_v_hat)**2, axis=reduce_ind))
    l2_v_rec = tf.sqrt(tf.reduce_sum((X_v - X_v_rec) ** 2, axis=reduce_ind))
    l1_v_rec = tf.reduce_sum(tf.abs(X_v - X_v_rec), axis=reduce_ind)
    l2_df = tf.sqrt(tf.reduce_sum((X_v - X_v_hat_df) ** 2, axis=reduce_ind))
    l2_df_norm = l2_df / tf.sqrt(tf.reduce_sum(X_v ** 2, axis=reduce_ind))
    l2_df_all = tf.sqrt(tf.reduce_sum((X_v - X_v_hat_df_all) ** 2, axis=reduce_ind))
    l2_df_norm_all = l2_df_all / tf.sqrt(tf.reduce_sum(X_v ** 2, axis=reduce_ind))
    tf.summary.histogram('l2_df', l2_df, collections=["adv_summaries"])
    tf.summary.histogram('l2_df_norm', l2_df_norm, collections=["adv_summaries"])

    # model_metrics
    pretrain_model_metrics = OrderedDict([('nll', nll),
                                          ('weight_decay', weight_decay),
                                          ('err', err)])
    model_metrics = OrderedDict([('loss', loss),
                                 ('nll', nll),
                                 ('l2_hat', tf.reduce_mean(l2_hat)),
                                 ('gan', gan),
                                 ('rec_l1', rec_l1),
                                 ('rec_l2', rec_l2),
                                 ('weight_decay', weight_decay),
                                 ('err', err),
                                 ('conf', conf),
                                 ('err_hat', err_hat),
                                 ('err_hat_f', err_hat_f),
                                 ('conf_t', tf.reduce_mean(target_conf_v[0])),
                                 ('conf_hat', conf_hat),
                                 ('conf_hat_f', conf_hat_f),
                                 ('err_rec', err_rec),
                                 ('conf_rec', conf_rec)])
    critic_metrics = OrderedDict([('c_loss', critic_loss),
                                  ('c_gan', critic_gan),
                                  ('c_gan_data', critic_gan_data),
                                  ('c_gan_adv', critic_gan_adv),
                                  ('c_grad_norm', critic_grad_penalty),
                                  ('c_err_adv', critic_err_adv),
                                  ('c_err_data', critic_err_data)])
    val_metrics = OrderedDict([('nll', nll_v),
                               ('err', err_v)])
    adv_metrics = OrderedDict([('l2_df', tf.reduce_mean(l2_df)),
                               ('l2_df_norm', tf.reduce_mean(l2_df_norm)),
                               ('l2_df_all', tf.reduce_mean(l2_df_all)),
                               ('l2_df_all_norm', tf.reduce_mean(l2_df_norm_all)),
                               ('l2_hat', tf.reduce_mean(l2_v_hat)),
                               ('conf_hat', conf_v_hat),
                               ('l1_rec', tf.reduce_mean(l1_v_rec)),
                               ('l2_rec', tf.reduce_mean(l2_v_rec)),
                               ('err_df', err_df),
                               ('err_df_all', err_df_all)])

    pretrain_metric_mean, pretrain_metric_upd = register_metrics(
        pretrain_model_metrics, collections="pretrain_model_summaries")
    metric_mean, metric_upd = register_metrics(
        model_metrics, collections="model_summaries")
    critic_metric_mean, critic_metric_upd = register_metrics(
        critic_metrics, collections="critic_summaries")
    val_metric_mean, val_metric_upd = register_metrics(
        val_metrics, prefix="val_", collections="val_summaries")
    adv_metric_mean, adv_metric_upd = register_metrics(
        adv_metrics, collections="adv_summaries")
    metrics_reset = tf.variables_initializer(tf.local_variables())

    # training ops
    lr = tf.Variable(FLAGS.lr, trainable=False)
    critic_lr = tf.Variable(FLAGS.critic_lr, trainable=False)
    tf.summary.scalar('lr', lr, collections=["model_summaries"])
    tf.summary.scalar('critic_lr', critic_lr, collections=["critic_summaries"])

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

    preinit_ops = tf.get_collection("PREINIT_OPS")
    with tf.control_dependencies(preinit_ops):
        pretrain_solver = optimizer.minimize(pretrain_loss, var_list=params)
    solver = optimizer.minimize(loss, var_list=params)
    critic_solver = (tf.train.AdamOptimizer(learning_rate=critic_lr, beta1=0.5)
                     .minimize(critic_loss, var_list=critic_params))

    # train
    summary_images, summary_labels = select_balanced_subset(
        train_ds.images, train_ds.labels, num_classes, num_classes)
    summary_images = summary_images.transpose((0, 3, 1, 2))
    save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
    save_images(summary_images, save_path)

    if FLAGS.gpu_memory < 1.0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        config = None
    with tf.Session(config=config) as sess:
        try:
            # summaries
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            summaries = tf.summary.merge_all("model_summaries")
            critic_summaries = tf.summary.merge_all("critic_summaries")
            val_summaries = tf.summary.merge_all("val_summaries")
            adv_summaries = tf.summary.merge_all("adv_summaries")

            # initialization
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            # pretrain model
            if FLAGS.pretrain_niter > 0:
                logging.info("Model pretraining")
                for epoch in range(1, FLAGS.pretrain_niter + 1):
                    train_iterator = batch_iterator(train_ds.images, train_ds.labels,
                                                    FLAGS.batch_size, shuffle=True)
                    sess.run(metrics_reset)

                    start_time = time.time()
                    for ind, (images, labels) in enumerate(train_iterator):
                        sess.run([pretrain_solver, pretrain_metric_upd],
                                 feed_dict={X: images, y: labels})

                    str_bfr = six.StringIO()
                    str_bfr.write("Pretrain epoch [{}, {:.2f}s]:".format(epoch, time.time() - start_time))
                    print_results_str(str_bfr, pretrain_model_metrics.keys(),
                                      sess.run(pretrain_metric_mean))
                    print_results_str(str_bfr, critic_metrics.keys(),
                                      sess.run(critic_metric_mean))
                    logging.info(str_bfr.getvalue()[:-1])

            # training
            for epoch in range(1, FLAGS.niter + 1):
                train_iterator = batch_iterator(train_ds.images, train_ds.labels,
                                                FLAGS.batch_size, shuffle=True)
                sess.run(metrics_reset)

                start_time = time.time()
                for ind, (images, labels) in enumerate(train_iterator):
                    batch_index = (epoch - 1) * (train_ds.images.shape[0] // FLAGS.batch_size) + ind
                    # train critic for several steps
                    X_hat_np = sess.run(X_hat, feed_dict={X: images})
                    for _ in range(FLAGS.critic_steps - 1):
                        sess.run([critic_solver], feed_dict={X: images, y: labels, X_hat_h: X_hat_np})
                    else:
                        summary = sess.run(
                            [critic_solver, critic_metric_upd, critic_summaries],
                            feed_dict={X: images, y: labels, X_hat_h: X_hat_np})[-1]
                        summary_writer.add_summary(summary, batch_index)
                    # train model
                    summary = sess.run([solver, metric_upd, summaries],
                                       feed_dict={X: images, y: labels})[-1]
                    summary_writer.add_summary(summary, batch_index)

                str_bfr = six.StringIO()
                str_bfr.write("Train epoch [{}, {:.2f}s]:".format(epoch, time.time() - start_time))
                print_results_str(str_bfr, model_metrics.keys(), sess.run(metric_mean))
                print_results_str(str_bfr, critic_metrics.keys(), sess.run(critic_metric_mean))
                logging.info(str_bfr.getvalue()[:-1])

                val_iterator = batch_iterator(val_ds.images, val_ds.labels, 100, shuffle=False)
                for images, labels in val_iterator:
                    summary = sess.run(
                        [val_metric_upd, val_summaries],
                        feed_dict={X_v: images, y_v: labels})[-1]
                    summary_writer.add_summary(summary, epoch)
                str_bfr = six.StringIO()
                str_bfr.write("Valid epoch [{}]:".format(epoch))
                print_results_str(str_bfr, val_metrics.keys(), sess.run(val_metric_mean))
                logging.info(str_bfr.getvalue()[:-1])

                # learning rate decay
                update_lr = lr_decay(lr, epoch)
                if update_lr is not None:
                    sess.run(update_lr)
                    logging.debug("learning rate was updated to: {:.10f}".format(lr.eval()))
                critic_update_lr = lr_decay(critic_lr, epoch, prefix='critic_')
                if critic_update_lr is not None:
                    sess.run(critic_update_lr)
                    logging.debug("critic learning rate was updated to: {:.10f}".format(critic_lr.eval()))

                if epoch % FLAGS.summary_frequency == 0:
                    samples_hat, samples_rec, samples_df, summary = sess.run(
                        [X_v_hat, X_v_rec, X_v_hat_df, adv_summaries, adv_metric_upd],
                        feed_dict={X_v: summary_images, y_v: summary_labels})[:-1]
                    summary_writer.add_summary(summary, epoch)
                    save_path = os.path.join(FLAGS.samples_dir, 'epoch_orig-%d.png' % epoch)
                    save_images(summary_images, save_path)
                    save_path = os.path.join(FLAGS.samples_dir, 'epoch-%d.png' % epoch)
                    save_images(samples_hat, save_path)
                    save_path = os.path.join(FLAGS.samples_dir, 'epoch_rec-%d.png' % epoch)
                    save_images(samples_rec, save_path)
                    save_path = os.path.join(FLAGS.samples_dir, 'epoch_df-%d.png' % epoch)
                    save_images(samples_df, save_path)

                    str_bfr = six.StringIO()
                    str_bfr.write("Summary epoch [{}]:".format(epoch))
                    print_results_str(str_bfr, adv_metrics.keys(), sess.run(adv_metric_mean))
                    logging.info(str_bfr.getvalue()[:-1])

                if FLAGS.checkpoint_frequency != -1 and epoch % FLAGS.checkpoint_frequency == 0:
                    save_checkpoint(sess, vars, epoch=epoch)
                    save_checkpoint(sess, critic_vars, name="critic_model", epoch=epoch)
        except KeyboardInterrupt:
            logging.debug("Keyboard interrupt. Stopping training...")
        except NanError as e:
            logging.info(e)
        finally:
            sess.run(metrics_reset)
            save_checkpoint(sess, vars)
            save_checkpoint(sess, critic_vars, name="critic_model")

        # final accuracy
        test_iterator = batch_iterator(test_ds.images, test_ds.labels, 100, shuffle=False)
        for images, labels in test_iterator:
            sess.run([val_metric_upd],
                     feed_dict={X_v: images, y_v: labels})
        str_bfr = six.StringIO()
        str_bfr.write("Final epoch [{}]:".format(epoch))
        for metric_name, metric_value in zip(val_metrics.keys(), sess.run(val_metric_mean)):
            str_bfr.write(" {}: {:.6f},".format(metric_name, metric_value))
        logging.info(str_bfr.getvalue()[:-1])


if __name__ == "__main__":
    tf.app.run()
