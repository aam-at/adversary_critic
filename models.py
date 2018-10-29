from __future__ import absolute_import, division, print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import gaussian_noise, lrelu, with_end_points

flags = tf.flags
logging = tf.logging

FLAGS = tf.app.flags.FLAGS


def register_model_flags(model_name="model", model="lenet5", w_init="msra", activation_fn="relu",
                         num_classes=10, layer_dims="1200-1200-1200", prefix=''):
    # model parameters
    flags.DEFINE_string("%smodel_name" % prefix, model_name,
                        "name of the model")
    flags.DEFINE_string("%smodel" % prefix, model, "model name (mlp or lenet5)")
    flags.DEFINE_string("%sw_init" % prefix, w_init, "weights initializer")
    flags.DEFINE_string("%sactivation_fn" % prefix, activation_fn,
                        "activation function")
    flags.DEFINE_integer("%snum_classes" % prefix, num_classes,
                         "number of classes")
    flags.DEFINE_string("%slayer_dims" % prefix, layer_dims,
                        "dimensions of fully-connected layers")


def mlp(layer_dims, num_classes, use_bias=True,
        w_init=slim.initializers.variance_scaling_initializer(),
        activation_fn=tf.nn.relu, name='mlp'):
    @with_end_points
    def net(inputs, train=True):
        net = slim.flatten(inputs)
        bias_init = tf.zeros_initializer() if use_bias else None
        with slim.arg_scope([slim.fully_connected], activation_fn=activation_fn,
                            biases_initializer=bias_init, weights_initializer=w_init):
            for i, layer_size in enumerate(layer_dims):
                assert layer_size >= 0
                net = slim.fully_connected(net, layer_size, scope='fc%d' % i)
            logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='logits')
            return logits

    return tf.make_template(name, net)


def critic_mlp(layer_dims, num_classes, use_bias=True,
               w_init=slim.initializers.variance_scaling_initializer(),
               noise_data=0.1, noise_hidden=0.5,
               name='mlp'):
    @with_end_points
    def net(inputs, train=True):
        net = slim.flatten(inputs)
        net = gaussian_noise(net, noise_data)
        bias_init = tf.zeros_initializer() if use_bias else None
        with slim.arg_scope([slim.fully_connected], activation_fn=lrelu,
                            biases_initializer=bias_init, weights_initializer=w_init):
            for i, layer_size in enumerate(layer_dims):
                assert layer_size >= 0
                net = slim.fully_connected(net, layer_size, scope='fc%d' % i)
                net = gaussian_noise(net, noise_hidden)
            logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='logits')
            return logits

    return tf.make_template(name, net)


def lenet5(num_classes, activation_fn=tf.nn.relu,
           w_init=slim.initializers.variance_scaling_initializer(),
           name='lenet5'):
    @with_end_points
    def net(inputs, train=True):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=activation_fn,
                            weights_initializer=w_init,
                            padding='VALID'), \
             slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            data_format="NCHW"):
            net = slim.conv2d(inputs, 32, 5)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 64, 5)
            net = slim.max_pool2d(net, 2)
        net = slim.flatten(net)
        with slim.arg_scope([slim.fully_connected], activation_fn=activation_fn,
                            weights_initializer=w_init):
            net = slim.fully_connected(net, 512)
            logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
            return logits

    return tf.make_template(name, net)


def _get_activation_fn(act):
    if act == 'relu':
        return tf.nn.relu
    elif act == 'lrelu':
        return lrelu
    else:
        raise ValueError


def _get_w_init(w_init):
    if w_init == 'msra':
        # factor=2.0, mode='FAN_IN', uniform=False
        return slim.initializers.variance_scaling_initializer()
    elif w_init == 'glorot':
        # Glorot w_init
        return slim.initializers.variance_scaling_initializer(
            factor=1.0, mode='FAN_AVG', uniform=True)
    else:
        raise ValueError("Unknown w_init")


def create_model(FLAGS, prefix='', name='model'):
    w_init = getattr(FLAGS, '%sw_init' % prefix)
    act = getattr(FLAGS, '%sactivation_fn' % prefix)
    model = getattr(FLAGS, '%smodel' % prefix)
    num_classes = getattr(FLAGS, '%snum_classes' % prefix)
    activation_fn = _get_activation_fn(act)
    w_init = _get_w_init(w_init)

    if model == 'mlp':
        layer_dims = getattr(FLAGS, '%slayer_dims' % prefix)
        layer_dims = [int(dim) for dim in layer_dims.split("-")]
        return mlp(layer_dims=layer_dims, num_classes=num_classes,
                   activation_fn=activation_fn, w_init=w_init, name=name)
    elif model == 'critic_mlp':
        layer_dims = getattr(FLAGS, '%slayer_dims' % prefix)
        layer_dims = [int(dim) for dim in layer_dims.split("-")]
        return critic_mlp(layer_dims=layer_dims, num_classes=num_classes,
                          w_init=w_init, name=name)
    elif model == 'lenet5':
        return lenet5(num_classes=num_classes, activation_fn=activation_fn,
                      w_init=w_init, name=name)
    else:
        raise ValueError
