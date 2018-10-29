from __future__ import absolute_import, division, print_function

import glob
import importlib
import inspect
import os

import decorator
import numpy as np
import six

import tensorflow as tf

flags = tf.flags

flags.DEFINE_boolean("train", True, "train models")
flags.DEFINE_boolean("carlini", False, "test models using Carlini l2-attack")

FLAGS = tf.app.flags.FLAGS

@decorator.decorator
def concat_commands(f, *args, **kwargs):
    commands = f(*args, **kwargs)
    print("\n".join(commands))


def get_tmpl_str(f, script_name, add_args=None, exclude_args=None):
    """
    script_name: python module where flags are defined
    add_args: additional flags which are not included in the function definition.
    exclude_args: flags to exclude from function signature
    """
    if add_args is None:
        add_args = []
    if exclude_args is None:
        exclude_args = []
    argspec = inspect.getargspec(f)
    arg_names = argspec.args
    if len(add_args) > 0:
        for arg in add_args:
            if isinstance(arg, (tuple, list)) > 0:
                arg_names.insert(arg[0], arg[1])
            else:
                arg_names.append(arg)

    script_module = importlib.import_module(script_name)
    defined_flags = script_module.FLAGS._flags().keys()
    str_bfr = six.StringIO()
    str_bfr.write("python %(script_name)s.py " % locals())
    for arg_name in arg_names:
        if arg_name not in exclude_args:
            assert arg_name in defined_flags, arg_name
            str_bfr.write("--%(arg_name)s=%%(%(arg_name)s)s " % locals())
    tmpl_str = str_bfr.getvalue()[:-1]
    return tmpl_str


@concat_commands
def generate_critic(name='critic',
                    model='mlp', layer_dims="1200-1200-1200", activation_fn='relu',
                    critic_model='mlp', critic_layer_dims="1200-1200",
                    pretrain_niter=0, niter=100,
                    lmbd=1.0, lmbd_rec_l1=0, lmbd_rec_l2=0, lmbd_grad=10, weight_decay=0.0,
                    lr=0.0005, lr_decay_factor=0.5, lr_decay_step=40,
                    critic_lr=0.001, critic_lr_decay_factor=0.5, critic_lr_decay_step=40,
                    attack_iter=50, attack_overshoot=0.02,
                    attack_confidence="same", val_attack_confidence=0.8,
                    attack_random=True,
                    attack_uniform=False,
                    attack_label_smoothing=0.0,
                    train_dir="runs_%(model)s",
                    seed=1,
                    gpu_memory=1.0,
                    runs=10):
    np.random.seed(seed)
    tmpl_str = get_tmpl_str(
        generate_critic, 'train_critic', exclude_args=['runs'])
    name = name % locals()
    train_dir = train_dir % locals()
    for i in range(runs):
        seed = np.random.randint(1234)
        yield tmpl_str % locals()


@concat_commands
def generate_test(root_dir,
                  working_dir=None,
                  attack_iter=50,
                  attack_clip=0.5,
                  attack_box_clip=False,
                  attack_overshoot=0.02,
                  hc_confidence=0.8,
                  dataset="test",
                  sort_labels=True,
                  filter_dirs=False):
    tmpl_str = get_tmpl_str(
        generate_test,
        'test',
        add_args=[(0, 'load_dir')],
        exclude_args=['root_dir', 'filter_dirs'])
    working_dirs = glob.glob(working_dir + '/*')
    for load_dir in sorted(glob.glob(root_dir)):
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(load_dir, os.pardir))
            working_dir = working_dir.replace("runs", "test")
        else:
            working_dir = working_dir % locals()
        working_dirs = [
            os.path.basename(working_path)
            for working_path in glob.glob(os.path.join(working_dir, '*'))
        ]
        if not filter_dirs or os.path.basename(load_dir) not in working_dirs:
            yield tmpl_str % locals()


@concat_commands
def generate_test_carlini(root_dir,
                          working_dir=None,
                          num_examples=10000,
                          batch_size=100,
                          carlini_batch_size=100,
                          carlini_max_iter=10000,
                          carlini_confidence=0,
                          carlini_binary_steps=9,
                          generate_summary=True,
                          sort_labels=True,
                          dataset="test",
                          filter_dirs=False):
    tmpl_str = get_tmpl_str(
        generate_test_carlini,
        'test_carlini',
        add_args=[(0, 'load_dir')],
        exclude_args=['root_dir', 'filter_dirs'])
    working_dirs = glob.glob(working_dir + '/*')
    for load_dir in sorted(glob.glob(root_dir)):
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(load_dir, os.pardir))
            working_dir = working_dir.replace("runs", "test") + '_ca'
        working_dirs = [
            os.path.basename(working_path)
            for working_path in glob.glob(os.path.join(working_dir, '*'))
        ]
        if not filter_dirs or os.path.basename(load_dir) not in working_dirs:
            yield tmpl_str % locals()


if __name__ == '__main__':
    layer_dims = '1200-1200-1200'
    critic_layer_dims = '1200-1200'

    runs = 10
    if FLAGS.train:
        generate_critic(pretrain_niter=1, niter=100,
                        model='mlp', critic_model='critic_mlp', activation_fn='relu',
                        name='critic', layer_dims=layer_dims, critic_layer_dims=critic_layer_dims,
                        lmbd=0.5, lmbd_grad=10.0, lmbd_rec_l1=0.0, lmbd_rec_l2=0.01,
                        lr=0.0005, lr_decay_step=40, lr_decay_factor=0.5,
                        critic_lr=0.001, critic_lr_decay_step=40, critic_lr_decay_factor=0.5,
                        attack_confidence="class_running_mean",
                        train_dir="runs_mlp",
                        attack_random=False, attack_uniform=False, attack_label_smoothing=0.0,
                        attack_overshoot=0.05, runs=runs, attack_iter=5, gpu_memory=0.8)

        generate_critic(pretrain_niter=1, niter=100,
                        model='lenet5', critic_model='critic_mlp', activation_fn='relu',
                        name='critic', layer_dims=layer_dims, critic_layer_dims=critic_layer_dims,
                        lmbd=0.1, lmbd_grad=10.0, lmbd_rec_l1=0.0, lmbd_rec_l2=0.01,
                        lr=0.0005, lr_decay_step=40, lr_decay_factor=0.5,
                        critic_lr=0.001, critic_lr_decay_step=40, critic_lr_decay_factor=0.5,
                        attack_confidence="class_running_mean",
                        train_dir="runs_lenet5",
                        attack_random=False, attack_uniform=False, attack_label_smoothing=0.0,
                        attack_overshoot=0.05, runs=runs, attack_iter=5, gpu_memory=0.8)
    else:
        if not FLAGS.carlini:
            generate_test('./runs_mlp/*', attack_iter=500,
                          attack_clip=0.1, dataset="test",
                          working_dir="test_mlp", attack_box_clip=True,
                          hc_confidence="same")

            generate_test('./runs_lenet5/*', attack_iter=500,
                          attack_clip=0.1, dataset="test",
                          working_dir="test_lenet5", attack_box_clip=True,
                          hc_confidence="same")
        else:
            generate_test_carlini('./runs_mlp/*', working_dir='test_mlp_ca',
                                  carlini_max_iter=1000)

            generate_test_carlini('./runs_lenet5/*', working_dir='test_lenet5_ca',
                                  carlini_max_iter=1000)
