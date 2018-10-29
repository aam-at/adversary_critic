from __future__ import absolute_import, division, print_function


import numpy as np
import tensorflow as tf

from utils import jacobian, prediction, random_targets


def find_next_target(inputs, logits, labels, random=False, uniform=False,
                     label_smoothing=0.0, attack_topk=None, ord=2):
    """Find closest decision boundary as in Deepfool algorithm"""
    ndims = inputs.get_shape().ndims
    batch_size = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]
    batch_indices = tf.range(batch_size)
    labels_idx = batch_indices * num_classes + labels
    if not random:
        logits_flt = tf.reshape(logits, (-1,))
        logits_labels = tf.expand_dims(tf.gather(logits_flt, labels_idx), 1)
        grad_labels = tf.gradients(logits_labels, inputs)[0]

        if attack_topk is not None:
            topk_logits, topk_indices = tf.nn.top_k(logits, k=attack_topk)
            topk_jac = jacobian(topk_logits, inputs)

            f = topk_logits - logits_labels
            w = topk_jac - tf.expand_dims(grad_labels, 1)
        else:
            jac = jacobian(logits, inputs)
            f = logits - logits_labels
            w = jac - tf.expand_dims(grad_labels, 1)

        reduce_ind = list(range(2, ndims + 1))
        if ord == 2:
            dist = tf.abs(f) / tf.sqrt(tf.reduce_sum(w**2, axis=reduce_ind))
        else:
            dist = tf.abs(f) / tf.reduce_sum(tf.abs(w), axis=reduce_ind)
        if attack_topk is not None:
            labels_tile = tf.expand_dims(labels, 1)
            topk_labels_onehot = tf.equal(
                topk_indices, tf.tile(labels_tile, [1, attack_topk]))
            dist = tf.where(topk_labels_onehot, np.inf * tf.ones_like(dist), dist)
        else:
            labels_onehot = tf.cast(
                tf.one_hot(labels, num_classes, dtype=tf.int32), tf.bool)
            dist = tf.where(labels_onehot, np.inf * tf.ones_like(dist), dist)

        l = tf.cast(tf.argmin(dist, axis=1), batch_indices.dtype)
        assert 0 <= label_smoothing < 1
        if label_smoothing > 0.0:
            l_onehot = tf.one_hot(l, num_classes, dtype=tf.float32)
            labels_onehot = tf.one_hot(labels, num_classes, dtype=tf.float32)
            p = (1 - label_smoothing) * l_onehot + (1.0 - l_onehot - labels_onehot) * label_smoothing / tf.cast((num_classes - 2), tf.float32)
            log_p = tf.log(p)
            l = tf.cast(tf.multinomial(log_p, 1), batch_indices.dtype)
            l = tf.reshape(l, (-1, ))

        if attack_topk is not None:
            topk_indices_flt = tf.reshape(topk_indices, (-1,))
            l_indices = batch_indices * attack_topk + l
            targets = tf.gather(topk_indices_flt, l_indices)
        else:
            targets = l
    else:
        targets = random_targets(logits, labels, uniform=uniform)
    return targets


def deepfool(model, x, labels=None, targets=None, ord=2, max_iter=25,
             clip_dist=None, over_shoot=0.02, boxmin=None, boxmax=None,
             epsilon=1e-4):
    """Tensorflow implementation of DeepFool https://arxiv.org/abs/1511.04599
    """
    ndims = x.get_shape().ndims
    batch_size = tf.shape(x)[0]
    batch_indices = tf.range(batch_size)
    num_classes = int(model(x).get_shape()[1])

    if labels is None:
        labels = prediction(model(x))

    labels_idx = batch_indices * num_classes + labels

    def should_continue(cond, i, x, r):
        return tf.logical_and(cond, tf.less(i, max_iter))

    def update_r(cond, i, x, r):
        x_adv = x + r
        logits = model(x_adv)
        x_adv_os = x + (1 + over_shoot) * r
        logits_os = model(x_adv_os)
        pred = prediction(logits_os)

        # closest boundary
        if targets is None:
            l = find_next_target(x, logits, labels, random=False)
        else:
            l = targets
        l_idx = batch_indices * num_classes + l

        logits_flt = tf.reshape(logits, (-1,))
        logits_labels = tf.gather(logits_flt, labels_idx)
        logits_targets = tf.gather(logits_flt, l_idx)

        f = logits_targets - logits_labels
        w = tf.gradients(f, x_adv)[0]
        reduce_ind = list(range(1, ndims))
        w2_norm = tf.sqrt(tf.reduce_sum(w ** 2, axis=reduce_ind))
        if ord == 2:
            dist = tf.abs(f) / w2_norm
        else:
            dist = tf.abs(f) / tf.reduce_sum(tf.abs(w), axis=reduce_ind)
        # avoid numerical instability and clip max value
        if clip_dist is not None:
            dist = tf.clip_by_value(dist, 0, clip_dist)
        if ord == 2:
            r_upd = w * tf.reshape(((dist + epsilon) / w2_norm), (-1,) + (1,) * (ndims - 1))
        else:
            r_upd = tf.sign(w) * tf.reshape(dist, (-1, ) + (1,) * (ndims - 1))

        # select and update
        is_mistake = tf.not_equal(labels, pred)
        # if targets is provides and it is equal to the class label
        target_is_label = tf.equal(labels, l)
        selector = tf.logical_or(is_mistake, target_is_label)
        r_new = tf.where(selector, r, r + r_upd)
        if boxmin is not None and boxmax is not None:
            x_adv_new = x + (1 + over_shoot) * r_new
            r_new = (tf.clip_by_value(x_adv_new, boxmin, boxmax) - x) / (1 + over_shoot)
        cond = tf.logical_not(tf.reduce_all(selector))
        return cond, i + 1, x, r_new

    cond = tf.constant(True, tf.bool)
    i = tf.constant(0)
    r0 = tf.zeros_like(x)
    r = tf.while_loop(should_continue, update_r,
                      [cond, i, x, r0],
                      back_prop=False)[-1]
    x_adv = tf.stop_gradient(x + (1 + over_shoot) * r)
    return x_adv


def high_confidence_attack(model, x, labels=None, targets=None, random=False,
                           uniform=False, max_iter=25, attack_topk=None,
                           over_shoot=0.02, confidence=0.8, clip_dist=None,
                           epsilon=1e-8, boxmin=None, boxmax=None):
    epsilonsqrt = np.sqrt(epsilon)
    ndims = x.get_shape().ndims
    reduce_ind = list(range(1, ndims))
    batch_size = tf.shape(x)[0]
    logits = model(x)
    num_classes = int(logits.get_shape()[1])

    if labels is None:
        labels = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    if targets is None:
        targets = find_next_target(x, logits, labels, random=random,
                                   uniform=uniform, attack_topk=attack_topk)
    targets_onehot = tf.one_hot(targets, depth=num_classes)

    if callable(confidence):
        confidence = confidence(targets_onehot)

    def should_continue(cond, i, x, r):
        return tf.logical_and(cond, tf.less(i, max_iter))

    def update_r(cond, i, x, r):
        x_adv = x + (1 + over_shoot) * r
        logits = model(x_adv)
        prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)

        prob_t = tf.reduce_sum(prob * targets_onehot, axis=1)
        log_prob_t = tf.reduce_sum(log_prob * targets_onehot, axis=1)

        f = tf.abs(log_prob_t - tf.log(confidence))
        f = tf.reshape(f, (-1,) + (1,) * (ndims - 1))
        w = tf.gradients(log_prob_t, x_adv)[0]

        w_norm = tf.sqrt(epsilon + tf.reduce_sum(w ** 2, axis=reduce_ind, keepdims=True))
        r_upd = (epsilonsqrt + f / w_norm) * w / w_norm
        if clip_dist is not None:
            r_upd = tf.clip_by_norm(r_upd, clip_dist, axes=reduce_ind)

        # select and update
        is_high_confidence = tf.greater_equal(prob_t, confidence)
        is_target_hit = tf.equal(prediction(logits), targets)
        target_is_label = tf.equal(labels, targets)
        selector = tf.logical_or(tf.logical_and(is_target_hit, is_high_confidence),
                                 target_is_label)
        r_new = tf.where(selector, r, r + r_upd)
        cond = tf.logical_not(tf.reduce_all(selector))
        if boxmin is not None and boxmax is not None:
            x_adv_new = x + (1 + over_shoot) * r_new
            r_new = (tf.clip_by_value(x_adv_new, boxmin, boxmax) - x) / (1 + over_shoot)
        return cond, i + 1, x, r_new

    cond = tf.constant(True, tf.bool)
    i = tf.constant(0)
    r0 = tf.zeros_like(x)
    # r0 = tf.random_normal_initializer(0, 0.1)(tf.shape(x))
    r = tf.while_loop(
        should_continue, update_r,
        [cond, i, x, r0],
        back_prop=True)[-1]
    x_adv = tf.stop_gradient(x + (1 + over_shoot) * r)
    return x_adv


def high_confidence_attack_unrolled(model, x, labels=None, targets=None, max_iter=5,
                                    over_shoot=0.05, confidence=0.8, epsilon=1e-8,
                                    attack_random=False, attack_uniform=False,
                                    attack_label_smoothing=0.0,
                                    ord=2):
    epsilonsqrt = np.sqrt(epsilon)
    ndims = x.get_shape().ndims
    reduce_ind = list(range(1, ndims))
    batch_size = tf.shape(x)[0]
    logits = model(x)
    num_classes = int(logits.get_shape()[1])

    if labels is None:
        labels = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    if targets is None:
        targets = find_next_target(x, logits, labels, random=attack_random,
                                   uniform=attack_uniform,
                                   label_smoothing=attack_label_smoothing,
                                   ord=ord)
    targets_onehot = tf.one_hot(targets, depth=num_classes)

    if callable(confidence):
        confidence = confidence(targets_onehot)

    r = tf.zeros_like(x)
    should_continue = tf.ones((batch_size, ))
    for i in range(max_iter):
        x_adv = x + (1 + over_shoot) * r
        x_adv = tf.stop_gradient(x_adv)
        logits = model(x_adv)
        prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)

        prob_t = tf.reduce_sum(prob * targets_onehot, axis=1)
        log_prob_t = tf.reduce_sum(log_prob * targets_onehot, axis=1)
        # indicator to include this perturbation into forward pass
        with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
            should_continue_i = tf.round(tf.sigmoid(confidence - prob_t))
            should_continue *= should_continue_i

        f = tf.abs(log_prob_t - tf.log(confidence))
        f = tf.reshape(f, (-1,) + (1,) * (ndims - 1))
        should_continue_br = tf.reshape(should_continue, (-1,) + (1,) * (ndims - 1))
        w = tf.gradients(log_prob_t, x_adv)[0]
        if ord == 2:
            w_norm = tf.sqrt(epsilon + tf.reduce_sum(w ** 2, axis=reduce_ind, keepdims=True))
            # compute perturbation for iteration i
            # add epsilon for stability
            r_i = (epsilonsqrt + f / w_norm) * w / w_norm
        else:
            w_norm = tf.reduce_sum(tf.abs(w), axis=reduce_ind, keepdims=True)
            r_i = (epsilonsqrt + f) * tf.sign(w) / w_norm
        r += should_continue_br * r_i

    x_adv = x + (1 + over_shoot) * r
    return x_adv
