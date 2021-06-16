import numpy as np
import tensorflow as tf
from sklearn.neighbors import BallTree
from influence.influence.influence_model import InfluenceModel
from influence.influence.influence_with_s_test import InfluenceWithSTest
from sklearn.utils import check_random_state


from .fisher import *
from .fisher import _get_fi_vector


def get_margin(model, X, y, i):
    """Computes the margin of an example wrt a model."""
    x = X[i]
    phat = model.predict(x[None, :]) # out shape is (1, n_classes)
    yhat = np.argmax(phat)
    y = np.argmax(y[i])

    return phat[0, yhat] - phat[0, y]


def _get_gradient_of_loss(model, X, label, n_labels, i, return_prob=False):
    """Computes the length of the gradient of the loss w.r.t. the params."""
    p = np.zeros(n_labels)
    p[label] = 1

    with tf.GradientTape() as tape:
        phat = model(X[i][None, :])
        loss = model.loss(p, phat)

    gradient = tape.gradient(
        loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    flat_gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in gradient]
    )

    if return_prob:
        return flat_gradient, phat
    else:
        return flat_gradient


def get_expected_gradient_len(model, X, y, i, n_labels):
    """Returns the expected (squared) length of the gradient of the loss w.r.t.
    the params.

    See Huang et al., Active Learning for Speech Recognition: the Power of
    Gradients, 2016.
    """
    squared_norms = []
    for label in range(n_labels):
        flat_gradient = _get_gradient_of_loss(
            model,
            X,
            label,
            n_labels,
            i,
            return_prob=True
        )
        squared_norms.append(np.dot(flat_gradient, flat_gradient))

    return np.dot(phat, np.array(squared_norms))


def get_influence_on_params(model, X, y, kn, i, **kwargs):
    """Computes the influence of a TRAINING example on a model's params."""
    return_vector = kwargs.pop('return_vector', False)
    inf_model = InfluenceModel(model,
                               X, y,  # used for both H and ∇loss
                               [], [],  # unused
                               model.loss,
                               **kwargs)
    influence = inf_model.get_inverse_hvp_lissa(i, known=kn)
    if return_vector:
        return influence
    norm_influence = np.linalg.norm(influence)
    assert np.isfinite(norm_influence), \
        '‖H^(-1)∇ℓ‖ is inf/nan! Use --bits 64, increase epochs or if-damping, decrease lissa-depth'
    return norm_influence / len(kn)


def get_suspiciousness(model,
                       X_tr, y_tr,
                       kn,
                       i,
                       n_labels,
                       inspector,
                       **kwargs):
    """Computes the ``suspiciousness'' of an example.

    Arguments
    ---------
    model :
        The Model.
    X_tr, y_tr : ndarrays
        The dataset used for training the model.
    kn : list
        Indices of examples that the model knows within the training set.
    i : int
        The index of the target example within the target set.
    inspector : str, one of ['always', 'never', 'margin', 'influence']
        Method to be used.

    Return
    ------
    suspiciousness : float
        Measure of suspiciousness of the target example.
    """
    if inspector == 'always':
        return -np.inf
    elif inspector == 'never':
        return np.inf
    elif inspector == 'margin':
        return get_margin(model, X_tr, y_tr, i)
    elif inspector == 'gradient':
        return get_expected_gradient_len(model, X_tr, y_tr, i, n_labels)
    elif inspector == 'influence':
        return get_influence_on_params(model, X_tr, y_tr, kn, i, **kwargs)
    elif inspector == 'fisher':
        fi = _get_fi_vector(model, X_tr, y_tr, i)
        return np.dot(fi, fi)
    else:
        raise ValueError(f'invalid suspiciousness inspector "{inspector}"')


def find_counterexample(model,
                        dataset,
                        kn, i,
                        negotiator,
                        if_config,
                        radius=None,
                        rng=None):
    """Computes a counter-example to a given example."""
    rng = check_random_state(rng)
    in_shape = (1,) + dataset.X_tr.shape[1:]

    # Select (indices of) candidates with the same annotated label as the predicted label of example i
    xi = dataset.X_tr[i].reshape(in_shape)
    phati = model.predict(xi)
    yhati = np.argmax(phati, axis=1)[0]

    candidates = []
    for k in sorted(set(kn) - {i}):
        yk = np.argmax(dataset.y_tr[k])
        if yk == yhati:
            candidates.append(k)
    n_candidates = len(candidates)
    assert n_candidates > 0, 'this is annoying'

    if negotiator == 'random':
        return rng.choice(candidates), None, candidates

    elif negotiator == 'nearest':
        X_candidates = dataset.X_tr[candidates].reshape(n_candidates, -1)

        tree = BallTree(X_candidates)
        indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                           return_distance=False,
                                           k=min(len(X_candidates), 25))[0]
        return candidates[indices_in_candidates[0]], None, indices_in_candidates

    elif negotiator == 'if':
        inf_model = InfluenceWithSTest(model,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       model.loss,
                                       **if_config)
        influences = np.array([
            inf_model.get_influence_on_prediction(k, i, known=kn)
            for k in candidates
        ])

        if False:
            print('influences =', influences)
        assert np.isfinite(influences).all(), \
            'IF is inf/nan! Use --bits 64, increase epochs or if-damping, decrease lissa-depth'
        argsort = np.argsort(influences)[::-1]
        ordered_candidates = np.array(candidates)[argsort]
        return candidates[np.argmax(influences)], None, ordered_candidates

    elif negotiator == 'nearest-if':
        X_candidates = dataset.X_tr[candidates].reshape(n_candidates, -1)

        # Identify the 100 closes examples
        tree = BallTree(X_candidates)
        n_neighbors = min(100, len(candidates))
        indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                           return_distance=False,
                                           k=n_neighbors)[0]
        closest_candidates = [candidates[l] for l in indices_in_candidates]

        # Sort them by influence
        inf_model = InfluenceWithSTest(model,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       model.loss,
                                       **if_config)
        influences = np.array([
            inf_model.get_influence_on_prediction(k, i, known=kn)
            for k in closest_candidates
        ])
        if False:
            print('influences =', influences)
        assert np.isfinite(influences).all(), \
            'IF is inf/nan! Use --bits 64, increase epochs or if-damping, decrease lissa-depth'
        return closest_candidates[np.argmax(influences)], None

    elif negotiator == 'nearest_fisher':
        # Identify the neighbours within a radius r
        tree = BallTree(dataset.X_tr[candidates].reshape(len(candidates), -1))
        indices_in_candidates = tree.query_radius(dataset.X_tr[i].reshape(1, -1),
                                                  radius,
                                                  return_distance=False)[0]
        if len(indices_in_candidates)==0:
            indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                               return_distance=False,
                                               k=1)[0]

        closest_candidates = [candidates[i] for i in indices_in_candidates]

        score = score_counterexamples_with_fisher_kernel(
            model,
            dataset,
            kn,
            i,
            closest_candidates,
            'top_fisher',
            damping=if_config['damping'],
            rng=rng
        )

        return closest_candidates[np.argmax(score)], None, None

    elif 'fisher' in negotiator or 'ce_removal' == negotiator:
        neg = 'top_fisher' if 'ce_removal' == negotiator else negotiator

        score = score_counterexamples_with_fisher_kernel(
            model,
            dataset,
            kn,
            i,
            candidates,
            neg,
            damping=if_config['damping'],
            rng=rng
        )
        argsort = np.argsort(score)[::-1]
        ordered_candidates = np.array(candidates)[argsort]
        return candidates[np.argmax(score)], np.amax(score), ordered_candidates

    else:
        raise ValueError(f'invalid negotiator {negotiator}')


def _get_fi(example_idx, model, X_tr, y_tr, y_idx=None, return_prob=False):
    parameters = model.trainable_variables
    in_shape = (1,) + X_tr.shape[1:]
    with tf.GradientTape() as tape:
        predicted_label = model(X_tr[example_idx].reshape(in_shape))
        if y_idx is None:
            label_idx = tf.math.argmax(y_tr[example_idx])
        else:
            label_idx = tf.constant(y_idx)
        conditional_log_likelihood = tf.math.log(predicted_label[0, label_idx])

    gradient = tape.gradient(
        conditional_log_likelihood,
        parameters,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    flat_gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in gradient]
    )
    if return_prob:
        return flat_gradient, predicted_label[0, label_idx]
    else:
        return flat_gradient
