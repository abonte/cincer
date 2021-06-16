import gc
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state


def _tl_tonp(tl, dtype=None):
    """Converts a list of tensors to a list of numpy arrays."""
    return [np.array(t, dtype=dtype or t.dtype) for t in tl]


def _tl_copy(tl):
    """Copy a list of tensors."""
    return [
        np.array(t, copy=True)
        for t in tl
    ]


def _tl_add(tl1, tl2, alpha=1.0):
    """Summation of two tensor lists."""
    return [
        t1 + alpha * t2
        for t1, t2 in zip(tl1, tl2)
    ]


def _tl_sub(tl1, tl2, alpha=1.0):
    """Difference between two tensor lists."""
    return [
        t1 - alpha * t2
        for t1, t2 in zip(tl1, tl2)
    ]


def _tl_dot(tl1, tl2):
    """Dot-product between two tensor lists."""
    return np.sum([
        np.dot(
            np.reshape(t1, [-1]),
            np.reshape(t2, [-1])
        )
        for t1, t2 in zip(tl1, tl2)
    ])


def _tl_bmvp(tl1, tl2):
    """Block-wise matrix-vector(like) product between two tensor lists."""
    result = []
    for t1, t2 in zip(tl1, tl2):
        t = np.dot(t1, np.reshape(t2, [-1]))
        result.append(np.reshape(t, t2.shape))
    return result


def _cg(A_times, b, tol=1e-5, atol=1e-8):
    """Solve Ax = b using conjugate gradient."""

    min_loss = max([tol * _tl_dot(b, b), atol])

    x = [np.zeros_like(t) for t in b]  # candidate
    r = _tl_sub(b, A_times(x))  # residual
    p = _tl_copy(r)
    loss = _tl_dot(r, r)

    t = 0
    while True:
        A_times_p = A_times(p)

        alpha = loss / _tl_dot(p, A_times_p)
        x = _tl_add(x, p, alpha=alpha)
        r = _tl_sub(r, A_times_p, alpha=alpha)
        new_loss = _tl_dot(r, r)
        t += 1

        print('CG: loss =', new_loss)
        if new_loss ** 0.5 < min_loss:
            return x, t

        beta = new_loss / loss
        p = _tl_add(r, p, alpha=beta)
        loss = new_loss


def _get_fi_vector(model, X, y, i, label=None, return_prob=False, flatten=True):
    """Computes ∇θ log P(y=label else y_i | x_i)."""
    x = X[i]

    if label is None:
        label = tf.math.argmax(y[i])

    with tf.GradientTape() as tape:
        phat_label = model(x[None, :])[0, label]
        conditional_log_likelihood = tf.math.log(phat_label)

    gradient = tape.gradient(
        conditional_log_likelihood,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    if flatten:
        gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in gradient]
        )

    if return_prob:
        return gradient, phat_label
    else:
        return gradient


def _get_top_fi_vector(model, X, y, i, label=None, return_prob=False):
    top_layer = model.get_layer('hack')

    x = X[i]

    if label is None:
        label = np.argmax(y[i])

    with tf.GradientTape() as tape:
        phat_label = model(x[None, :])[0, label]
        conditional_log_likelihood = tf.math.log(phat_label)

    gradient = tape.gradient(
        conditional_log_likelihood,
        top_layer.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in gradient]
    )

    if return_prob:
        return gradient, phat_label
    else:
        return gradient


def _get_fim(model, kn, X, y, n_classes):
    """Computes the full FIM."""
    accum = None
    for i in kn:
        for label in range(n_classes):
            fi, p = _get_fi_vector(
                model,
                X,
                y,
                i,
                label=label,
                return_prob=True)

            fim_slice = np.outer(fi, fi) * p
            if accum is None:
                accum = fim_slice
            else:
                accum += fim_slice

    return accum / len(kn)


def _get_top_fim(model, kn, X, y, n_classes, rng=None):
    """Computes the full FIM for the top layer only."""
    rng = check_random_state(rng)

    accum = None
    for i in rng.permutation(kn)[:30]:
        for label in range(n_classes):
            top_fi, p = _get_top_fi_vector(
                model,
                X,
                y,
                i,
                label=label,
                return_prob=True
            )

            top_fim_slice = p * np.outer(top_fi, top_fi)

            if accum is None:
                accum = top_fim_slice
            else:
                accum += top_fim_slice

    return accum / 30


def _get_block_fim(model, kn, X, y, n_classes, n_samples=10, rng=None):
    """Computes the block-diagonal (layerwise) approximation to the FIM."""
    rng = check_random_state(rng)

    expectation = None
    for l in range(n_classes):
        for i in rng.permutation(kn)[:n_samples]:

            block_fi, p = _get_fi_vector(
                model,
                X,
                y,
                i,
                label=l,
                return_prob=True,
                flatten=False
            )

            block_fim_slice = []
            for t in block_fi:
                t = np.array(t, dtype=np.float32)
                t = np.reshape(t, [-1])
                T = np.outer(t, t)
                T *= np.array([(p / len(kn))], dtype=np.float32)
                block_fim_at_k.append(T)

            if expectation is None:
                expectation = block_fim_slice
            else:
                expectation = _tl_add(expectation, block_fim_slice)

        gc.collect()

    return expectation


def _get_inv_diagonal_fim(model, kn, X, y, n_classes):
    """Computes the inverse of the diagonal approximation of the FIM."""
    expectation = None
    for i in kn:
        for label in range(n_classes):
            fi, conditional_prob = _get_fi_vector(
                model,
                X,
                y,
                i,
                label=label,
                return_prob=True
            )
            vec = (fi ** -2) * conditional_prob
            if expectation is None:
                expectation = vec
            else:
                expectation += vec
    return expectation / len(kn)


def get_fisher_kernel_on_test_point(model, i, i_test, kn, X_tr, y_tr, X_ts, y_ts, n_classes, negotiator, rng):
    if negotiator == 'top_fisher':
        fi = _get_top_fi_vector(model, X_tr, y_tr, i)
        fi_test = _get_top_fi_vector(model, X_ts, y_ts, i_test)
        fim = _get_top_fim(model, kn, X_tr, y_tr, n_classes, rng=rng)

    elif negotiator == 'full_fisher':
        fi = _get_fi_vector(model, X_tr, y_tr, i)
        fi_test = _get_fi_vector(model, X_ts, y_ts, i_test)
        fim = _get_fim(model, kn, X_tr, y_tr, n_classes)

    elif negotiator == 'practical_fisher':
        fi = _get_fi_vector(model, X_tr, y_tr, i)
        fi_test = _get_fi_vector(model, X_ts, y_ts, i_test)
        return np.dot(fi_test, fi)

    else:
        raise ValueError(negotiator)


    fim_inv = np.linalg.pinv(fim, hermitian=True)

    # try:
    #     fim_inv = np.linalg.inv(fim)
    # except np.linalg.LinAlgError:
    #     print('LinAlgError')
    #     fim = fim * (np.eye(len(fim)) + 1e-8)
    #     try:
    #         fim_inv = np.linalg.inv(fim)
    #     except:
    #         print('LinAlgError --> 0')
    #         return 0

    # ki_itest = fi.T.dot(fim_inv).dot(fi_test)[0, 0]

    ki_itest = np.dot(fim_inv, fi_test).dot(fi)
    #assert np.allclose(ki_itest, _fisher_kernel(fi_test, fi, fim))

    return ki_itest


def score_counterexamples_with_fisher_kernel(
        model,
        dataset,
        kn,
        i,
        candidates,
        negotiator,
        damping=0.0,
        rng=None
):
    X, y, n_classes = dataset.X_tr, dataset.y_tr, dataset.n_classes

    if negotiator == 'practical_fisher':
        fi = _get_fi_vector(model, X, y, i)

        kernel = [
            np.dot(fi, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'approx_fisher':
        fi = _get_fi_vector(model, X, y, i)

        inv_diag_fim = _get_inv_diagonal_fim(model, kn, X, y, n_classes)
        fi_times_inv_fim = inv_diag_fim * fi

        kernel = [
            np.dot(fi_times_inv_fim, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'top_fisher':
        top_fi = _get_top_fi_vector(model, X, y, i)

        # NOTE use Moore-Penrose pseudo-inverse to avoid issues with singular matrices
        top_fim = _get_top_fim(model, kn, X, y, n_classes, rng=rng)
        inv_top_fim = np.linalg.pinv(top_fim, hermitian=True)

        top_fi_times_inv_top_fim = np.dot(inv_top_fim, top_fi)

        kernel = [
            np.dot(
                top_fi_times_inv_top_fim,
                _get_top_fi_vector(model, X, y, j)
            )
            for j in candidates
        ]

    elif negotiator == 'full_fisher':
        fi = _get_fi_vector(model, X, y, i)

        fim = _get_fim(model, kn, X, y, n_classes)
        inv_fim = np.linalg.pinv(fim, hermitian=True)

        fi_times_inv_fim = np.dot(inv_fim, fi)

        kernel = [
            np.dot(fi_times_inv_fim, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'block_fisher':
        fi = _get_fi_vector(model, X, y, i, flatten=False)
        fi = _tl_tonp(fi, dtype=np.float32)

        print('computing block FIM')
        block_fim = _get_block_fim(model, kn, X, y, n_classes, rng=rng)

        preconditioned_block_fim = [
            np.array(t) + damping * np.eye(len(t), dtype=t.dtype)
            for t in block_fim
        ]

        print('inverting block FIM')
        if False:
            fi_times_inv_fim = _cg(
                lambda v: _tl_bmvp(preconditioned_block_fim, v),
                fi,
                tol=1e-5,
                atol=1e-8
            )

        else:
            inv_block_fim = [
                np.linalg.inv(t).astype(np.float32)
                for t in preconditioned_block_fim
            ]
            fi_times_inv_fim = _tl_bmvp(inv_block_fim, fi)

        # Computes the (block-wise) Fisher kernel
        print('scoring candidates')
        kernel = [
            _tl_dot(
                fi_times_inv_fim,
                _tl_tonp(
                    _get_fi_vector(model, X, y, j, flatten=False),
                    dtype=np.float32
                )
            )
            for j in candidates
        ]
    else:
        raise ValueError(negotiator)
    return kernel
