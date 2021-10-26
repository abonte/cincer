import os, requests
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from influence.influence.influence_model import InfluenceModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.utils import check_random_state, Bunch
import matplotlib.pyplot as plt
import argparse

from negsup.datasets import *
from negsup.models import *
from negsup.utils import *
from negsup.negotiation import *
from negsup.fisher import *

FEW = 100

ecefunc = tfp.stats.expected_calibration_error_quantiles
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def prf(p, phat):
    """Computes precision, recall, F1."""
    y, yhat = np.argmax(p, axis=1), np.argmax(phat, axis=1)
    pr, rc, f1, _ = prfs(y, yhat, average='weighted')
    # expected_calibration_error
    log_pred = tf.math.log(np.max(phat, axis=1))
    label = tf.cast(y == yhat, dtype=tf.bool)
    (ece, _, _, _, _, _,) = ecefunc(label, log_pred, num_buckets=3)

    return pr, rc, f1, ece.numpy()


# ===========================================================================


def sample_counterexamples(args, if_config):
    """Sample counter-examples using kNN, IF, and RIF."""
    rng = np.random.RandomState(args.seed)

    # Build a subsampled noisy dataset
    dataset = DATASETS[args.dataset]()
    subsample_train(dataset, args.p_known + args.max_iters, rng=rng)
    noisy_dataset, noisy, kn, indices = gen_run_data(dataset, args, rng=rng)[0]
    # noisy_dataset, clean, noisy = inject_noise(args, dataset, rng=rng)

    print(f'{len(kn + indices)} examples, {len(noisy)} are noisy')

    # Train model (or load model trained) on noisy dataset
    basename = _get_basename(args, model_only=True)
    # model_path = os.path.join('model-cache', basename)
    # model = make_or_load_model(model_path,
    #                            args.model,
    #                            noisy_dataset,
    #                            n_epochs=args.n_epochs,
    #                            from_logits=args.from_logits,
    #                            no_cache=args.no_cache)
    model = make_model(args.model,
                       noisy_dataset,
                       from_logits=args.from_logits)
    model.fit(noisy_dataset.X_tr[kn],
              noisy_dataset.y_tr[kn],
              epochs=args.n_epochs)

    y = np.argmax(dataset.y_tr, axis=1)
    y_noisy = np.argmax(noisy_dataset.y_tr, axis=1)
    X = tf.convert_to_tensor(dataset.X_tr)  # XXX work-around for memory leak
    phat = model.predict(X, batch_size=args.batch_size)
    yhat = np.argmax(phat, axis=1)

    rows = np.arange(len(y))
    margins = phat[rows, yhat] - phat[rows, y_noisy]

    uncertain_mistakes = [
                             i for i in np.argsort(margins)
                             if y_noisy[i] != yhat[i] and i not in kn
                         ][:FEW // 4]

    certain_mistakes = [
                           i for i in np.argsort(margins)
                           if y_noisy[i] != yhat[i] and i not in kn
                       ][-FEW // 4:]

    selected = uncertain_mistakes + certain_mistakes

    # Dump the images and their counter-examples using kNN, IF, IF+kNN
    basename = _get_basename(args)
    for t, i in enumerate(selected):
        fig, axes = plt.subplots(1, 4, figsize=(3.2 * 4, 2.4))

        labeli, labelhati = dataset.class_names[y[i]], dataset.class_names[yhat[i]]

        print(f'EX {i}, "{labeli}" predicted as "{labelhati}" ({margins[i]})')

        axes[0].imshow(dataset.X_tr[i], cmap=plt.get_cmap('gray'))
        axes[0].set_title(f'True label  "{labeli}"\n'
                          f'Annotated as "{dataset.class_names[y_noisy[i]]}"\n'
                          f'Predicted as "{labelhati}"', fontsize=20, pad=15)
        axes[0].axis('off')

        negotiators = ['top_fisher', 'nearest', 'if']
        names = ['CINCER', '1-NN', 'IF']
        for n, (negotiator, name) in enumerate(zip(negotiators, names)):
            print(f'{t}/{len(selected)} : running {negotiator}')

            j, _, _ = find_counterexample(model,
                                          noisy_dataset,
                                          kn, i,
                                          negotiator,
                                          if_config,
                                          rng=rng)

            assert j in kn and j != i

            labelj, labeltildej = dataset.class_names[y[j]], dataset.class_names[
                y_noisy[j]]

            print(
                f'{t}/{len(selected)} : EX {i}, {negotiator} picked {j}, annotatated "{labeltildej}" (actually "{labelj}")')

            axes[n + 1].imshow(dataset.X_tr[j], cmap=plt.get_cmap('gray'))
            axes[n + 1].set_title(f'True label "{labelj}"\n'
                                  f'Annotated as "{labeltildej}"', fontsize=20, pad=15)

            axes[n + 1].set_xlabel(name, fontsize=20, labelpad=15)
            axes[n + 1].tick_params(axis='both', which='both', bottom=False, left=False,
                                    right=False, top=False, labelleft=False,
                                    labelbottom=False)
            # axes[n + 1].axis('off')

        fig.savefig(os.path.join('images', basename + f'__{t}.png'),
                    bbox_inches='tight',
                    pad_inches=0.3)
        plt.close(fig)


# ===========================================================================


def _get_suspiciousness_aucs(args, if_config, rng=None):
    """Check whether margin and IFs spot noisy train examples."""
    rng = check_random_state(rng)

    # Build a subsampled noisy dataset
    dataset = DATASETS[args.dataset]()
    subsample_train(dataset, args.p_known, rng=rng)
    noisy_dataset, clean, noisy = inject_noise(args, dataset, rng=rng)

    kn = list(range(len(dataset.y_tr)))
    print(f'q1: {len(kn)} examples, {len(noisy)} are noisy')

    # Train model (or load model trained) on noisy dataset
    basename = _get_basename(args, model_only=True)
    model_path = os.path.join('model-cache', basename)
    model = make_or_load_model(model_path,
                               args.model,
                               noisy_dataset,
                               n_epochs=args.n_epochs,
                               from_logits=args.from_logits,
                               no_cache=args.no_cache)

    # Pick a subset of clean + noisy examples, up to $FEW each
    n_samples = min(len(clean), len(noisy), FEW)
    selected = np.concatenate([
        rng.permutation(clean)[:n_samples],
        rng.permutation(noisy)[:n_samples],
    ]).astype(int)
    is_mistake = np.concatenate([
        0 * np.ones(n_samples),  # clean
        1 * np.ones(n_samples),  # noisy
    ])
    assert len(selected) == len(is_mistake)
    assert len(selected) > 0

    n_labels = len(dataset.class_names)

    # Compute suspiciousness of selected examples & their AUC
    print(f'computing margin of {len(selected)} examples')
    margins = [
        get_margin(model,
                   noisy_dataset.X_tr,
                   noisy_dataset.y_tr,
                   i)
        for i in selected
    ]

    # print(f'computing exp. gradient len of {len(selected)} examples')
    # expgradlens = [
    #    get_expected_gradient_len(model,
    #                              noisy_dataset.X_tr,
    #                              noisy_dataset.y_tr,
    #                              i,
    #                              n_labels,
    #                              **if_config)
    #    for i in selected
    # ]

    print(f'computing Fisher kernel of {len(selected)} examples')
    fishervalues = [
        fisher_kernel(i, i,
                      model,
                      noisy_dataset.X_tr,
                      noisy_dataset.y_tr)
        for i in selected
    ]

    print(f'computing influence of {len(selected)} examples')
    influences = [
        get_influence_on_params(model,
                                noisy_dataset.X_tr,
                                noisy_dataset.y_tr,
                                kn, i,
                                **if_config)
        for i in selected
    ]

    m_auc = roc_auc_score(is_mistake, margins)
    g_auc = roc_auc_score(is_mistake, fishervalues)
    i_auc = None  # roc_auc_score(is_mistake, influences)
    return m_auc, g_auc, i_auc


def eval_identification(args, if_config):
    m_aucs, g_aucs, i_aucs = [], [], []
    for repeat in range(args.n_repeats):
        rng = np.random.RandomState(args.seed + repeat)
        m_auc, g_auc, i_auc = _get_suspiciousness_aucs(args, if_config, rng=rng)
        print(f'REP {repeat} AUCs: m={m_auc} g={g_auc} i={i_auc}')
        m_aucs.append(m_auc)
        g_aucs.append(g_auc)
        i_aucs.append(i_auc)

    i_aucs = np.array(i_aucs)
    g_aucs = np.array(g_aucs)
    m_aucs = np.array(m_aucs)

    print(f'AVG : ' \
          f'margin AUC={m_aucs.mean()}±{m_aucs.std()}, ' \
          f'fisher value AUC={g_aucs.mean()}±{g_aucs.std()}, ' \
          # f'IF AUC={i_aucs.mean()}±{i_aucs.std()}'
          )


# ===========================================================================


def _negotiate(model,
               initial_weights,
               dataset,
               noisy_dataset,
               noisy,
               kn,
               indices,
               threshold,
               if_config,
               args,
               return_suspiciousnesses=False,
               rng=None):
    """Run the negotiation loop and record various stats."""
    rng = check_random_state(rng)

    # ===== validate dataset ====
    noisy_in_experiment = sum(int(el in noisy) for el in kn + indices)
    if args.noise_type != 'random' and noisy_in_experiment == 0:
        raise RuntimeError("p_noise = 0.0 (for computing upper bound) for"
                           "no-random noise is not support,"
                           "because rng is called a different number of times respect to"
                           "other baselines.")

    expected_n_noisy_ex = args.p_noise * len(kn) if args.ce_precision \
        else args.p_noise * len(kn + indices)
    print(noisy_in_experiment, expected_n_noisy_ex)
    lim = 25 if args.dataset == '20ng' else 10
    assert expected_n_noisy_ex - lim < noisy_in_experiment < expected_n_noisy_ex + lim
    if args.inspector == 'margin' and not 0 < threshold < 1:
        raise RuntimeError('threshold is not between 0 and 1')
    if args.inspector == 'fisher' and not threshold > 1:
        raise RuntimeError('threshold is not above 1')
    # ===========

    print(f'NEGOTIATING: {noisy_in_experiment} '
          f'noisy, {len(kn)} kn, {len(indices)} iters '
          f' -- negotiator "{args.negotiator}" '
          f'noise_type "{args.noise_type}" '
          f'dataset "{args.dataset}"')

    # Evaluate model learned on initial known set
    X_ts = tf.convert_to_tensor(dataset.X_ts)  # XXX work-around for memory leak

    model.set_weights(initial_weights)

    baseline = 0.7 if args.model == 'logreg' else 0.9
    callback = EarlyStopping(monitor='acc', baseline=baseline, patience=5)

    model.fit(noisy_dataset.X_tr[kn],
              noisy_dataset.y_tr[kn],
              epochs=args.n_epochs,
              callbacks=[callback],
              verbose=0)

    radius = None
    if args.negotiator == 'nearest_fisher':
        path_dist = 'data/dist_examples_' + args.dataset + '.pickle'
        path_max_dist = 'data/max_dist_examples_' + args.dataset + '.pickle'
        if os.path.exists(path_max_dist):
            print('nearest-fisher: load example max dist')
            max_dist = load(path_max_dist)['max_dist']
        else:
            print('nearest-fisher: computing max dist')
            dist = pdist(
                noisy_dataset.X_tr.ravel().reshape(noisy_dataset.X_tr.shape[0], -1))
            max_dist = np.max(dist)
            dump(path_dist, {'dist': dist})
            dump(path_max_dist, {'max_dist': max_dist})
        assert 0 < args.nfisher_radius < 1
        radius = max_dist * args.nfisher_radius
        print(f'max distance {max_dist}, radius {radius}')

    # phat_ts = model.predict(X_ts, batch_size=args.batch_size)
    pr, rc, f1, ece = prf(noisy_dataset.y_ts, model.predict(noisy_dataset.X_ts))

    # Negotiate

    trace = pd.DataFrame()
    stat = Bunch(n_queried=0,
                 n_mistakes_seen=0,
                 n_cleaned=0,
                 n_cleaned_ce=0,
                 n_cleaned_ex=0,
                 precision=pr,
                 recall=rc,
                 f1=f1,
                 ece=ece,
                 zs_value=0,
                 noisy_ce=0,
                 suspiciousnesses=0,
                 case1=0,
                 case2=0,
                 case3=0,
                 case4=0,
                 case5=0,
                 case6=0,
                 case7=0,
                 case8=0,
                 case9=0,
                 case10=0,
                 case11=0,
                 case12=0,
                 case13=0,
                 case14=0,
                 ce_pr_at_5=np.nan,
                 ce_pr_at_10=np.nan,
                 ce_pr_at_25=np.nan)

    trace = trace.append(stat, ignore_index=True)

    for t, i in enumerate(indices):
        kn.append(i)

        if args.inspector == 'random':
            suspiciousness = None
            suspicious = rng.binomial(1, args.threshold)
        else:
            suspiciousness = get_suspiciousness(model,
                                                noisy_dataset.X_tr,
                                                noisy_dataset.y_tr,
                                                kn, i,
                                                len(noisy_dataset.class_names),
                                                args.inspector,
                                                **if_config)
            suspicious = suspiciousness > threshold
        stat.suspiciousnesses = suspiciousness

        print(f'{t:3d}/{len(indices):3d} : EX {i} '
              f'noisy={i in noisy} '
              f'suspicious={suspicious} ({suspiciousness} > {threshold})')

        stat.n_mistakes_seen += int(i in noisy)

        stat.ce_pr_at_5 = np.nan
        stat.ce_pr_at_10 = np.nan
        stat.ce_pr_at_25 = np.nan
        stat.noisy_ce, stat.zs_value = 0, 0

        if suspicious:

            in_shape = (1,) + dataset.X_tr.shape[1:]
            xi = dataset.X_tr[i].reshape(in_shape)
            phati = model.predict(xi)
            yhati = np.argmax(phati, axis=1)[0]

            candidates = []
            # Identify examples to be cleaned
            if args.no_ce or return_suspiciousnesses:
                if yhati != np.argmax(noisy_dataset.y_tr[i]):
                    stat.n_queried += int(suspicious)
                    candidates = [i]
            else:
                # user and machine don't agree, query the user
                if yhati != np.argmax(noisy_dataset.y_tr[i]):
                    stat.n_queried += int(suspicious)

                    print('ce start')
                    j, stat.zs_value, ordered_candidates = find_counterexample(
                        model,
                        noisy_dataset,
                        kn, i,
                        args.negotiator,
                        if_config,
                        radius,
                        rng=rng)
                    assert j in kn and j != i

                    if 'ce_removal' == args.negotiator:
                        candidates = [i]
                        if i not in noisy:
                            kn.remove(j)
                    else:
                        candidates = [i, j]
                    stat = _compute_stat(args, dataset, i, j, noisy,
                                         ordered_candidates, stat, t, yhati)
                    # if args.dataset == 'synthetic':
                    #     plot_synthetic_dataset(noisy_dataset.X_tr, noisy_dataset.y_tr, kn, i, j, t)

            mistakes = [c for c in candidates if c in noisy]
            print(f'        : EX/CE {candidates}, noisy{mistakes}')

            # Clean any mistakes on i and j

            for c in mistakes:
                assert (dataset.y_tr[c] != noisy_dataset.y_tr[c]).any()
                noisy_dataset.y_tr[c] = dataset.y_tr[c]
                noisy.remove(c)
                stat.n_cleaned += 1
                print('cleaned!')

        # Update the model
        if not args.ce_precision:
            if not args.no_reload:
                model.set_weights(initial_weights)
            model.fit(noisy_dataset.X_tr[kn],
                      noisy_dataset.y_tr[kn],
                      epochs=args.n_epochs,
                      callbacks=[callback],
                      verbose=0)
        phat_ts = model.predict(X_ts, batch_size=args.batch_size)
        stat.precision, stat.recall, stat.f1, stat.ece = prf(noisy_dataset.y_ts,
                                                             phat_ts)

        trace = trace.append(stat, ignore_index=True)

    if return_suspiciousnesses:
        return trace.suspiciousnesses.to_list()
    return trace


def _compute_stat(args, dataset, i, j, noisy, ordered_candidates, stat, t, yhati):
    if args.ce_precision:
        precisions = []
        for lim in [5, 10, 25]:
            prec = sum([int(ce in noisy) for ce in ordered_candidates[:lim]]) / lim
            precisions.append(prec)
        stat.ce_pr_at_5, stat.ce_pr_at_10, stat.ce_pr_at_25 = precisions

    stat.n_cleaned_ce += int(j in noisy)
    stat.n_cleaned_ex += int(i in noisy)
    user_correct = i not in noisy
    machine_corret = yhati == np.argmax(dataset.y_tr[i])
    ce_corret = j not in noisy
    stat.noisy_ce = t if not ce_corret else 0

    stat.case1 += int(
        not user_correct and not machine_corret and ce_corret)
    stat.case2 += int(not user_correct and machine_corret and ce_corret)
    stat.case3 += int(user_correct and not machine_corret and ce_corret)
    stat.case4 += int(
        not user_correct and not machine_corret and not ce_corret)
    stat.case5 += int(
        not user_correct and machine_corret and not ce_corret)
    stat.case6 += int(
        user_correct and not machine_corret and not ce_corret)
    stat.case7 += int(not machine_corret and not ce_corret)
    stat.case8 += int(not user_correct and not ce_corret)
    stat.case9 += int(machine_corret and not ce_corret)
    stat.case10 += int(user_correct and not ce_corret)
    stat.case11 += int(not machine_corret and ce_corret)
    stat.case12 += int(not user_correct and ce_corret)
    stat.case13 += int(machine_corret and ce_corret)
    stat.case14 += int(user_correct and ce_corret)
    return stat


def _plot_synthetic_dataset(X_tr, y_tr, kn, i, j, iteration):
    plt.figure()
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    c = np.array([color[x] for x in np.argmax(y_tr, axis=1)])
    plt.scatter(X_tr[kn, 0], X_tr[kn, 1], marker='x', c=c[kn])
    plt.scatter(X_tr[j, 0], X_tr[j, 1], label='ce', marker='o', c=c[j], s=60,
                edgecolors='red')
    plt.scatter(X_tr[i, 0], X_tr[i, 1], label='i', marker='s', c=c[i], s=60,
                edgecolors='red')
    plt.legend()
    plt.savefig(f'gif/{iteration}_gif.png')


def gen_run_data(dataset, args, rng=None):
    """Generate several noisy datasets and example sequences."""

    n_known = get_n_known(dataset, args.p_known)
    n_non_test = len(dataset.y_tr)

    run_data = []
    for _ in range(args.n_repeats):
        noisy_dataset, clean, noisy = inject_noise(args, dataset, rng=rng)

        if args.noise_type == 'outlier':
            n_non_test = len(noisy_dataset.y_tr)

        kn = list(sorted(rng.permutation(n_non_test)[:n_known]))
        tr = list(sorted(set(range(n_non_test)) - set(kn)))

        n_iters = min(args.max_iters, len(tr))
        indices = list(rng.permutation(tr)[:n_iters])

        if args.ce_precision:
            noisy_dataset.X_tr[indices] = dataset.X_tr[indices]
            noisy_dataset.y_tr[indices] = dataset.y_tr[indices]
            noisy = list(set(noisy) - set(indices))

        run_data.append((noisy_dataset, noisy, kn, indices))

    return run_data


def eval_negotiation(args, if_config):
    """Measures the impact of negotiation on learning."""
    rng = np.random.RandomState(args.seed)

    # Builds noisy datasets and example sequence for all the reruns
    print('builds datasets')
    dataset = DATASETS[args.dataset]()
    if args.noise_type == 'outlier':
        # reduce size of dataset to speed up computation
        n_ex = int(args.p_known + args.max_iters + 50)
        indices = list(rng.permutation(list(set(range(dataset.X_tr.shape[0]))))[:n_ex])
        dataset.X_tr, dataset.y_tr = dataset.X_tr[indices], dataset.y_tr[indices]

    run_data = gen_run_data(dataset, args, rng=rng)

    # Build and save the untrained model
    model = make_model(args.model, dataset)
    initial_weights = model.get_weights()

    # df = pd.DataFrame()
    # for iteration, (_, _, _, indices) in enumerate(run_data):
    #    df[str(iteration)] = indices

    # df.to_csv(f'results/indices{_get_basename(args)}.csv')

    print('Start runs')
    traces = [
        _negotiate(model,
                   initial_weights,
                   dataset,
                   noisy_dataset,
                   noisy,
                   kn,
                   indices,
                   args.threshold,
                   if_config,
                   args,
                   rng=rng)
        for noisy_dataset, noisy, kn, indices in run_data
    ]

    dump(os.path.join('results', _get_basename(args) + '.pickle'), {'args': args, 'traces': traces})


def find_threshold(args, if_config):
    """Finds a threshold that makes the inspector."""
    rng = np.random.RandomState(args.seed)

    # Load clean dataset
    dataset = DATASETS[args.dataset]()
    run_data = gen_run_data(dataset, args, rng=rng)

    # Build and save the untrained model
    model = make_model(args.model, dataset)
    initial_weights = model.get_weights()

    # Compute suspiciousnesses
    values = np.concatenate([
        _negotiate(model,
                   initial_weights,
                   dataset,
                   noisy_dataset,
                   noisy,
                   kn,
                   indices,
                   args.threshold,
                   if_config,
                   args,
                   return_suspiciousnesses=True,
                   rng=rng)
        for noisy_dataset, noisy, kn, indices in run_data
    ])

    # Look for a threshold that catches ~half the # of mislabeled examples
    n_noisy = int(len(values) * args.p_noise)
    ideal_n_of_queries = n_noisy // 2

    best_loss, best_value = np.inf, None
    for value in sorted(values):
        n_suspicious = len([v for v in values if v > value])
        loss = np.abs(ideal_n_of_queries - n_suspicious)
        if loss < best_loss:
            best_loss, best_value = loss, value

    print(f'threshold {best_value} ({best_loss})')


# ===========================================================================


def get_logreg_params(dataset, kn):
    """Fits logistic regression to a local optimum, helps with Hessian."""
    C = 1 / (len(kn) * 1)
    lr = LogisticRegression(penalty='l2',
                            tol=1e-8,
                            C=C,
                            solver='lbfgs',
                            max_iter=1000,
                            fit_intercept=False,
                            multi_class='auto',
                            # none for binary, multinomial otherwise
                            warm_start=False,
                            verbose=1)
    lr.fit(dataset.X_tr[kn].reshape(len(kn), -1),
           np.argmax(dataset.y_tr[kn], axis=1))

    w, b = lr.coef_, lr.intercept_
    if w.shape[0] == 1:  # binary classification
        w = np.concatenate([-w, w], axis=0)
        b = np.concatenate([-b, b], axis=0)
    return w.T, b


def eval_fisher_and_influence(args, if_config):
    """Measures the correlation between IF and parameter/output changes and between
    fisher kernel and output changes."""
    rng = np.random.RandomState(args.seed)

    # Build a subsampled noisy dataset
    dataset = DATASETS[args.dataset]()
    subsample_train(dataset, args.p_known, rng=rng)
    noisy_dataset, clean, noisy = inject_noise(args, dataset, rng=rng)

    kn = list(range(len(dataset.y_tr)))
    print(f'{len(kn)} examples, {len(noisy)} are noisy')

    # Build and train model
    model = make_model(args.model, dataset)
    model.fit(noisy_dataset.X_tr[kn],
              noisy_dataset.y_tr[kn],
              epochs=args.n_epochs,
              verbose=0)

    if args.model == 'logreg':
        w, b = get_logreg_params(noisy_dataset, kn)
        model.get_layer('hack').set_weights([w, b])

    # Measure difference between removing a point and IF at i_test
    y_hat = model.predict(dataset.X_ts)
    ee = np.where(np.argmax(y_hat, axis=1) != np.argmax(dataset.y_ts, axis=1))
    i_test = ee[0][0]

    print(i_test)

    # i_test = 0
    loss = model.loss(model(noisy_dataset.X_ts[None, i_test]),
                      noisy_dataset.y_ts[None, i_test])

    coords_inf = []
    coords_fisher = []
    corrds_fisher_mistake = []
    try:
        e = 0
        for i in rng.permutation(kn):
            if np.argmax(noisy_dataset.y_tr[i]) != np.argmax(y_hat[i_test]):
                continue
            if e == FEW:
                break
            print(e, FEW)
            e += 1
            # Compute loss at i_test after retraining
            kn_minus_i = list(sorted(set(kn) - {i}))
            model_minus_i = make_model(args.model, dataset)
            model_minus_i.fit(noisy_dataset.X_tr[kn_minus_i],
                              noisy_dataset.y_tr[kn_minus_i],
                              epochs=args.n_epochs,
                              verbose=0)

            if args.model == 'logreg':
                w, b = get_logreg_params(noisy_dataset, kn_minus_i)
                model_minus_i.get_layer('hack').set_weights([w, b])

            loss_i = model.loss(model_minus_i(noisy_dataset.X_ts[None, i_test]),
                                noisy_dataset.y_ts[None, i_test])

            # Approximate loss at i_test using IF
            if_model = InfluenceModel(model,
                                      noisy_dataset.X_tr,
                                      noisy_dataset.y_tr,
                                      noisy_dataset.X_ts,
                                      noisy_dataset.y_ts,
                                      model.loss,
                                      **if_config)
            if_loss = if_model.get_influence_on_loss(i, i_test, known=kn)

            coords_inf.append((loss_i - loss, if_loss))
            fisher = get_fisher_kernel_on_test_point(model, i, i_test, kn,
                                                     noisy_dataset.X_tr,
                                                     noisy_dataset.y_tr,
                                                     noisy_dataset.X_ts,
                                                     noisy_dataset.y_ts,
                                                     dataset.n_classes,
                                                     args.negotiator,
                                                     rng)
            if np.argmax(dataset.y_tr[i]) == np.argmax(noisy_dataset.y_tr[i]):
                coords_fisher.append((loss_i - loss, fisher))
            else:
                corrds_fisher_mistake.append((loss_i - loss, fisher))
    except KeyboardInterrupt:
        print('exit')
        pass

    coords_inf = np.array(coords_inf)
    coords_fisher = np.array(coords_fisher)
    coords_fisher_mistake = np.array(corrds_fisher_mistake)
    dump('results/' + _get_basename(args) + '.pickle',
         {'args': args, 'influence': coords_inf,
          'fisher': coords_fisher,
          'fisher_mst': coords_fisher_mistake})


# ===========================================================================


def _get_basename(args, model_only=False):
    fields_model = [
        (None, args.exp_name),
        (None, args.dataset),
        (None, args.model),
        (None, args.seed),
        ('p', args.p_noise),
        ('noise_type', args.noise_type),
        ('B', args.batch_size),
        ('E', args.n_epochs),
        ('logits', args.from_logits),
    ]

    fields_nomodel = [
        ('k', args.p_known),
        ('T', args.max_iters),
        ('R', args.n_repeats),
        ('t', args.threshold),
        ('I', args.inspector),
        ('N', args.negotiator),
        ('nold', args.no_reload),
        ('noce', args.no_ce),
        ('damp', args.if_damping),
        ('depth', args.lissa_depth),
        ('samples', args.lissa_samples),
        ('bits', args.bits),
        ('cepr', int(args.ce_precision)),
        ('fr', args.nfisher_radius)
    ]

    fields = fields_model if model_only else fields_model + fields_nomodel
    basename = '__'.join([name + '=' + str(value) if name else str(value)
                          for name, value in fields])
    return basename


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('exp_name', type=str, default=None)
    parser.add_argument('question', help='research question to be answered')
    parser.add_argument('dataset', choices=sorted(DATASETS.keys()),
                        help='name of the dataset')
    parser.add_argument('model', choices=sorted(MODELS.keys()), help='model to be used')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached model')
    parser.add_argument('--seed', type=int, default=1, help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-R', '--n-repeats', type=int, default=10,
                       help='# of times the experiment is repeated')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='# of interaction rounds')
    group.add_argument('-k', '--p-known', type=float, default=1,
                       help='Proportion or # of initially known training examples')
    group.add_argument('-p', '--p-noise', type=float, default=0, help='Noise rate')
    group.add_argument('--noise-type', type=str, default='random')
    group.add_argument('--ce-precision', action='store_true', default=False,
                       help='precision of fisher in finding counterexamples')
    group.add_argument('--bits', type=int, choices=[32, 64], default=32)

    group = parser.add_argument_group('Model')
    group.add_argument('-B', '--batch-size', type=int, default=1024, help='Batch size')
    group.add_argument('-E', '--n-epochs', type=int, default=10,
                       help='Number of epochs (passes through the dataset)')
    group.add_argument('--from-logits', action='store_true',
                       help='Use logit trick  *KILLS EXP ∇ LEN!*')

    group = parser.add_argument_group('Method')
    group.add_argument('-I', '--inspector', type=str, default='always',
                       help='inspector to be used')
    group.add_argument('-N', '--negotiator', type=str, default='random',
                       help='negotiator to be used')
    group.add_argument('-t', '--threshold', type=float, default=0,
                       help='Suspicion threshold')
    group.add_argument('--no-reload', action='store_true',
                       help='whether to reload the initial model in every iter')
    group.add_argument('--no-ce', action='store_true',
                       help='negotiates without counter-examples')
    group.add_argument('--nfisher-radius', type=float, default=0)

    group = parser.add_argument_group('Influence Functions')
    group.add_argument('--if-damping', type=float, default=0,
                       help='Hessian preconditioner')
    group.add_argument('--lissa-depth', type=int, default=1000,
                       help='LISSA recursion depth')
    group.add_argument('--lissa-samples', type=int, default=1,
                       help='LISSA recursion depth')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if_dtype = set_bits(args.bits)
    if_config = {
        'method': 'lissa',
        'damping': args.if_damping,
        'scaling': 1,  # no scaling
        'lissa_depth': args.lissa_depth,
        'lissa_samples': args.lissa_samples,
        'dtype': if_dtype,
    }

    # if args.question == 'q1':
    #    eval_identification(args, if_config)
    # elif args.question == 'q2':
    #    sample_counterexamples(args, if_config)
    # elif args.question == 'find-threshold':
    #     find_threshold(args, if_config)
    # elif args.question == 'eval-influence':
    #     eval_fisher_and_influence(args, if_config)
    # else:
    #    raise ValueError('invalid question')

    eval_negotiation(args, if_config)


if __name__ == '__main__':
    main()
