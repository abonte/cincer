import os, requests
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from copy import copy

from matplotlib import pyplot as plt
from sklearn.neighbors import BallTree
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_random_state

from .models import make_model
from .utils import load, dump


def _download(path, urls):
    if not os.path.exists(path):
        os.mkdir(path)
    for url in urls:
        filename = os.path.join(path, os.path.basename(url))
        if not os.path.exists(filename):
            print(f'downloading {filename} from {url} ...')
            data = requests.get(url).content
            with open(filename, 'wb') as fp:
                fp.write(data)


def _df_to_instances(df, numerical_columns, categorical_columns):
    X = []

    for column in numerical_columns:
        values = df[column].to_numpy()
        values = MinMaxScaler(copy=False).fit_transform(values.reshape(-1, 1))
        X.append(values)

    for column in categorical_columns:
        categories = list(sorted(df[column].value_counts().index.tolist()))
        values = (df[column]
                  .replace({cat: idx for idx, cat in enumerate(categories)})
                  .to_numpy())
        imputer = SimpleImputer(strategy='most_frequent')
        values = imputer.fit_transform(values.reshape(-1, 1)).ravel()
        encoder = OneHotEncoder(drop='if_binary', sparse=False, dtype=int)
        values = encoder.fit_transform(values.reshape(-1, 1))
        X.append(values)

    return np.concatenate(X, axis=1)


def load_synthetic():
    X, y = make_classification(n_samples=1000,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=2,
                               flip_y=0,
                               random_state=1)

    X, y = make_blobs(n_samples=1000,
                      n_features=2,
                      centers=np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]]),
                      cluster_std=2,
                      shuffle=True,
                      random_state=1)

    X_tr, X_ts, y_tr, y_ts = train_test_split(X,
                                              y,
                                              test_size=0.2,
                                              random_state=1)
    return Bunch(X_tr=X_tr,
                 X_ts=X_ts,
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=4,
                 class_names=['red', 'blue', 'green', 'yellow']
                 )


def load_iris_():
    dataset = load_iris()
    X_tr, X_ts, y_tr, y_ts = train_test_split(dataset.data,
                                              dataset.target,
                                              test_size=0.2,
                                              random_state=1)
    scaler = MinMaxScaler(copy=False).fit(X_tr)
    return Bunch(X_tr=scaler.transform(X_tr),
                 X_ts=scaler.transform(X_ts),
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=len(dataset.target_names),
                 class_names=dataset.target_names)


def load_breast():
    dataset = load_breast_cancer()
    X_tr, X_ts, y_tr, y_ts = train_test_split(dataset.data,
                                              dataset.target,
                                              test_size=0.2,
                                              random_state=1)
    scaler = MinMaxScaler(copy=False).fit(X_tr)
    return Bunch(X_tr=scaler.transform(X_tr),
                 X_ts=scaler.transform(X_ts),
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=len(dataset.target_names),
                 class_names=dataset.target_names)


def load_wine_():
    dataset = load_wine()
    X_tr, X_ts, y_tr, y_ts = train_test_split(dataset.data,
                                              dataset.target,
                                              test_size=0.2,
                                              random_state=1)
    scaler = MinMaxScaler(copy=False).fit(X_tr)
    return Bunch(X_tr=scaler.transform(X_tr),
                 X_ts=scaler.transform(X_ts),
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=len(dataset.target_names),
                 class_names=dataset.target_names)


def load_adult():
    urls = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    ]
    _download('data', urls)

    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]

    df_tr = pd.read_csv(os.path.join('data', 'adult.data'),
                        names=columns,
                        sep=' *, *',
                        na_values='?',
                        engine='python')
    df_ts = pd.read_csv(os.path.join('data', 'adult.test'),
                        names=columns,
                        sep=' *, *',
                        skiprows=1,
                        na_values='?',
                        engine='python')
    df = pd.concat([df_tr, df_ts])

    label_to_int = {'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1}
    y = df['income'].replace(label_to_int).to_numpy()
    df = df.drop('income', axis=1)

    X = _df_to_instances(df,
                         list(df.select_dtypes(include=['int64'])),
                         list(df.select_dtypes(exclude=['int64'])))

    n_tr = len(df_tr)
    X_tr = X[:n_tr, :]
    X_ts = X[n_tr:, :]

    scaler = MinMaxScaler(copy=False).fit(X_tr)
    return Bunch(X_tr=scaler.transform(X_tr),
                 X_ts=scaler.transform(X_ts),
                 y_tr=keras.utils.to_categorical(y[:n_tr]),
                 y_ts=keras.utils.to_categorical(y[n_tr:]),
                 n_classes=2,
                 class_names=['<=50K', '>50K'])


def load_german():
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    ]
    _download('data', urls)

    columns = [
        'existingchecking', 'duration', 'credithistory', 'purpose',
        'creditamount', 'savings', 'employmentsince', 'installmentrate',
        'statussex', 'otherdebtors', 'residencesince', 'property', 'age',
        'otherinstallmentplans', 'housing', 'existingcredits', 'job',
        'peopleliable', 'telephone', 'foreignworker', 'classification'
    ]

    df = pd.read_csv(os.path.join('data', 'german.data'),
                     names=columns,
                     delimiter=' ',
                     skiprows=1)

    y = df['classification'].map({1: 0, 2: 1}).to_numpy()
    df = df.drop('classification', axis=1)

    X = _df_to_instances(df,
                         list(df.select_dtypes(include=['int64'])),
                         list(df.select_dtypes(exclude=['int64'])))

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2,
                                              random_state=1)

    scaler = MinMaxScaler(copy=False).fit(X_tr)
    return Bunch(X_tr=scaler.transform(X_tr),
                 X_ts=scaler.transform(X_ts),
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=2,
                 class_names=['negative', 'positive'])


def load_mnist_4_9():
    (X_tr, y_tr), (X_ts, y_ts) = keras.datasets.mnist.load_data()
    tr_idxs = np.where(((y_tr == 4) | (y_tr == 9)))
    y_tr = y_tr[tr_idxs]
    X_tr = X_tr[tr_idxs]
    y_tr[y_tr == 4] = 0
    y_tr[y_tr == 9] = 1
    ts_idxs = np.where(((y_ts == 4) | (y_ts == 9)))
    X_ts = X_ts[ts_idxs]
    y_ts = y_ts[ts_idxs]
    y_ts[y_ts == 4] = 0
    y_ts[y_ts == 9] = 1
    return Bunch(X_tr=X_tr.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 X_ts=X_ts.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=2,
                 class_names=['4','9'])


def load_mnist():
    (X_tr, y_tr), (X_ts, y_ts) = keras.datasets.mnist.load_data()
    return Bunch(X_tr=X_tr.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 X_ts=X_ts.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=10,
                 class_names=list(map(str, range(10))))


def load_fashion_mnist():
    (X_tr, y_tr), (X_ts, y_ts) = keras.datasets.fashion_mnist.load_data()
    return Bunch(X_tr=X_tr.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 X_ts=X_ts.reshape(-1, 28, 28, 1) / 255 - 0.5,
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=10,
                 class_names=list(map(str, range(10))))


def load_cifar10():
    (X_tr, y_tr), (X_ts, y_ts) = keras.datasets.cifar10.load_data()
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]
    return Bunch(X_tr=X_tr / 255 - 0.5,
                 X_ts=X_ts / 255 - 0.5,
                 y_tr=keras.utils.to_categorical(y_tr),
                 y_ts=keras.utils.to_categorical(y_ts),
                 n_classes=10,
                 class_names=class_names)


def loat_20ng():
    dataset_path = os.path.join('data', 'newsgroups+embeddings+pca@100.pickle')
    dataset = load(dataset_path)
    X_tr, X_ts, y_tr, y_ts = train_test_split(dataset['data'],
                                              dataset['target'],
                                              test_size=0.2,
                                              random_state=1)
    return Bunch(X_tr=X_tr,
                 X_ts=X_ts,
                 y_tr=y_tr.astype(np.float),
                 y_ts=y_ts.astype(np.float),
                 n_classes=len(dataset['target_names']),
                 class_names=dataset['target_names'])


def _get_basename(args):
    fields_model = [
        (None, args.dataset),
        (None, args.model),
        ('noise_type', args.noise_type),
        ('B', args.batch_size),
        ('E', args.n_epochs),
        ('logits', args.from_logits),
    ]

    basename = '__'.join([name + '=' + str(value) if name else str(value)
                          for name, value in fields_model])
    return basename


def inject_noise(args, dataset, rng=None):
    """Returns a noisy copy of a dataset.

    The returned dataset is *identical* to the original (it actually references
    the same arrays!) except for the training labels y_tr, which points to a
    completely new, noisy array.

    Arguments
    ---------
    dataset : Bunch
        The dataset to be corrupted.
    p_noise : float
        The amount of noise in [0, 1].
    rng : None or int or RandomState
        The RNG.

    Returns
    -------
    noisy_dataset : Bunch
        The corrupted dataset.
    """
    p_noise = args.p_noise
    rng = check_random_state(rng)
    assert 0 <= p_noise <= 1

    noisy_dataset = copy(dataset)
    noisy_dataset.y_tr = np.array(dataset.y_tr)

    n_examples, n_classes = noisy_dataset.y_tr.shape
    n_noisy = int(np.round(n_examples * p_noise))

    if args.noise_type in ['max_margin', 'min_margin', 'min_max_margin',
                           'max_margin_cluster']:
        if os.path.exists(f'data/{_get_basename(args)}.pickle'):
            print('load for noise injection')
            proba = load(f'data/{_get_basename(args)}.pickle')
        else:
            model = make_model(args.model, dataset)
            model.fit(dataset.X_tr, dataset.y_tr, epochs=10, verbose=0)

            proba = model.predict(dataset.X_tr)
            dump(f'data/{_get_basename(args)}.pickle', proba)

        proba_margin = np.amax(proba, axis=1) - np.amin(proba, axis=1)
        sorted_examples = np.argsort(proba_margin)  # ascendent

        if args.noise_type == 'max_margin':
            sorted_examples = sorted_examples[::-1]

        elif args.noise_type == 'min_margin':
            pass

        elif args.noise_type == 'min_max_margin':
            # half min margin half max margin
            half = int(np.round(n_noisy / 2))
            sorted_examples = np.hstack(
                [sorted_examples[:half], sorted_examples[-(n_noisy - half):]])

        elif args.noise_type == 'max_margin_cluster':
            clean, noisy = _max_margin_cluster_noise(dataset, n_classes,
                                                     n_examples, n_noisy, noisy_dataset,
                                                     rng, sorted_examples)

            return noisy_dataset, clean, noisy

        else:
            raise ValueError(args.noise_type)

        noisy = list(sorted(sorted_examples[:n_noisy]))
    elif args.noise_type == 'outlier':
        noisy = _outlier_noise(args, n_noisy, noisy_dataset, rng)

    elif args.noise_type == 'random':
        noisy = list(sorted(rng.permutation(n_examples)[:n_noisy]))

    else:
        raise ValueError(f'Invalid noise type {args.noise_type}')

    clean = list(sorted(set(range(n_examples)) - set(noisy)))
    all_labels = set(range(dataset.n_classes))
    for i in noisy:
        label = np.nonzero(dataset.y_tr[i])[0][0]
        noisy_label = rng.choice(list(sorted(all_labels - {label})))
        noisy_dataset.y_tr[i] = np.zeros(n_classes)
        noisy_dataset.y_tr[i, noisy_label] = 1

    if args.dataset == 'synthetic':
        _plot_synthetic_dataset(noisy, noisy_dataset)

    return noisy_dataset, clean, noisy


def _plot_synthetic_dataset(noisy, noisy_dataset):
    plt.figure()
    c = np.argmax(noisy_dataset.y_tr, axis=1)
    plt.scatter(noisy_dataset.X_tr[:, 0], noisy_dataset.X_tr[:, 1], marker='x',
                c=c)
    plt.scatter(noisy_dataset.X_tr[noisy, 0],
                noisy_dataset.X_tr[noisy, 1], marker='o', c='red')
    plt.savefig(f'{str(np.random.rand())}_synthetic_dataset.png')


def _max_margin_cluster_noise(dataset, n_classes, n_examples, n_noisy,
                              noisy_dataset, rng, sorted_examples):
    # make clusters in different classes
    min_n_examples_per_cluster = 10
    n_cluster = min(dataset.n_classes, n_noisy / min_n_examples_per_cluster)
    n_examples_per_cluster = int(n_noisy / n_cluster)
    sorted_examples = sorted_examples[::-1]
    classes_already_taken = []
    cluster_center = []
    noisy = set()
    i = 0

    ds = dataset.X_tr.reshape(dataset.X_tr.shape[0], -1)
    tree = BallTree(ds)
    while len(classes_already_taken) < n_cluster:
        ex_id = sorted_examples[i]
        ex_y = np.argmax(dataset.y_tr[ex_id])
        i += 1
        if ex_y in classes_already_taken:
            continue

        cluster_center.append(ex_id)
        classes_already_taken.append(ex_y)

        indices_of_example = tree.query(dataset.X_tr[ex_id].reshape(1, -1),
                                        return_distance=False,
                                        sort_results=True,
                                        k=n_examples_per_cluster*5)[0]

        all_labels = set(range(dataset.n_classes))
        noisy_label = rng.choice(list(sorted(all_labels - {ex_y})))
        ex_in_cluster = 0
        for ex in indices_of_example:
            if ex_in_cluster == n_examples_per_cluster:
                break
            if np.argmax(dataset.y_tr[ex]) == noisy_label:
                continue

            ex_in_cluster += 1
            noisy_dataset.y_tr[ex] = np.zeros(n_classes)
            noisy_dataset.y_tr[ex, noisy_label] = 1
            noisy.add(ex)

    clean = list(sorted(set(range(n_examples)) - set(noisy)))

    return clean, noisy


def _outlier_noise(args, n_noisy, noisy_dataset, rng):
    path = 'data/probabilities_outlier_' + args.dataset + '.pickle'
    if os.path.exists(path) and False:
        print('load for noise injection')
        p = load(path)['p']
    else:
        print('Computing')
        X = noisy_dataset.X_tr.ravel().reshape(noisy_dataset.X_tr.shape[0], -1)
        square_form_mx = squareform(pdist(X))

        print('Computing 2')
        mask = np.ones(square_form_mx.shape)
        np.fill_diagonal(mask, np.nan)
        min_value = np.nanmin(square_form_mx * mask, axis=1)

        Z = np.sum(np.exp(min_value))
        p = np.exp(min_value) / Z

        # dump(path, {'dist_mx': square_form_mx, 'p': p})
    noisy = list(rng.choice(np.arange(len(p)), p=p, size=n_noisy, replace=False))
    return noisy


def get_n_known(dataset, p_known):
    n_examples = len(dataset.y_tr)
    if p_known > 1:
        n_known = int(np.round(p_known))
    elif p_known > 0:
        n_known = int(np.round(n_examples * p_known))
    else:
        raise ValueError('p_known must be > 0')
    return n_known


def subsample_train(dataset, p_known, rng=None):
    """Inplace."""
    n_examples = len(dataset.y_tr)
    n_known = get_n_known(dataset, p_known)
    kn = rng.permutation(range(n_examples))[:n_known]
    dataset.X_tr = dataset.X_tr[kn]
    dataset.y_tr = dataset.y_tr[kn]


DATASETS = {
    'synthetic': load_synthetic,
    'iris': load_iris_,
    'breast': load_breast,
    'wine': load_wine_,
    'adult': load_adult,
    'german': load_german,
    'mnist': load_mnist,
    'cifar10': load_cifar10,
    '20ng': loat_20ng,
    'mnist49': load_mnist_4_9,
    'fashion_mnist': load_fashion_mnist
}
