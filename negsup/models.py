import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class KernelLogreg(Sequential):
    
    def __init__(self, kwargs):
        super(KernelLogreg, self).__init__(kwargs)

    def call(self, inputs, **kwargs):

        def fn(xi):
            return tf.reshape(tf.matmul(xi[:, tf.newaxis], xi[tf.newaxis, :]), [-1])

        inputs = tf.map_fn(fn=fn, elems=inputs)
        return super(KernelLogreg, self).call(inputs, **kwargs)
    # #
    # def fit(self, x, y, **kwargs):
    #     x = np.vstack([np.outer(xi, xi).ravel() for xi in x])
    #     return super(KernelLogreg, self).fit(x,y, **kwargs)
    #
    # def predict(self, x, **kwargs):
    #     x = np.vstack([np.outer(xi, xi).ravel() for xi in x])
    #     return super(KernelLogreg, self).predict(x, **kwargs)


def make_logreg(dataset, to_proba=False):
    input_shape = dataset.X_tr[0].shape
    out_activation = tf.nn.softmax if to_proba else None

    return Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(dataset.n_classes, activation=out_activation, name='hack')
    ])


def make_kernel_logreg(dataset, to_proba=False):
    input_shape = (dataset.X_tr[0].shape[0]**2, )
    out_activation = tf.nn.softmax if to_proba else None

    return KernelLogreg([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(dataset.n_classes, activation=out_activation, name='hack')
    ])


def make_fullnet(dataset, to_proba=False):
    input_shape = dataset.X_tr[0].shape
    out_activation = tf.nn.softmax if to_proba else None

    return Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(dataset.n_classes, activation=out_activation, name='hack')
    ])


def make_convnet(dataset, to_proba=False):
    input_shape = dataset.X_tr[0].shape
    out_activation = tf.nn.softmax if to_proba else None

    return Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(dataset.n_classes, activation=out_activation, name='hack'),
    ])


MODELS = {
    'logreg': make_logreg,
    'kernel_logreg': make_kernel_logreg,
    'fullnet': make_fullnet,
    'convnet': make_convnet,
}


def make_model(model_type, dataset, from_logits=False):
    """Builds a model.

    XXX: from_logits=True is incompatible with expected gradient length.
    """
    model = MODELS[model_type](dataset, to_proba=not from_logits)
    loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    # XXX doesn't stabilize IFs
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=0.001, # adam default
    #    decay_steps=len(dataset.X_tr), # one epoch
    #    decay_rate=0.6) # after 10 epochs it's 0.006
    #optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def make_or_load_model(model_path,
                       model_type,
                       dataset,
                       **kwargs):
    """Builds, trains, and caches a model or loads one from cache."""
    no_cache = kwargs.pop('no_cache', False)
    try:
        if no_cache:
            raise
        model = keras.models.load_model(model_path)
    except:
        model = make_model(model_type,
                           dataset,
                           from_logits=kwargs.pop('from_logits'))
        model.fit(dataset.X_tr,
                  dataset.y_tr,
                  epochs=kwargs.pop('n_epochs'))
        model.save(model_path)
    return model
