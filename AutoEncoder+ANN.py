import pandas as pd
import numpy as np

import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from .getLists import *
PATHS,col = getList()


def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    # Encoder
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)

    # Decoder
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='ae_action')(x_ae)

    # Multi Layer perceptron
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)

    out = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='action')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=[out_ae, out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={
                      'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing=ls),
                      'action': tf.keras.losses.BinaryCrossentropy(label_smoothing=ls),
                  },
                  metrics={
                      'ae_action': tf.keras.metrics.AUC(name='AUC'),
                      'action': tf.keras.metrics.AUC(name='AUC'),
                  },
                  )

    return model


params = {'num_columns': len(col),
          'num_labels': 15,
          'hidden_units': [96, 96, 896, 448, 448, 256],
          'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448],
          'ls': 0,
          'lr':1e-3,
         }

if __name__ == '__main__':
    train_df = pd.read_csv('train_df.csv')
    valid_df = pd.read_csV('valid_df.csv')

    batch_size = 64

    fold = 5
    ckp_path = f'AE+ANN_{fold}.hdf5'
    model = create_ae_mlp(**params)
    ckp = ModelCheckpoint(ckp_path, monitor='val_action_AUC', verbose=0,
                          save_best_only=True, save_weights_only=False, mode='max')
    es = EarlyStopping(monitor='val_action_AUC', min_delta=1e-4, patience=20, mode='max',
                       baseline=None, restore_best_weights=True, verbose=0)
    history = model.fit(train_df[col], [train_df.iloc[:, -15:], train_df.iloc[:, -15:]],
                        validation_data=(valid_df[col], [valid_df.iloc[:, -15:], valid_df.iloc[:, -15:]]),
                        epochs=10, batch_size=batch_size, callbacks=[ckp, es], verbose=True)
    hist = pd.DataFrame(history.history)
