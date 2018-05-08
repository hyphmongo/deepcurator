from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization


def cnn():
    # Architecture based on Keunwoo Choi's paper
    # https://github.com/keunwoochoi/music-auto_tagging-keras

    model = Sequential()
    model.add(BatchNormalization(axis=2, input_shape=(1, 128, 1291)))

    layers = [
        {'filters': 64, 'pool_size': (2, 4)},
        {'filters': 128, 'pool_size': (2, 4)},
        {'filters': 128, 'pool_size': (2, 4)},
        {'filters': 128, 'pool_size': (3, 5)},
        {'filters': 64, 'pool_size': (4, 4)}
    ]

    def add_layer(model, layer):
        model.add(Conv2D(layer.filters, (3, 3),
                         data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(layer.pool_size, data_format='channels_first'))
        model.add(Dropout(0.25))

        return model

    for layer in layers:
        model = add_layer(model, layer)

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    return model
