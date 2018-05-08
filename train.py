import numpy as np
import os.path
import math
import csv
import re
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from PIL import Image

from s3 import Client
from models import cnn

s3 = Client()

BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MODEL_NAME = 'model.hdf5'


def load_labels():
    global labels
    labels = {}

    with open('labels.csv') as f:
        for line in csv.reader(f):
            labels[line[0]] = line[1]


def generate(ids):
    while 1:
        indexes = np.arange(len(ids))
        np.random.shuffle(indexes)
        batches = int((len(indexes))/BATCH_SIZE)

        for i in range(batches):
            batch_start = i * BATCH_SIZE
            batch_end = (i+1) * BATCH_SIZE

            ids_to_load = [ids[k] for k in indexes[batch_start:batch_end]]
            yield loader(ids_to_load)


def loader(ids):
    x = np.empty((BATCH_SIZE, 1, 128, 1291))
    y = np.empty((BATCH_SIZE), dtype=int)

    for i, id in enumerate(ids):
        x[i, ] = get_audio_slice(id)
        y[i] = labels[get_id_from_slice(id)]

    return x, y


def get_id_from_slice(slice):
    return re.search(r'/(?<=/)(.*)(?=-[0-9]+.png)', slice).group(1)


def get_audio_slice(slice):
    if not os.path.exists(slice):
        s3.download(slice)

    img = Image.open(slice)
    image_data = np.asarray(img, dtype=np.uint8)
    return image_data/255.


class UploadCheckpoint(Callback):

    def __init__(self):
        self.last_change = None
        self.path_local = 'models/' + MODEL_NAME

    def on_epoch_end(self, *args):
        if os.path.getmtime(self.path_local) != self.last_change:
            s3.upload(self.path_local)
            self.last_change = os.path.getmtime(self.path_local)


def segment(slices, ids):
    return [slice for slice in slices if get_id_from_slice(slice) in ids]


def train():
    slices = s3.list_slices()

    ids = list(labels.keys())
    np.random.shuffle(ids)

    split = int(math.ceil(len(ids) * 0.7))

    train = segment(slices, ids[:split])
    test = segment(slices, ids[split:])

    training_generator = generate(train)
    validation_generator = generate(test)

    s3_upload = UploadCheckpoint()

    early_stop = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    filepath = 'models/' + MODEL_NAME
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', save_best_only=True, mode='max')
    callbacks_list = [checkpoint, s3_upload, early_stop]

    model = cnn()

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=len(train)//BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=len(test)//BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list)


if __name__ == "__main__":
    load_labels()
    train()
