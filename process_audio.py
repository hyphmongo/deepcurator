import librosa
import numpy
import scipy
import boto3
import os
import errno
import csv
import multiprocessing as mp
from s3 import Client

s3 = Client()

MELS = 128
HOP_SIZE = 512
SAMPLE_RATE = 22050
DURATION = 30
SLICE_SIZE = (DURATION * SAMPLE_RATE) // HOP_SIZE

AUDIO_DIR = 'audio/'
SLICE_DIR = 'slices/'
SPEC_DIR = 'spectrograms/'
LABEL_FILE = 'labels.csv'


def save_file(filepath, spectrum):
    scipy.misc.imsave(filepath, numpy.flipud(spectrum))
    s3.upload(filepath)


def create_spectrogram(file, store_file=False):
    aud, sr = librosa.load(AUDIO_DIR + file + '.mp3', sr=SAMPLE_RATE,
                           res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(aud, sr=sr, n_mels=MELS)
    log_mel = librosa.amplitude_to_db(mel)

    if store_file is True:
        save_file(SPEC_DIR + file + '.png', log_mel)

    return log_mel


def slice_spectrogram(file, spectrum, store_file=False):
    if len(spectrum[1]) < SLICE_SIZE:
        return

    number_of_slices = len(spectrum[1]) // SLICE_SIZE

    slices = []

    for i in range(number_of_slices):
        start = SLICE_SIZE * i
        end = SLICE_SIZE * (i + 1)

        sliced = spectrum[:, start:end]

        slices.append(sliced)

        if store_file is True:
            save_file(SLICE_DIR + file + '-' + str(i) + '.png', sliced)

    return slices


def process_track(id, store_file=True):
    if not os.path.exists(AUDIO_DIR + id + '.mp3'):
        s3.download(AUDIO_DIR + id + '.mp3')

    spectrum = create_spectrogram(id, store_file)
    slices = slice_spectrogram(id, spectrum, store_file)
    return slices


def main():
    try:
        os.makedirs('spectrograms')
        os.makedirs('slices')
        os.makedirs('audio')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    existing = s3.list_existing_spectrograms()

    with open(LABEL_FILE) as f:
        missing = [v[0] for v in csv.reader(f) if v[0] not in existing]

    pool = mp.Pool(mp.cpu_count())
    pool.map(process_track, (missing))


if __name__ == "__main__":
    main()
