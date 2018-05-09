import flask
import heapq
import uuid
import os
import numpy as np
import errno
import tensorflow as tf
from keras.models import load_model
from process_audio import process_track


app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024


@app.route("/rate", methods=["POST"])
def rating():
    audio = flask.request.files.get("audio")

    if audio is None:
        return 'Missing required parameter: audio', 400

    audio_id = uuid.uuid4().hex
    filename = 'audio/' + audio_id + '.mp3'

    audio.save(filename)

    try:
        slices = process_track(audio_id, store_file=False)
    except:
        return 'Could not process provided file', 400
    finally:
        os.remove(filename)

    if slices is None:
        return 'Audio must be longer than 30 seconds', 400

    x = np.empty((len(slices), 1, 128, 1291))

    for i in range(len(slices)):
        x[i, ] = slices[i]/255.

    with graph.as_default():
        predictions = [pred[0] for pred in model.predict(x)]
        best_predictions = heapq.nlargest(5, predictions)

        score = (sum(best_predictions) / len(best_predictions)) * 100

        return str('{:.2f}'.format(score))


if __name__ == "__main__":
    global model
    global graph

    try:
        os.makedirs('audio')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    model = load_model('models/model.hdf5')
    graph = tf.get_default_graph()

    app.run()
