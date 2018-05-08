from __future__ import unicode_literals
import os
import csv
import youtube_dl
import multiprocessing as mp
import errno
from s3 import Client

s3 = Client()

ydl = youtube_dl.YoutubeDL({
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    }],
    'outtmpl': 'audio/%(id)s.mp3'
})


def save_video(id):
    try:
        ydl.download(['http://www.youtube.com/watch?v=' + id])
    except:
        return

    filename = 'audio/' + str(id) + '.mp3'

    if os.path.exists(filename):
        s3.upload(filename)
        os.remove(filename)


def main():
    try:
        os.makedirs('audio')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    existing_audio = s3.list_audio_files()

    with open('labels.csv') as f:
        ids = [v[0]
               for v in csv.reader(f) if v[0] + '.mp3' not in existing_audio]

        pool = mp.Pool(mp.cpu_count())
        pool.map(save_video, (ids))


if __name__ == "__main__":
    main()
