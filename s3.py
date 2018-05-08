import boto3
import re
import yaml


class Client:
    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            config = yaml.load(ymlfile)['s3']

        self.client = boto3.client(
            's3',
            region_name=config['region'],
            aws_access_key_id=config['access_key'],
            aws_secret_access_key=config['secret']
        )

        self.bucket = config['bucket']

    def upload(self, source, destination=None):
        if destination is None:
            destination = source

        self.client.upload_file(source, self.bucket, destination)

    def download(self, source, destination=None):
        if destination is None:
            destination = source

        print (self.bucket, source, destination)

        self.client.download_file(self.bucket, source, destination)

    def list_items(self, prefix=None):
        paginator = self.client.get_paginator('list_objects_v2')

        page_iterator = paginator.paginate(
            Bucket=self.bucket, Prefix=prefix)

        items = []

        for page in page_iterator:
            if "Contents" in page:
                for item in page["Contents"]:
                    key = item["Key"]
                    items.append(key)

        return items

    def list_slices(self):
        return self.list_items('slices/')

    def list_audio_files(self):
        return self.list_items('audio/')

    def list_existing_spectrograms(self):
        items = self.list_items('spectrograms/')
        regex = r'/(?<=/)(.*)(?=.png)'
        return [re.search(regex, key).group(1) for key in items]
