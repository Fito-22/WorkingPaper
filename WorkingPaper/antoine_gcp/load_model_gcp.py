from google.cloud import storage
#import os

def load_model_GCP():

    source = "MODEL_WorkingPaper/my_model.h5"
    target = "WorkingPaper/local/models/my_model.h5"
    BUCKET_NAME = "antolemaire_955"
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name=BUCKET_NAME, user_project=None)
    blob = storage.Blob(source, bucket)
    blob.download_to_filename(filename=target, client=client)
    return None

if __name__ == '__main__':
    load_model_GCP()
    print('Done')
