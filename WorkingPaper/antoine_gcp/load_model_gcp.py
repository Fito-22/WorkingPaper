from google.cloud import storage
import os

def load_data_GCP():
    local_file = '/home/adolfo/code/Fito-22/WorkingPaper/local_Fito/models/models'
    project_id = os.environ.get("PROJECT")
    bucket_name = os.environ.get("BUCKET_NAME")

    client = storage.Client(project_id)

    bucket = client.get_bucket(bucket_name)


    blob = bucket.blob('MODEL_WorkingPaper/my_model.h5')

    blob.download_to_filename(local_file+'/model_from_gcp.h5')


    return None

if __name__ == '__main__':
    load_data_GCP()
