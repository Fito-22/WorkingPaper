from distutils.log import error
from fastapi import FastAPI, UploadFile, File
from pdfminer.high_level import extract_text
from WorkingPaper.save_load_models.load_model import load_model
from WorkingPaper.preprocessing.preprocessing_prediction import preprocessor_pred
from starlette.responses import Response
import io
import codecs
from keras import backend as K
import numpy as np


app = FastAPI()

@app.get('/')
def index():
    return{'ok':True}

@app.get('/path')
def path(path: str):
    pdf = open(path, 'rb')

    print('Extracting_text...')
    text = extract_text(pdf)

    print('Preprocessing data...')
    X_prep = preprocessor_pred(text)
    X_pred = np.expand_dims(X_prep, axis=0)
    print(X_prep.shape)

    model = load_model()

    print('Making the prediction...')
    y_pred = model.predict(X_pred)
    print(y_pred)

    print('Finding values...')
    max_accuracy=np.max(y_pred)
    y=list(y_pred[0])
    i = y.index(max_accuracy)

    if i==0: topic='biology'
    elif i==1: topic='chemistry'
    elif i==2: topic='medicine'
    else: topic='psychology'


    return {'accuracy':round(float(max_accuracy),2),'topic':str(topic)}


@app.post('/uploadfile')
async def uploadfile(file: bytes=File(...)):

    a = extract_text(io.BytesIO(file))
    #X_prep = preprocessor_pred(text)
    #model = load_model()
    #y_pred=model.predict(X_preprocess)

    return a
