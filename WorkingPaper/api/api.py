from distutils.log import error
from fastapi import FastAPI, UploadFile, File
from pdfminer.high_level import extract_text
from WorkingPaper.save_load_models.load_model import load_model
from WorkingPaper.preprocessing.preprocessing_prediction import preprocessor_pred
from starlette.responses import Response
import io
import codecs

app = FastAPI()

@app.get('/')
def index():
    return{'ok':True}

@app.get('/path')
def path(path: str):
    pdf = open(path, 'rb')
    text = extract_text(pdf)
    X_prep = preprocessor_pred(text)
    model = load_model()
    y_pred = model.predict(X_prep)
    return {'text':y_pred.shape}


@app.post('/uploadfile')
async def uploadfile(file: bytes=File(...)):

    a = extract_text(io.BytesIO(file))
    #X_prep = preprocessor_pred(text)
    #model = load_model()
    #y_pred=model.predict(X_preprocess)

    return a
