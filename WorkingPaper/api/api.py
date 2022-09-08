from fastapi import FastAPI
from pdfminer.high_level import extract_text
from WorkingPaper.save_load_models.load_model import load_model
from WorkingPaper.preprocessing.preprocessing_prediction import preprocessor_pred
from WorkingPaper.python_model_antoine.DL_logic.cleaning import final_cleaning
from starlette.responses import Response
import io
import numpy as np
import joblib

app = FastAPI()

@app.get('/')
def index():
    return{'ok':True}

@app.get('/path')
def path(path: str):
    pdf = open(path, 'rb')

    print('Extracting_text...')
    text = extract_text(pdf)
    text_clean = final_cleaning(text)
    a=np.array([text_clean])
    b=np.array([text_clean])

    print('Loading Model...')
    model_1 = joblib.load('/home/adolfo/code/Fito-22/WorkingPaper/WorkingPaper/local/models/models/ML_model_layer1.joblib')
    model_2 = joblib.load('/home/adolfo/code/Fito-22/WorkingPaper/WorkingPaper/local/models/models/ML_model_layer2.joblib')

    print('Making the first prediction...')
    y_pred = model_1.predict(a)
    print(y_pred)

    print('Finding values...')
    i = y_pred[0]
    print(i)
    if i==1: topic='mathematics'
    elif i==2: topic='physics'
    else:
        print('Making the second prediction...')
        y_pred_2layer = model_2.predict(b)
        i_2 = y_pred_2layer
        if i_2==1: topic='medicine'
        elif i_2==0: topic='biology'
        else: topic='Something is wrong'
    return {'topic':str(topic)}



#@app.post('/uploadfile')
#async def uploadfile(file: bytes=File(...)):

 #   a = extract_text(io.BytesIO(file))
    #X_prep = preprocessor_pred(text)
    #model = load_model()
    #y_pred=model.predict(X_preprocess)

  #  return a
