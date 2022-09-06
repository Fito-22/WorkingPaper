from fastapi import FastAPI
from pdfminer.high_level import extract_text
from WorkingPaper.save_load_models.load_model import load_model

app = FastAPI()

@app.get('/')
def index():
    return{'ok':True}

@app.get('/pred')
def predict(pdf):
    text = extract_text(pdf)
    #X_preprocess= preprocess(X_clean)
    model = load_model()
    #model.predict(X_preprocess)
    return{'pred':text}
