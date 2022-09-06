from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return{'ok':True}


if __name__ == '__main__':
    index()
