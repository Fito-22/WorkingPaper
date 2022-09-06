FROM python:3.8.6-buster
COPY WorkingPaper /WorkingPaper
COPY requirements.txt /requirements.txt
COPY .env /.env
COPY .envrc /.envrc
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn WorkingPaper.api.api:app --host 0.0.0.0 --port $PORT
