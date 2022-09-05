FROM python:3.8.6-buster
COPY WorkingPaper/api /api
COPY requirements.txt /requirements.txt
CMD uvicorn api:app --host 0.0.0.0
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
