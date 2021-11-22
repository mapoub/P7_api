# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /app

#ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt
COPY app.py app.py 
RUN pip3 install -r requirements.txt
#EXPOSE  7878
COPY . .

CMD [ "flask", "run"]
