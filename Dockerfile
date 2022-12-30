FROM python:3.8.10-slim-buster

RUN python -m pip install --upgrade pip
ADD requirements.txt .
ADD app.py .
ADD train_dataset.csv .


RUN pip install -r requirements.txt

EXPOSE 8200

CMD [ "python", "app.py" ]
