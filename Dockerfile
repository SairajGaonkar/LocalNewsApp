FROM python:3.6.8

WORKDIR /NewsApp

COPY . .

RUN chmod 644 app.py
RUN  export PYTHONPATH=/usr/bin/python \
 && pip install -r requirements.txt
CMD  python app.py
