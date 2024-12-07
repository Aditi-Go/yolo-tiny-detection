FROM python:3.10


WORKDIR /app


COPY . /app


ENV PIP_NO_CACHE_DIR=off
ENV PIP_CACHE_DIR=/tmp/pip-cache


RUN mkdir -p /tmp/pip-cache && pip install --cache-dir=/tmp/pip-cache -r requirements.txt

EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
