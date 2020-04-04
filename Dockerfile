FROM python:3.7-slim

COPY ./app /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt
RUN pipenv install

EXPOSE 5000

# Define environment variable
ENV MODEL_NAME MLScore
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD pipenv run seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
