FROM python:3.12-slim
LABEL authors="josephcat"

COPY ./resources /resources

WORKDIR /code

ENV PYTHONPATH=/code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./metal_knight /code

CMD ["fastapi", "run", "app.py", "--port", "80"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app.py", "--port", "80", "--proxy-headers"]