# Create the image base on the Miniconda3 image
FROM python:3.9-slim

# Creating the working directory in the container
WORKDIR /nlp
# Copy the local code to the container
COPY . .

# Install requirements
RUN /usr/local/bin/python -m pip install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/nlp/src
ENV PYTHONUNBUFFERED=1