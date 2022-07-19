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

# Download en models for spacy
RUN python -m spacy download en
# Download nltk resources
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
