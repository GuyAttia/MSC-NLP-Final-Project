# MSC-NLP-Final-Project
## Introduction
The challenge focused on classifying whether posts from Twitter and Reddit support, deny, query, or comment a hidden rumour, truthfulness of which is the topic of an underlying discussion thread. 

## Raw data
Download data into the "data" folder inside the root directory. 
The structure should be:
- data
    - rumoureval-2019-test-data
        - reddit-test-data
        - twitter-en-test-data
    - rumoureval-2019-training-data
        - reddit-dev-data
        - reddit-training-data
        - twitter-english
        - dev-key.json
        - train-key.json
    - final-eval-key.json

## Environment
There are two environments that you can run this project scripts through: local Docker container or Google Colab notebook 
### Docker Container
1. Create the requested environment using docker-compose: `docker-compose up -d`
2. Attach to the running container with bash: `docker exec -it {container_id} bash`
3. Run python scripts (detailed below)
### Google Colab
1. Copy this entire repository into your Google Drive (including the data folder)
2. Open the "run_project.ipynb" notebook that you can find in the root folder of this Repo.
3. Run cells.

## Relevant Scripts
There are two main scripts you have to run:
1. data processing: `python src/data_processing.py`. 
The preprocessed data (output) should be available in data_preprocessing/saved_data_RumEval2019
2. Run a model: `python src/run_model.py -m {model_name}`. You need to provide the script the name of the model you want to run.
Valid options are:
    - baseline - our baseline - BERT
    - gpt2 - GPT2 model
    - roberta - Plain RoBERTa model
    - roberta_with_features - RoBERTa model with the best features combinations we found.

    For example: `python src/run_model.py -m gpt2`

** Notice - you need to run everything from root directory (so the working directory is root directory)