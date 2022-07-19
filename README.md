# MSC-NLP-Final-Project
## Introduction
Determining Rumour Veracity and Support for Rumours, Subtask A (Gorrell et al., 2019).
The challenge focused on classifying whether posts from Twitter and Reddit support, deny, query, or comment a hidden rumour, truthfulness of which is the topic of an underlying discussion thread. 
We formulate the problem as a stance classification, determining the rumour stance of a post with respect to the previous thread post and the source thread post.


## Environment
- Use docker to create the requested environment
`docker-compose up -d`
- Develop and run scripts inside the container

## Processing the original data into model's format and running the training
** Notice - you need to run everything from root directory (so the working directory is root directory)
- Download data into the "data" folder inside the root directory. The structure should be:
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

- process data `python src/data_processing.py`
The preprocessed data should be available in data_preprocessing/saved_data_RumEval2019
- Run our baseline model - BERT_textonly: `python src/run_model.py -m baseline`
- Run our new model: `python src/run_model.py -m new`