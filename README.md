# MSC-NLP-Final-Project
## Introduction
Determining Rumour Veracity and Support for Rumours, Subtask A (Gorrell et al., 2019).
The challenge focused on classifying whether posts from Twitter and Reddit support, deny, query, or comment a hidden rumour, truthfulness of which is the topic of an underlying discussion thread. 
We formulate the problem as a stance classification, determining the rumour stance of a post with respect to the previous thread post and the source thread post.


## Processing the original data into model's format and running the training
- download en models for spacy `python -m spacy download en`
- download data and change the paths accordingly, all paths can be changed in data_preprocessing/paths.py. run everything from root directory (so the working directory is root directory), set PYTHONPATH for root directory
export PYTHONPATH=<your_project_root_directory>
- process data `python data_preprocessing/prep_pipeline.py`
The preprocessed data should be available in data_preprocessing/saved_data_RumEval2019
- Then you should be able to run the model BERT_textonly `python solver.py`