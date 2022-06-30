from os import path
import pandas as pd


def load_olid_files():
    df_train = pd.read_csv(
        path.join('data', 'OLID', 'olid-training-v1.0.tsv'), 
        sep='\t',
        usecols=['tweet', 'subtask_a']
        ).rename(columns={'subtask_a': 'label'})

    df_test_text = pd.read_csv(
        path.join('data', 'OLID', 'testset-levela.tsv'), 
        sep='\t',
        usecols=['id', 'tweet']
        )
    df_test_labels = pd.read_csv(
        path.join('data', 'OLID', 'labels-levela.csv'),
        header=None
        )
    df_test_labels.columns = ['id', 'label']
    
    df_test = pd.merge(left=df_test_text, right=df_test_labels, on='id', how='inner')

    return df_train, df_test
    