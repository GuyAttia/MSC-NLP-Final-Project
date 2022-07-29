from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_array_values_against_length(arr: list, title: str=''):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(arr[0]) # train
    ax.plot(arr[1]) # validation
    ax.plot(arr[2]) # test
    fig.savefig(f"plots/{title}.png")
    plt.show()

def plot_confusion_matrix(y_actu, y_pred, title='Confusion Matrix'):
    df_confusion = pd.crosstab(y_actu, y_pred)
    fig, ax = plt.subplots()
    plt.matshow(df_confusion)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(df_confusion.columns, labels=['support', 'deny', 'query', 'comment'], rotation=45)
    plt.yticks(df_confusion.index, labels=['support', 'deny', 'query', 'comment'])
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig(f"plots/confusion_matrix.png")

