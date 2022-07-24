from typing import List
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_array_values_against_length(arr: list, title: str=''):
    print(arr)
    fig, ax = plt.subplots()
    ax.plot(arr)
    fig.savefig("test.png")
    plt.show()

def plot_confusion_matrix(arr1, arr2):
    confusion_matrix(arr1, arr2, labels=[0,1,2,3])