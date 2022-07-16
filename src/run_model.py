from os import path
import json as js
from argparse import ArgumentParser
from modeling.bert_framework import BERT_Framework
from modeling.text_bert import BertModelForStanceClassification


def main(model='baseline'):
    fworkf = BERT_Framework

    with open(path.join('src', 'config.json')) as f:
        config = js.load(f)

    if model == 'baseline':
        modelf = BertModelForStanceClassification
        modelframework = fworkf(config=config['baseline'])
    elif model == 'new':
        modelf = None
        modelframework = fworkf(config=config['new_model'])
    
    modelframework.fit(modelf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Rather running the baseline or new model', required=True, choices=['baseline', 'new'], default='baseline')

    args = parser.parse_args()
    kwargs = vars(args)
    model = kwargs['model']

    main(model=model)