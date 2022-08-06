from os import path
import json as js
from argparse import ArgumentParser
from modeling.bert_framework import BERT_Framework, RoBERTa_Framework, GPT2_Framework
from modeling.text_bert import BertModelForStanceClassification, RoBertaModelForStanceClassification, GPT2ModelForStanceClassification


def main(model='baseline'):
    

    with open(path.join('src', 'config.json')) as f:
        config = js.load(f)

    if model == 'baseline':
        fworkf = BERT_Framework
        modelf = BertModelForStanceClassification
        modelframework = fworkf(config['baseline'], modelf)
    elif model == 'roberta':
        fworkf = RoBERTa_Framework
        modelf = RoBertaModelForStanceClassification
        modelframework = fworkf(config['roberta'], modelf)
    elif model == 'gpt2':
        fworkf = GPT2_Framework
        modelf = GPT2ModelForStanceClassification
        modelframework = fworkf(config['gpt2'], modelf)
    
    
    modelframework.fit(modelf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Rather running the baseline or new model', required=False, choices=['baseline', 'roberta', 'gpt2'], default='roberta')

    args = parser.parse_args()
    kwargs = vars(args)
    model = kwargs['model']

    main(model=model)