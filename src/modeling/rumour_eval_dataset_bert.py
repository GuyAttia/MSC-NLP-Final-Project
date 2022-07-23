import json
import torch
import torchtext as tt

from typing import List, Tuple
from pytorch_pretrained_bert import BertTokenizer
from torchtext.data import Example
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
This file contains implementation of RumourEval2019 datasets extending torchtext.data.Dataset class
"""

class RumourEval2019Dataset_BERTTriplets(tt.data.Dataset):
    """
    Creates dataset, where each example is composed as triplet: (source post, previous post, target post)
    """

    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, **kwargs):
        max_length = max_length - 3  # Count without special tokens
        sentiment_analyser = SentimentIntensityAnalyzer()
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            counter = 0
            for example in data_json["Examples"]:
                ##### Remove for full run #####
                counter += 1
                if counter > 30:
                    break
                ##### Remove for full run #####
                make_ids = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
                text = make_ids(example["spacy_processed_text"])
                prev = make_ids(example["spacy_processed_text_prev"])
                src = make_ids(example["spacy_processed_text_src"])
                segment_A = src + prev
                segment_B = text
                text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                           [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                # truncate if exceeds max length
                if len(text_ids) > max_length:
                    # Truncate segment A
                    segment_A = segment_A[:max_length // 2]
                    text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                               [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                    if len(text_ids) > max_length:
                        # Truncate also segment B
                        segment_B = segment_B[:max_length // 2]
                        text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                                   [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                segment_ids = [0] * (len(segment_A) + 2) + [1] * (len(segment_B) + 1)
                input_mask = [1] * len(segment_ids)
                
                sentiment = sentiment_analyser.polarity_scores(example["raw_text"])
                example_list = [example["id"], example["branch_id"], example["tweet_id"], example["stance_label"],
                                example["veracity_label"],
                                "\n-----------\n".join(
                                    [example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),
                                example["issource"], sentiment["pos"], sentiment["neu"], sentiment["neg"]] + [
                                    text_ids, segment_ids, input_mask]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields_for_text():
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        return [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('input_mask', text_field())]
