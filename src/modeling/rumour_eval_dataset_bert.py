import json
import torchtext as tt
from typing import List, Tuple
from pytorch_pretrained_bert import BertTokenizer
from transformers import RobertaTokenizer, GPT2Tokenizer
from torchtext.data import Example
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

"""
This file contains implementation of RumourEval2019 datasets extending torchtext.data.Dataset class
"""

class RumourEval2019Dataset_BERTTriplets(tt.data.Dataset):
    """
    Creates dataset, where each example is composed as triplet: (source post, previous post, target post)
    """

    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, max_examples=None, with_features=False, **kwargs):
        max_length = max_length - 3  # Count without special tokens

        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            counter = 0
            for example in data_json["Examples"]:
                counter += 1
                if max_examples and (counter >= max_examples):  # Only for quick tests
                    break
                make_ids = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
                text = make_ids(example["spacy_processed_text"])
                prev = make_ids(example["spacy_processed_text_prev"])
                src = make_ids(example["spacy_processed_text_src"])
                segment_A = src + prev  # Combine the source sentence and the previous sentence together
                segment_B = text

                # Tokenize the sentences with start and seperators tokens
                if type(tokenizer) == BertTokenizer:
                    text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                            [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                if type(tokenizer) == RobertaTokenizer or type(tokenizer) == GPT2Tokenizer:
                    text_ids = tokenizer.encode("[CLS]") + segment_A + \
                            tokenizer.encode("[SEP]") + segment_B + tokenizer.encode("[SEP]")

                # truncate if exceeds max length
                if len(text_ids) > max_length:
                    # Truncate segment A
                    segment_A = segment_A[:max_length // 2]
                    if type(tokenizer) == BertTokenizer:
                        text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                                [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                    if type(tokenizer) == RobertaTokenizer or type(tokenizer) == GPT2Tokenizer:
                        text_ids = tokenizer.encode("[CLS]") + segment_A + \
                                tokenizer.encode("[SEP]") + segment_B + tokenizer.encode("[SEP]")
                    if len(text_ids) > max_length:
                        # Truncate also segment B
                        segment_B = segment_B[:max_length // 2]
                        if type(tokenizer) == BertTokenizer:
                            text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                                    [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                        if type(tokenizer) == RobertaTokenizer or type(tokenizer) == GPT2Tokenizer:
                            text_ids = tokenizer.encode("[CLS]") + segment_A + \
                                    tokenizer.encode("[SEP]") + segment_B + tokenizer.encode("[SEP]")

                segment_ids = [0] * (len(segment_A) + 2) + [1] * (len(segment_B) + 1)
                input_mask = [1] * len(segment_ids)

                # Number of NER entities
                # NER_entities = len([i for i in example['spacy_processed_NERvec'] if i>0])

                # Create sentiment analysis for the raw data
                # sentiment_analyser = SentimentIntensityAnalyzer()
                # sentiment_raw = sentiment_analyser.polarity_scores(example["raw_text"])
                # sentiment_src = sentiment_analyser.polarity_scores(example["raw_text_src"])
                # sentiment_prev = sentiment_analyser.polarity_scores(example["raw_text_prev"])

                # Build the example with all the relevant features
                example_list = [
                    example["id"],  # id
                    example["branch_id"],   # branch_id
                    example["tweet_id"], # tweet_id
                    example["stance_label"], # stance_label
                    "\n-----------\n".join([example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),  # raw_text
                    example["issource"],  # issource
                    # sentiment_raw["pos"],  # sentiment_raw_pos
                    # sentiment_raw["neu"],  # sentiment_raw_neu
                    # sentiment_raw["neg"],    # sentiment_raw_neg
                    # sentiment_src["pos"],  # sentiment_src_pos
                    # sentiment_src["neu"],  # sentiment_src_neu
                    # sentiment_src["neg"],    # sentiment_src_neg
                    # sentiment_prev["pos"],  # sentiment_prev_pos
                    # sentiment_prev["neu"],  # sentiment_prev_neu
                    # sentiment_prev["neg"]    # sentiment_prev_neg
                    text_ids,   # text
                    segment_ids, # type_mask
                    input_mask  # input_mask
                ] 

                if with_features:
                    example_list += [
                        example['hasqmark'],
                        example['hasemark'],
                        example['hashashtag'],
                        example['hasurl'],
                        example['haspic'],
                        example['hasnegation'],
                        example['hasswearwords'],
                        example['src_rumour'],
                        example['thread_rumour']
                        # NER_entities
                    ]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields_for_text(with_features=False):
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        float_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=False, dtype=torch.float)

        fields = [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_raw_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_raw_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_raw_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_src_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_src_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_src_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_prev_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_prev_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            # ('sentiment_prev_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('input_mask', text_field())
        ]

        if with_features:
            fields += [
                ('hasqmark', float_field()),
                ('hasemark', float_field()),
                ('hashashtag', float_field()),
                ('hasurl', float_field()),
                ('haspic', float_field()),
                ('hasnegation', float_field()),
                ('hasswearwords', float_field()),
                ('src_rumour', float_field()),
                ('thread_rumour', float_field())
                # ('NER_entities', float_field())
            ]
        return fields

