import json
import os

import numpy as np
from tqdm import tqdm

from preprocessing.extract_thread_features import extract_thread_features_incl_response
from preprocessing.preprocessing_reddit import load_data, load_test_data_reddit
from preprocessing.preprocessing_tweets import load_dataset, load_test_data_twitter
from preprocessing.transform_feature_dict import transform_feature_dict


def prep_pipeline():
    path = os.path.join('data_preprocessing', 'saved_data_RumEval2019')
    feature_set = [
        'issource',
        'raw_text',
        'spacy_processed_text',
        'spacy_processed_BLvec',
        'spacy_processed_POSvec',
        'spacy_processed_DEPvec',
        'spacy_processed_NERvec'
    ]

    folds = {}
    folds = load_dataset()

    folds["test"] = load_test_data_twitter()["test"]
    
    # use reddit data
    reddit = load_data()
    folds['train'].extend(reddit['train'])
    folds['dev'].extend(reddit['dev'])
    reddit_test_data = load_test_data_reddit()['test']
    folds["test"].extend(reddit_test_data)

    # data folds , i.e. train, dev, test
    for fold in folds.keys():
        print(fold)
        # contains features for each branch in all conversations
        # shape shape conversations_count *BRANCH_COUNT x BRANCH_LEN x FEATURE vector
        fold_features = []
        fold_features_dict = []
        # contains ids of processed tweets in branches in all conversations
        #  shape conversations_count * BRANCH_COUNT x BRANCH_len x String
        tweet_ids = []
        # contains stance labels for all branches in all conversations
        # final shape conversations_count * BRANCH_COUNT for the conversation x BRANCH_len
        fold_stance_labels = []
        conv_ids = []

        all_fold_features = []
        for conversation in tqdm(folds[fold]):
            # extract features for source and replies
            thread_feature_dict = extract_thread_features_incl_response(conversation)
            all_fold_features.append(thread_feature_dict)

            thread_features_array, thread_features_dict, thread_stance_labels, branches = transform_feature_dict(
                thread_feature_dict, conversation,
                feature_set=feature_set)

            fold_features_dict.extend(thread_features_dict)
            fold_stance_labels.extend(thread_stance_labels)
            tweet_ids.extend(branches)
            fold_features.extend(thread_features_array)

            # build data for source tweet for veracity
            for i in range(len(thread_features_array)):
                conv_ids.append(conversation['id'])

        # % 0 supp, 1 comm,2 deny, 3 query
        if fold_features != []:
            path_fold = os.path.join(path, fold)
            print(f"Writing dataset {fold}")
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)

            jsonformat = {"Examples": []}
            cnt = 0
            already_known_tweetids = set()
            for fold_idx in tqdm(range(len(fold_features_dict))):
                e = fold_features_dict[fold_idx]
                tweet_ids_branch = tweet_ids[fold_idx]
                branch_labels = fold_stance_labels[fold_idx].tolist()
                for idx in range(len(e)):
                    if tweet_ids_branch[idx] in already_known_tweetids:
                        continue
                    else:
                        already_known_tweetids.add(tweet_ids_branch[idx])
                    example = {
                        "id": cnt,
                        "branch_id": f"{fold_idx}.{idx}",
                        "tweet_id": tweet_ids_branch[idx],
                        "stance_label": branch_labels[idx], # if not fold == "test" else -1,
                        "raw_text": e[idx]["raw_text"],
                        "raw_text_prev": e[idx - 1]["raw_text"] if idx - 1 > -1 else "",
                        "raw_text_src": e[0]["raw_text"] if idx - 1 > -1 else "",
                        "issource": e[idx]["issource"],
                        "spacy_processed_text": e[idx]["spacy_processed_text"],
                        'spacy_processed_BLvec': e[idx]['spacy_processed_BLvec'],
                        'spacy_processed_POSvec': e[idx]['spacy_processed_POSvec'],
                        'spacy_processed_DEPvec': e[idx]['spacy_processed_DEPvec'],
                        'spacy_processed_NERvec': e[idx]['spacy_processed_NERvec'],
                        "spacy_processed_text_prev": e[idx - 1]["spacy_processed_text"] if idx - 1 > -1 else "",
                        "spacy_processed_text_src": e[0]["spacy_processed_text"] if idx - 1 > -1 else ""
                    }
                    cnt += 1
                    example = {i: (v if type(v) is not np.ndarray else v.tolist())
                                for i, v in example.items()}
                    jsonformat["Examples"].append(example)

            json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))
            

if __name__ == '__main__':
    prep_pipeline()