import re
import nltk
from preprocessing.text_preprocessing import preprocess_text


def extract_thread_features(conversation):
    # Extract features for the source thread of the conversation

    feature_dict = {}
    # Tokenize
    # Regexp replaces all except non-words and numbers with empty
    tw = conversation['source']
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                       tw['text'].lower()))

    # Gather all other response on the conversation
    otherthreadtweets = ''
    for response in conversation['replies']:
        otherthreadtweets += ' ' + response['text']
    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                                  otherthreadtweets.lower()))

    raw_txt = tw['text']
    feature_dict['raw_text'] = raw_txt
    feature_dict['spacy_processed_text'], feature_dict['spacy_processed_NERvec'] = preprocess_text(raw_txt)
    feature_dict['hasqmark'] = 0
    if tw['text'].find('?') >= 0:
        feature_dict['hasqmark'] = 1
    feature_dict['hasemark'] = 0
    if tw['text'].find('!') >= 0:
        feature_dict['hasemark'] = 1
    feature_dict['hasperiod'] = 0
    if tw['text'].find('.') >= 0:
        feature_dict['hasperiod'] = 1
    feature_dict['hashashtag'] = 0
    if tw['text'].find('#') >= 0:
        feature_dict['hashashtag'] = 1
    feature_dict['hasurl'] = 0
    if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
        feature_dict['hasurl'] = 1
    feature_dict['haspic'] = 0
    if (tw['text'].find('picpicpic') >= 0) or (
            tw['text'].find('pic.twitter.com') >= 0) or (
            tw['text'].find('instagr.am') >= 0):
        feature_dict['haspic'] = 1
    feature_dict['hasnegation'] = 0
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                     'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                     'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                     'couldn', 'doesn']
    for negationword in negationwords:
        if negationword in tokens:
            feature_dict['hasnegation'] += 1
    swearwords = []
    with open('src/preprocessing/data/badwords.txt', 'r') as f:
        for line in f:
            swearwords.append(line.strip().lower())
    feature_dict['hasswearwords'] = 0
    for token in tokens:
        if token in swearwords:
            feature_dict['hasswearwords'] += 1

    feature_dict['src_rumour'] = 0
    feature_dict['thread_rumour'] = 0
    if 'rumour' in tokens or 'gossip' in tokens:
        feature_dict['src_rumour'] = 1
    if ('rumour' in otherthreadtokens) or ('gossip' in otherthreadtokens):
        feature_dict['thread_rumour'] = 1

    return feature_dict

def extract_thread_features_incl_response(conversation):
    source_features = extract_thread_features(conversation)   # For the conversation source text
    source_features['issource'] = 1
    fullthread_featdict = {}
    fullthread_featdict[conversation['source']['id_str']] = source_features

    for tw in conversation['replies']:
        feature_dict = {}
        feature_dict['issource'] = 0
        tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                           tw['text'].lower()))
        otherthreadtweets = ''
        otherthreadtweets += conversation['source']['text']

        for response in conversation['replies']:
            otherthreadtweets += ' ' + response['text']

        otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',  otherthreadtweets.lower()))
        
        raw_txt = tw['text']
        # Add boolean features
        feature_dict['hasqmark'] = 0
        if tw['text'].find('?') >= 0:
            feature_dict['hasqmark'] = 1
        feature_dict['hasemark'] = 0
        if tw['text'].find('!') >= 0:
            feature_dict['hasemark'] = 1
        feature_dict['hashashtag'] = 0
        if tw['text'].find('#') >= 0:
            feature_dict['hashashtag'] = 1
        feature_dict['hasurl'] = 0
        if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
            feature_dict['hasurl'] = 1
        feature_dict['haspic'] = 0
        if (tw['text'].find('picpicpic') >= 0) or (
                tw['text'].find('pic.twitter.com') >= 0) or (
                tw['text'].find('instagr.am') >= 0):
            feature_dict['haspic'] = 1
        # Add number of negative words
        feature_dict['hasnegation'] = 0
        negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                         'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                         'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                         'couldn', 'doesn']
        for negationword in negationwords:
            if negationword in tokens:
                feature_dict['hasnegation'] += 1
        # Add number of swear words
        swearwords = []
        with open('src/preprocessing/data/badwords.txt', 'r') as f:
            for line in f:
                swearwords.append(line.strip().lower())
        feature_dict['hasswearwords'] = 0
        for token in tokens:
            if token in swearwords:
                feature_dict['hasswearwords'] += 1

        # Added textual features
        feature_dict['raw_text'] = raw_txt
        feature_dict['spacy_processed_text'], feature_dict['spacy_processed_NERvec'] = preprocess_text(raw_txt)

        feature_dict['src_rumour'] = 0
        feature_dict['thread_rumour'] = 0
        if 'rumour' in tokens or 'gossip' in tokens:
            feature_dict['src_rumour'] = 1
        if ('rumour' in otherthreadtokens) or ('gossip' in otherthreadtokens):
            feature_dict['thread_rumour'] = 1
        
        fullthread_featdict[tw['id_str']] = feature_dict
    return fullthread_featdict