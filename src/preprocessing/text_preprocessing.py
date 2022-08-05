import re
import preprocessor as twitter_preprocessor
import spacy
# See spacy tag_map.py for tag explanation
from spacy.symbols import ORTH

validNER = ["UNK",
            "PERSON",  # People, including fictional.
            "NORP",  # Nationalities or religious or political groups.
            "FAC",  # Buildings, airports, highways, bridges, etc.
            "ORG",  # Companies, agencies, institutions, etc.
            "GPE",  # Countries, cities, states.
            "LOC",  # Non-GPE locations, mountain ranges, bodies of water.
            "PRODUCT",  # Objects, vehicles, foods, etc. (Not services.)
            "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
            "WORK_OF_ART",  # Titles of books, songs, etc.
            "LAW",  # Named documents made into laws.
            "LANGUAGE",  # Any named language.
            "DATE",  # Absolute or relative dates or periods.
            "TIME",  # Times smaller than a day.
            "PERCENT",  # Percentage, including "%".
            "MONEY",  # Monetary values, including unit.
            "QUANTITY",  # Measurements, as of weight or distance.
            "ORDINAL",  # "first", "second", etc.
            "CARDINAL",  # Numerals that do not fall under another type.
            ]


def preprocess_text(text: str, lang='en_core_web_sm'):
    text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "$URL$",
                    text.strip())
    twitter_preprocessor.set_options('mentions')
    text = twitter_preprocessor.tokenize(text)

    nlpengine = spacy.load(lang)
    nlpengine.add_pipe('sentencizer')
    for x in ['URL', 'MENTION', 'HASHTAG', 'RESERVED', 'EMOJI', 'SMILEY', 'NUMBER', ]:
        nlpengine.tokenizer.add_special_case(f'${x}$', [{ORTH: f'${x}$'}])

    NERvec = []
    processed_chunk = ""
    doc = nlpengine(text)
    doclen = 0
    for sentence in doc.sents:
        for w in sentence:

            # Some phrases are automatically tokenized by Spacy
            # i.e. New York, in that case we want New_York in our dictionary
            word = "_".join(w.text.split())
            if word.isspace() or word == "":
                continue

            output = word
            output = output.replace("n't", "not")
            doclen += 1
            processed_chunk += "%s " % (output)

            try:
                NERvec.append(validNER.index(w.ent_type_))
            except ValueError:
                NERvec.append(validNER.index('UNK'))

        doclen += 1
        processed_chunk += "[EOS]" + "\n"
        NERvec.append(0)

    processed_chunk = processed_chunk.strip()
    assert len(processed_chunk.split()) == len(NERvec)
    return processed_chunk, NERvec
