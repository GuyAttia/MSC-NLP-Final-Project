import math
import torch
import datetime
import os
from collections import Iterable


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')


class VERACITY_LABELS:
    true = 0
    false = 1
    unverified = 2


def rmse(labels, pred_probabilities):
    errors = []
    for i, l in enumerate(labels):
        confidence = pred_probabilities[i][l]
        if l == VERACITY_LABELS.unverified:
            errors.append((confidence) ** 2)

        else:
            errors.append((1 - confidence) ** 2)

    return math.sqrt(sum(errors) / len(errors))


def totext(batch, vocab, batch_first=True, remove_specials=True, check_for_zero_vectors=True):
    textlist = []
    if not batch_first:
        batch = batch.transpose(0, 1)
    for ex in batch:
        if remove_specials:
            textlist.append(
                ' '.join(
                    [vocab.itos[ix.item()] for ix in ex
                     if ix != vocab.stoi["<pad>"] and ix != vocab.stoi["<eos>"]]))
        else:
            if check_for_zero_vectors:
                text = []
                for ix in ex:
                    if vocab.vectors[ix.item()].equal(vocab.vectors[vocab.stoi["<unk>"]]):
                        text.append("<OOV>")
                    else:
                        text.append(vocab.itos[ix.item()])
                textlist.append(' '.join(text))
            else:
                textlist.append(' '.join([vocab.itos[ix.item()] for ix in ex]))
    return textlist


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()


class DotDict(dict):
    """
    A dictionary with dot notation for key indexing
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

    # These are needed when python pickle-ing is done
    # for example, when object is passed to another process
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


def totext(batch, vocab, batch_first=True, remove_specials=False, check_for_zero_vectors=True):
    textlist = []
    if not batch_first:
        batch = batch.transpose(0, 1)
    for ex in batch:
        if remove_specials:
            textlist.append(
                ' '.join(
                    [vocab.itos[ix.item()] for ix in ex
                     if ix != vocab.stoi["<pad>"] and ix != vocab.stoi["<eos>"]]))
        else:
            if check_for_zero_vectors:
                text = []
                for ix in ex:
                    if ix != vocab.stoi["<pad>"] and ix != vocab.stoi["<eos>"] \
                            and vocab.vectors[ix.item()].equal(vocab.vectors[vocab.stoi["<unk>"]]):
                        text.append(f"[OOV]{vocab.itos[ix.item()]}")
                    else:
                        text.append(vocab.itos[ix.item()])
                textlist.append(' '.join(text))
            else:
                textlist.append(' '.join([vocab.itos[ix.item()] for ix in ex]))
    return textlist


def dump_detokenize_batch(batch, vocab):
    print("*" * 100)
    print('\n'.join(totext(batch.spacy_processed_text, vocab)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_weights(examples: Iterable, label_field_name: str, classes: int) -> torch.FloatTensor:
    """
    Calculate class weight in order to enforce a flat prior
    :param examples:  data examples
    :param label_field_name: a name of label attribute of the field (if e is an Example and a name is "label",
           e.label will be reference to access label value
    :param classes: number of classes
    :return: an array of class weights (cast as torch.FloatTensor)
    """
    arr = torch.zeros(classes)
    for e in examples:
        arr[int(getattr(e, label_field_name))] += 1

    arrmax = arr.max().expand(classes)
    return arrmax / arr


map_stance_label_to_s = {
    0: "support",
    1: "comment",
    2: "deny",
    3: "query"
}
map_s_to_label_stance = {y: x for x, y in map_stance_label_to_s.items()}