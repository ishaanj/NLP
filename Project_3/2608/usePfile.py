import time
import re
import string
import spacy
import json
import os.path


def normalize_context(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def spacy_me(sentence):
    sentence_ents = [(ent.text, int(ent.start_char), int(ent.end_char), ent.label_) for ent in
                     sentence.ents]
    return


def classify_write(filename):
    pass
