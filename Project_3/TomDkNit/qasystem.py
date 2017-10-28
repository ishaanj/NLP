import json
#from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import sys



def normalize_text(s):
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


def make_pred(context, question):
    context = context.split()
    question_dict = {}
    cnt = 0
    maxcnt = 0
    for q in question.split():
        question_dict[q] = True
    n = len(context)
    indices = (0,0)
    for i in range(10):
        if(context[i] in question_dict):
            cnt += 1
    if(cnt>=maxcnt):
        maxcnt = cnt
        indices = (0,10)
    for i in range(10,n):
        if(context[i-10] in question_dict):
            cnt -= 1
        if(context[i] in question_dict):
            cnt += 1
        if(cnt>maxcnt):
            maxcnt = cnt
            indices = (i-9,i+1) #remember that i+1 should not be included while slicing
    result = context[indices[0]:indices[1]]
    restr = ""
    for res in result:
        restr += res + " "
    return restr


with open('development.json') as data_file:
    with open('AnswerDev.json','w') as write_file:
        data = json.load(data_file)
        predictions = {}
        for data in data['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                normalized_context = normalize_text(context)
                for qas in paragraph['qas']:
                    question = qas['question']
                    id = qas['id']
                    answers = qas['answers']
                    normalized_question = normalize_text(question)
                    prediction = make_pred(normalized_context, normalized_question)
                    predictions[id] = prediction
        json.dump(predictions,write_file)



