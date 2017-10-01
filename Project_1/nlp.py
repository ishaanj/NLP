import os
import re
from nltk.tokenize import word_tokenize

cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'

pos = train_dir + "pos.txt"
neg = train_dir + "neg.txt"

def add_start_end_tokens(filename):
    unigram_count = {}
    bigram_count = {}
    f = open(filename, 'r')
    for line in f.readlines():
        temp = ["<s>"]
        temp.extend(word_tokenize(line))
        temp.append("</s>")
        unigram_total_count = 0
        for t in range(len(temp)):
            if temp[t] in unigram_count:
                unigram_count[temp[t]]+=1
            else:
                unigram_count[temp[t]] = 1

            unigram_total_count+=1

            if t > 0:
                tup = (temp[t-1], temp[t])
                if tup in bigram_count:
                    bigram_count[tup] += 1
                else:
                    bigram_count[tup] = 1
    return unigram_count, bigram_count, unigram_total_count


def unigram(unigram_count, unigram_total_count):
    prob_unigram = {}
    for i in unigram_count:
        prob = unigram_count[i] / (unigram_total_count * 1.0)
        prob_unigram[i] = prob
    return prob_unigram


def bigram(bigram_count, unigram_count):
    prob_bigram = {}
    for i in bigram_count:
        prob = bigram_count[i] / (unigram_count[i[0]] * 1.0)
        prob_bigram[i] = prob
    return prob_bigram

unigram_count, bigram_count, unigram_total_count = add_start_end_tokens(pos)
print unigram(unigram_count, unigram_total_count)
print bigram(bigram_count, unigram_count)
