import os
import numpy as np

cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'

pos = train_dir + "pos.txt"
neg = train_dir + "neg.txt"


def add_start_end_tokens(filename):
    unigram_count = {}
    bigram_count = {}
    next_words = {}
    f = open(filename, 'r')
    unigram_total_count = 0
    for line in f.readlines():
        temp = ["<s>"]
        line = line.replace(";", " ")
        line = line.replace(":", " ")
        line = line.replace(",", " ")
        line = line.replace("'", " ")
        line = line.replace("`", " ")
        line = line.replace('"', " ")
        line = line.replace('.', " ")
        line = line.replace('\n', " ")
        temp.extend(line.split(" "))
        temp.append("</s>")
        while '' in temp:
            temp.remove('')

        for t in range(len(temp)):
            unigram_total_count+=1
            if temp[t] in unigram_count:
                unigram_count[temp[t]]+=1
            else:
                unigram_count[temp[t]] = 1

            if t > 0:
                tup = (temp[t-1], temp[t])
                if tup in bigram_count:
                    bigram_count[tup] += 1
                else:
                    bigram_count[tup] = 1
                if temp[t-1] in next_words:
                    next_words[temp[t-1]].add(temp[t])
                else:
                    next_words[temp[t-1]] = set()
                    next_words[temp[t-1]].add(temp[t])

    return unigram_count, bigram_count, unigram_total_count, next_words


def unigram(unigram_count, unigram_total_count):
    prob_unigram = {}
    for i in unigram_count:
        if i == '<s>':
            pass
        prob = unigram_count[i] / (unigram_total_count * 1.0)
        prob_unigram[i] = prob
    return prob_unigram


def bigram(bigram_count, unigram_count):
    prob_bigram = {}
    for i in bigram_count:
        prob = bigram_count[i] / (unigram_count[i[0]] * 1.0)
        prob_bigram[i] = prob
    return prob_bigram


def gen_unigram_sentence(prob_unigram):
    unigrams = []
    unigram_probs = []
    for word in prob_unigram:
        unigrams.append(word)
        unigram_probs.append(prob_unigram[word])
    unigram_sentence = np.random.choice(unigrams,10,False,unigram_probs)
    return unigram_sentence


def gen_bigram_sentence(n, prob_bigram, next_words):
    sentence = ["<s>"]
    cur = "<s>"
    for i in range(n):
        next_possible = list(next_words[cur])
        next_probs = []
        for nextt in next_possible:
            next_probs.append(prob_bigram[(cur, nextt)])

        x = np.random.choice(next_possible, 1, False, next_probs)
        sentence.append(x[0])
        cur = x[0]
        if cur == '</s>':
            break
    return sentence

unigram_count, bigram_count, unigram_total_count, next_words = add_start_end_tokens(pos)
prob_unigram = unigram(unigram_count, unigram_total_count)
prob_bigram = bigram(bigram_count, unigram_count)

unigram_sentence = gen_unigram_sentence(prob_unigram)
print "Unigram generated sentence:\t%s" % (" ".join(unigram_sentence))

bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words)
print "Bigram generated sentence:\t%s" % (" ".join(bigram_sentence))
