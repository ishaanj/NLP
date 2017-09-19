import os
import numpy as np
import time
import math

def clean_data_and_split(line, add_start_end_tokens):
    line = line.lower()

    temp = []
    if add_start_end_tokens:
        temp.append("<s>")
    line = line.replace(";", " ")
    line = line.replace(":", " ")
    line = line.replace(",", " ")
    line = line.replace("'", " ")
    line = line.replace("`", " ")
    line = line.replace('"', " ")
    line = line.replace('.', " ")
    line = line.replace('\n', " ")
    temp.extend(line.split(" "))
    if add_start_end_tokens:
        temp.append("</s>")
    while '' in temp:
        temp.remove('')
    return temp

def add_start_end_tokens(filename):
    """
    The function reads every line of the file and forms tokens based on delimiters
    It also adds start and end tokens
    :param filename:
    :return:
    unigram_count: {word:count},
    bigram_count: {(word1, word2), count},
    unigram_total_count: total number of word tokens
    next_words: {word:[list of words that follow the word]}
    """
    unigram_count = {}
    bigram_count = {}
    next_words = {}
    f = open(filename, 'r')
    unigram_total_count = 0
    for line in f.readlines():
        temp = clean_data_and_split(line, True)

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
    f.close()
    return unigram_count, bigram_count, unigram_total_count, next_words


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


def gen_unigram_sentence(n, prob_unigram):
    unigrams = []
    unigram_probs = []
    unigram_sentence = []
    for word in prob_unigram:
        unigrams.append(word)
        unigram_probs.append(prob_unigram[word])
    i = 0
    while i < n:
        x = np.random.choice(unigrams, 1, False, unigram_probs)
        if x[0] == '<s>' or x[0] == "</s>":
            continue
        else:
            unigram_sentence.append(x[0])
            i+=1
    return unigram_sentence


def gen_bigram_sentence(n, prob_bigram, next_words, seed="<s>"):
    seed = seed.lower()
    sentence = seed.split(" ")
    cur = sentence[-1]
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

"""
The python script has to be placed in \\Project1 directory
"""
cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'
dev_dir =cur_dir + '\\SentimentDataset\\Dev\\'

pos = train_dir + "pos.txt"
neg = train_dir + "neg.txt"

pos_val = dev_dir + "pos.txt"
neg_val = dev_dir + "neg.txt"

unigram_count, bigram_count, unigram_total_count, next_words = add_start_end_tokens(pos)
# prob_unigram = unigram(unigram_count, unigram_total_count)
# prob_bigram = bigram(bigram_count, unigram_count)
# unigram_sentence = gen_unigram_sentence(10, prob_unigram)
# print "Unigram generated sentence:\t%s" % (" ".join(unigram_sentence))
#
# bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words)
# print "Bigram generated sentence:\t%s" % (" ".join(bigram_sentence))
#
# seed = "simply radiates star-power potential"
# bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words, seed=seed)
# print "Bigram generated sentence with seed[%s] :\t%s" % (seed, " ".join(bigram_sentence))

def define_unk(unigram_count, bigram_count, next_words):
    count = 0
    unk_word = set()
    tempKeys = list(unigram_count.keys())
    for i in tempKeys:
        if unigram_count[i] == 1:
            unk_word.add(i)
            count+=1
            del unigram_count[i]
    unigram_count['<unk>'] = count

    tempKeys = list(bigram_count.keys())
    for i in tempKeys:
        if i[0] in unk_word and i[1] in unk_word:
            if ('<unk>', '<unk>') in bigram_count:
                bigram_count[('<unk>', '<unk>')] += bigram_count[i]
            else:
                bigram_count[('<unk>', '<unk>')] = bigram_count[i]
            del bigram_count[i]

        elif i[0] in unk_word:
            if ('<unk>', i[1]) in bigram_count:
                bigram_count[('<unk>', i[1])] += bigram_count[i]
            else:
                bigram_count[('<unk>', i[1])] = bigram_count[i]
            del bigram_count[i]

        elif i[1] in unk_word:
            if (i[0], '<unk>') in bigram_count:
                bigram_count[(i[0], '<unk>')] += bigram_count[i]
            else:
                bigram_count[(i[0], '<unk>')] = bigram_count[i]
            del bigram_count[i]
    next_word_keys = list(next_words.keys())
    for word in next_word_keys:
        for nextt in list(next_words[word]):
            if nextt in unk_word:
                next_words[word].remove(nextt)
                next_words[word].add('<unk>')

        if word in unk_word:
            if '<unk>' in next_words:
                next_words['<unk>'].update(next_words[word])
            else:
                next_words['<unk>']= next_words[word]
            del next_words[word]
    pass
def add_zero_prob_words(unigram_count, bigram_count):
    for word1 in unigram_count:
        for word2 in unigram_count:
            if (word1,word2) not in bigram_count:
                bigram_count[(word1,word2)] = 0

def add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k):
    total_word_type = len(set(unigram_count))
    #add +k to all counts
    prob_unigram = {}
    for i in unigram_count:
        prob = (unigram_count[i] + k)
        base = (unigram_total_count * 1.0 + total_word_type * k * 1.0)
        prob_unigram[i] = prob/base
    return prob_unigram

def add_plus_k_smoothing_bigram(bigram_count, unigram_count, k):
    total_word_type = len(set(bigram_count))
    #add +k to all counts
    prob_bigram = {}
    for i in bigram_count:
        prob = (bigram_count[i] + k) / (unigram_count[i[0]] * 1.0+ total_word_type * k * 1.0)
        prob_bigram[i] = prob
    return prob_bigram

def evaluate_dev_model_bigram(prob_bigram_smooth, filename):
    f = open(filename, 'r')

    corpus = ["<s>"]
    for line in f.readlines():
        corpus.extend(clean_data_and_split(line, False))
    corpus.append("</s>")

    running_log_prob = 0
    perplexity = 0

    for t in range(len(corpus)):
        if t > 1:
            tup = (corpus[t - 1], corpus[t])
            #print "output"
            #print tup
            if tup in prob_bigram_smooth:
                running_log_prob += math.log(prob_bigram_smooth[tup])
                perplexity -= math.log(prob_bigram_smooth[tup])
                continue

            tup = ("<unk>", corpus[t])
            if tup in prob_bigram_smooth:
                running_log_prob += math.log(prob_bigram_smooth[tup])
                perplexity -= math.log(prob_bigram_smooth[tup])
                continue

            tup = (corpus[t - 1], "<unk>")
            if tup in prob_bigram_smooth:
                running_log_prob += math.log(prob_bigram_smooth[tup])
                perplexity -= math.log(prob_bigram_smooth[tup])
                continue

            tup = ("<unk>", "<unk>")
            running_log_prob += math.log(prob_bigram_smooth[tup])
            perplexity -= math.log(prob_bigram_smooth[tup])

    return running_log_prob, math.log(perplexity/len(corpus))

def calculate_k_on_corpus(bigram_count, unigram_count):
    k_arr = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5]
    k_val = {}
    # k_best = []
    for k in k_arr:
        # prob_unigram_smooth = add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k)
        prob_bigram_smooth = add_plus_k_smoothing_bigram(bigram_count, unigram_count, k)
        evaluated_prob, perplexity = evaluate_dev_model_bigram(prob_bigram_smooth, pos_val)

        k_val[k] = (evaluated_prob, perplexity)
    return k_val

start_time = time.time()

print "*"*80
define_unk(unigram_count, bigram_count, next_words)
add_zero_prob_words(unigram_count, bigram_count)

prob_unigram = unigram(unigram_count, unigram_total_count)
prob_bigram = bigram(bigram_count, unigram_count)
unigram_sentence = gen_unigram_sentence(10, prob_unigram)
print "Unigram generated sentence:\t%s" % (" ".join(unigram_sentence))

bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words)
print "Bigram generated sentence:\t%s" % (" ".join(bigram_sentence))

seed = "simply radiates star-power potential"
bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words, seed=seed)
print "Bigram generated sentence with seed[%s] :\t%s" % (seed, " ".join(bigram_sentence))

# next, do smoothing
k_val = calculate_k_on_corpus(bigram_count, unigram_count)

k_max = None
k_max_k = None
k_min_perpl = None
k_min_perpl_k = None

for x in k_val:
    if k_max is None or k_max < k_val[x][0]:
        k_max = k_val[x][0]
        k_max_k = x
    if k_min_perpl is None or k_min > k_val[x][1]:
        k_min = k_val[x][1]
        k_min_k = x
print "Unigram prob smoothed: %s %s, perpexity: %s %s" % (k_max, k_max_k, k_min, k_min_k)

# count = 0
#
# for i in unigram_count:
#     print "Unigram prob: %s %s" % (i, prob_unigram[i])
#     count+=1
#     if count is 10:
#         break

print("--- %s seconds ---" % (time.time() - start_time))
