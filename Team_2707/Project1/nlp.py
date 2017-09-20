import os
import numpy as np
import time
import math

start_time = time.time()

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
    temp.extend(line.split())
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
#    print("--- add_start_end_tokens %s seconds ---" % (time.time() - start_time))
    return unigram_count, bigram_count, unigram_total_count, next_words


def unigram(unigram_count, unigram_total_count):
    prob_unigram = {}
    for i in unigram_count:
        prob = unigram_count[i] / (unigram_total_count * 1.0)
        prob_unigram[i] = prob
#    print("--- unigram %s seconds ---" % (time.time() - start_time))
    return prob_unigram

def bigram(bigram_count, unigram_count):
    prob_bigram = {}
    for i in bigram_count:
        prob = bigram_count[i] / (unigram_count[i[0]] * 1.0)
        prob_bigram[i] = prob
    # print("--- bigram %s seconds ---" % (time.time() - start_time))
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
    # print("--- gen_unigram_sentence %s seconds ---" % (time.time() - start_time))
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
    # print("--- gen_bigram_sentence %s seconds ---" % (time.time() - start_time))
    return sentence

"""
The python script has to be placed in \\Project1 directory
"""
cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'
dev_dir =cur_dir + '\\SentimentDataset\\Dev\\'

pos_train = train_dir + "pos.txt"
neg_train = train_dir + "neg.txt"

pos_val = dev_dir + "pos.txt"
neg_val = dev_dir + "neg.txt"

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
    # print("--- define_unk %s seconds ---" % (time.time() - start_time))
    pass
def add_zero_prob_words(unigram_count, bigram_count):
    for word1 in unigram_count:
        for word2 in unigram_count:
            if (word1,word2) not in bigram_count:
                bigram_count[(word1,word2)] = 0
    # print("--- add_zero_prob_words %s seconds ---" % (time.time() - start_time))

def add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k):
    total_word_type = len(set(unigram_count))
    #add +k to all counts
    prob_unigram = {}
    for i in unigram_count:
        prob = (unigram_count[i] + k)
        base = (unigram_total_count * 1.0 + total_word_type * k * 1.0)
        prob_unigram[i] = prob/base
    # print("--- add_plus_k_smoothing_unigram %s seconds ---" % (time.time() - start_time))
    return prob_unigram

def add_plus_k_smoothing_bigram(bigram_count, unigram_count, k):
    total_word_type = len(set(bigram_count))
    #add +k to all counts
    prob_bigram = {}
    for i in bigram_count:
        prob = (bigram_count[i] + k) / (unigram_count[i[0]] * 1.0+ total_word_type * k * 1.0)
        prob_bigram[i] = prob
    # print("--- add_plus_k_smoothing_bigram %s seconds ---" % (time.time() - start_time))
    return prob_bigram

def evaluate_dev_model_bigram(prob_bigram_smooth, unigram_count, k, filename):
    f = open(filename, 'r')

    corpus = ["<s>"]
    for line in f.readlines():
        corpus.extend(clean_data_and_split(line, False))
    corpus.append("</s>")

    running_log_prob = 0
    perplexity = 0
    len_bigram = len(corpus)

    for t in range(1, len(corpus)):
        tup = (corpus[t - 1], corpus[t])

        unseen_prob = 0

        # first step; figure out if we are dealing with UNK or UNSEEN
        # we do this by checking the t-1 element; if it's in our unigram corpus, then it's UNSEEN otherwise, UNK
        is_unseen = False
        if (corpus[t - 1] in unigram_count and corpus[t] in unigram_count):
            is_unseen = True
            unseen_prob = k / (unigram_count[corpus[t - 1]] * 1.0 + len_bigram * 1.0)

        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
            # perplexity -= math.log(prob_bigram_smooth[tup])
            continue

        # not found; if it's merely unseen, then add the basic probability
        if is_unseen:
            running_log_prob -= math.log(unseen_prob)
            # perplexity -= math.log(prob_bigram_smooth[tup])

        tup = ("<unk>", corpus[t])
        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
            # perplexity -= math.log(prob_bigram_smooth[tup])
            continue

        tup = (corpus[t - 1], "<unk>")
        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
            # perplexity -= math.log(prob_bigram_smooth[tup])
            continue

        tup = ("<unk>", "<unk>")
        running_log_prob -= math.log(prob_bigram_smooth[tup])
        # perplexity -= math.log(prob_bigram_smooth[tup])

    # print("--- evaluate_dev_model_bigram %s seconds ---" % (time.time() - start_time))
    return running_log_prob, math.log(running_log_prob/len(corpus))

def calculate_k_on_corpus(bigram_count, unigram_count, datasource):
    k_arr = [0.0001, 0.001, 0.005, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.015, 0.025, 0.045, 0.1, 1, 5]
    k_val = {}
    # k_best = []
    for k in k_arr:
        # prob_unigram_smooth = add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k)
        prob_bigram_smooth = add_plus_k_smoothing_bigram(bigram_count, unigram_count, k)
        evaluated_prob, perplexity = evaluate_dev_model_bigram(prob_bigram_smooth, unigram_count, k, datasource)

        k_val[k] = (evaluated_prob, perplexity)
    # print("--- calculate_k_on_corpus %s seconds ---" % (time.time() - start_time))
    return k_val

def test_on_corpus(type):

    train = pos_train
    validation = pos_val
    if type is 'neg':
        train = neg_train
        validation = neg_val

    unigram_count, bigram_count, unigram_total_count, next_words = add_start_end_tokens(train)
    print "*"*80
    define_unk(unigram_count, bigram_count, next_words)
    #add_zero_prob_words(unigram_count, bigram_count)

    prob_unigram = unigram(unigram_count, unigram_total_count)
    prob_bigram = bigram(bigram_count, unigram_count)
    unigram_sentence = gen_unigram_sentence(10, prob_unigram)
    print "%s: Unigram generated sentence:\t%s" % (type, " ".join(unigram_sentence))

    bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words)
    print "%s: Bigram generated sentence:\t%s" % (type, " ".join(bigram_sentence))

    seed = "simply radiates star-power potential"
    bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words, seed=seed)
    print "%s: Bigram generated sentence with seed[%s] :\t%s" % (type, seed, " ".join(bigram_sentence))

    # next, do smoothing
    k_val = calculate_k_on_corpus(bigram_count, unigram_count, validation)

    k_min = None
    k_min_k = None
    k_min_perpl = None
    k_min_perpl_k = None

    for x in k_val:
        # print x, k_val[x]

        if k_min is None or k_min> k_val[x][0]:
            k_min = k_val[x][0]
            k_min_k = x
        if k_min_perpl is None or k_min_perpl > k_val[x][1]:
            k_min_perpl = k_val[x][1]
            k_min_perpl_k = x
    print "%s: Bigram prob smoothed: %s %s, perpexity: %s %s" % (type, k_min, k_min_k, k_min_perpl, k_min_perpl_k)

# count = 0
#
# for i in unigram_count:
#     print "Unigram prob: %s %s" % (i, prob_unigram[i])
#     count+=1
#     if count is 10:
#         break

test_on_corpus('pos')
test_on_corpus('neg')

print("--- %s seconds ---" % (time.time() - start_time))
