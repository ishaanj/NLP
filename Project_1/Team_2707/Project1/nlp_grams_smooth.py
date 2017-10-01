import os
import numpy as np
import time
import math
import csv

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
test_dir =cur_dir + '\\SentimentDataset\\Test\\'

pos_train = train_dir + "pos.txt"
neg_train = train_dir + "neg.txt"

pos_val = dev_dir + "pos.txt"
neg_val = dev_dir + "neg.txt"

test_data = test_dir + "test.txt"

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
    total_count = 0
    unk_word = set()
    tempKeys = list(unigram_count.keys())
    for i in tempKeys:
        #only add every third entry to unknown; this reduces the number of unknown records, and hopefully increases model performance
        if unigram_count[i] == 1 and total_count%1 == 0:
            unk_word.add(i)
            count+=1
            del unigram_count[i]
        total_count+=1
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
# def add_zero_prob_words(unigram_count, bigram_count):
#     for word1 in unigram_count:
#         for word2 in unigram_count:
#             if (word1,word2) not in bigram_count:
#                 bigram_count[(word1,word2)] = 0
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
        prob = (bigram_count[i] + k) / (unigram_count[i[0]] * 1.0 + total_word_type * k * 1.0)
        prob_bigram[i] = prob
    # print("--- add_plus_k_smoothing_bigram %s seconds ---" % (time.time() - start_time))
    return prob_bigram

def evaluate_dev_model_unigram(prob_unigram_smooth, unigram_count, k, filename):
    f = open(filename, 'r')

    corpus = ["<s>"]
    for line in f.readlines():
        corpus.extend(clean_data_and_split(line, False))
    corpus.append("</s>")

    f.close()

    running_log_prob = 0

    for token in corpus:
        if token in prob_unigram_smooth:
            running_log_prob -= math.log(prob_unigram_smooth[token])
        else:
            running_log_prob -= math.log(prob_unigram_smooth['<unk>'])

    # print("--- evaluate_dev_model_bigram %s seconds ---" % (time.time() - start_time))
    return running_log_prob, math.log(running_log_prob/len(corpus))

def evaluate_dev_model_bigram(prob_bigram_smooth, unigram_count, k, filename):
    f = open(filename, 'r')

    corpus = ["<s>"]
    for line in f.readlines():
        corpus.extend(clean_data_and_split(line, False))
    corpus.append("</s>")

    f.close()

    running_log_prob = 0
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
            continue

        # not found; if it's merely unseen, then add the basic probability
        if is_unseen:
            running_log_prob -= math.log(unseen_prob)
            continue

        tup = ("<unk>", corpus[t])
        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
            continue

        tup = (corpus[t - 1], "<unk>")
        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
            continue

        tup = ("<unk>", "<unk>")
        if tup in prob_bigram_smooth:
            running_log_prob -= math.log(prob_bigram_smooth[tup])
        else:
            #if we've never seen the double unk, use unseen
            running_log_prob -= math.log(unseen_prob)

    # print("--- evaluate_dev_model_bigram %s seconds ---" % (time.time() - start_time))
    return running_log_prob, math.log(running_log_prob/len(corpus))

def calculate_k_on_corpus(bigram_count, unigram_count, unigram_total_count, datasource):
    k_arr = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    # print k_arr
    k_val_unigram = {}
    k_val_bigram = {}
    k_min_unigram = None
    k_min_unigram_val = None
    k_min_bigram = None
    k_min_bigram_val = None
    # k_best = []
    for k in k_arr:
        prob_unigram_smooth = add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k)
        prob_bigram_smooth = add_plus_k_smoothing_bigram(bigram_count, unigram_count, k)

        evaluated_prob_unigram, perplexity_unigram = evaluate_dev_model_unigram(prob_unigram_smooth, unigram_count, k, datasource)
        evaluated_prob_bigram, perplexity_bigram = evaluate_dev_model_bigram(prob_bigram_smooth, unigram_count, k,
                                                                             datasource)

        k_val_unigram[k] = (evaluated_prob_unigram, perplexity_unigram)
        k_val_bigram[k] = (evaluated_prob_bigram, perplexity_bigram)

        if k_min_unigram is None or k_min_unigram_val > evaluated_prob_unigram:
            k_min_unigram_val = evaluated_prob_unigram
            k_min_unigram = k

        if k_min_bigram is None or k_min_bigram_val > evaluated_prob_bigram:
            k_min_bigram_val = evaluated_prob_bigram
            k_min_bigram = k

    k_val_unigram = {}
    k_val_bigram = {}

    #now that we have narrowed down the k_val, we will recall around that
    k_arr = np.arange(k_min_unigram*0.1, k_min_unigram*1.1, step=k_min_unigram*0.1)
    k_arr = np.append(k_arr, np.arange(k_min_bigram * 0.1, k_min_bigram * 1.1, step=k_min_bigram * 0.1))

    for k in k_arr:
        prob_unigram_smooth = add_plus_k_smoothing_unigram(unigram_count, unigram_total_count, k)
        prob_bigram_smooth = add_plus_k_smoothing_bigram(bigram_count, unigram_count, k)

        evaluated_prob_unigram, perplexity_unigram = evaluate_dev_model_unigram(prob_unigram_smooth, unigram_count, k, datasource)
        evaluated_prob_bigram, perplexity_bigram = evaluate_dev_model_bigram(prob_bigram_smooth, unigram_count, k,
                                                                             datasource)

        k_val_unigram[k] = (evaluated_prob_unigram, perplexity_unigram)
        k_val_bigram[k] = (evaluated_prob_bigram, perplexity_bigram)

    # print("--- calculate_k_on_corpus %s seconds ---" % (time.time() - start_time))
    return k_val_unigram, k_val_bigram

def test_on_corpus(run_type):

    train = pos_train
    validation = pos_val
    if run_type is 'neg':
        train = neg_train
        validation = neg_val

    unigram_count, bigram_count, unigram_total_count, next_words = add_start_end_tokens(train)
    print "*"*80
    define_unk(unigram_count, bigram_count, next_words)
    #add_zero_prob_words(unigram_count, bigram_count)

    prob_unigram = unigram(unigram_count, unigram_total_count)
    prob_bigram = bigram(bigram_count, unigram_count)
    unigram_sentence = gen_unigram_sentence(10, prob_unigram)
    print "%s: Unigram generated sentence:\t%s" % (run_type, " ".join(unigram_sentence))

    bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words)
    print "%s: Bigram generated sentence:\t%s" % (run_type, " ".join(bigram_sentence))

    seed = "simply radiates star-power potential"
    bigram_sentence = gen_bigram_sentence(10, prob_bigram, next_words, seed=seed)
    print "%s: Bigram generated sentence with seed[%s] :\t%s" % (run_type, seed, " ".join(bigram_sentence))

    # next, do smoothing
    k_val_unigram, k_val_bigram = calculate_k_on_corpus(bigram_count, unigram_count, unigram_total_count, validation)

    k_min = [None, None]
    k_min_k = [None, None]
    k_min_perpl = [None, None]
    k_min_perpl_k = [None, None]

    for x in k_val_unigram:
        # print x, k_val[x]

        if k_min[0] is None or k_min[0]> k_val_unigram[x][0]:
            k_min[0] = k_val_unigram[x][0]
            k_min_k[0] = x
        if k_min_perpl[0] is None or k_min_perpl[0] > k_val_unigram[x][1]:
            k_min_perpl[0] = k_val_unigram[x][1]
            k_min_perpl_k[0] = x

        if k_min[1] is None or k_min[1]> k_val_bigram[x][0]:
            k_min[1] = k_val_bigram[x][0]
            k_min_k[1] = x
        if k_min_perpl[1] is None or k_min_perpl[1] > k_val_bigram[x][1]:
            k_min_perpl[1] = k_val_bigram[x][1]
            k_min_perpl_k[1] = x
    print "%s: Bigram prob smoothed: %s %s, perpexity: %s %s" % (run_type, k_min, k_min_k, k_min_perpl, k_min_perpl_k)

    return k_min_k

def get_prob_of_unigram(item, prob_unigram_smooth):
    if item in prob_unigram_smooth:
        return -math.log(prob_unigram_smooth[item])
    else:
        return -math.log(prob_unigram_smooth['<unk>'])

def get_prob_of_bigram(first, second, len_split, prob_bigram_smooth, unigram_count, k):
    tup = (first, second)
    # first step; figure out if we are dealing with UNK or UNSEEN
    # we do this by checking the t-1 element; if it's in our unigram corpus, then it's UNSEEN otherwise, UNK
    is_unseen = False
    if (first in unigram_count and second in unigram_count):
        is_unseen = True
        unseen_prob = k / (unigram_count[first] * 1.0 + len_split * 1.0 * k)

    if tup in prob_bigram_smooth:
        return -math.log(prob_bigram_smooth[tup])

    # not found; if it's merely unseen, then add the basic probability
    if is_unseen:
        return -math.log(unseen_prob)

    tup = ("<unk>", second)
    if tup in prob_bigram_smooth:
        return -math.log(prob_bigram_smooth[tup])

    tup = (first, "<unk>")
    if tup in prob_bigram_smooth:
        return -math.log(prob_bigram_smooth[tup])

    tup = ("<unk>", "<unk>")
    return -math.log(prob_bigram_smooth[tup])

def predict_sentiment(k_pos, k_neg):
    pos_unigram_count, pos_bigram_count, pos_unigram_total_count, pos_next_words = add_start_end_tokens(pos_train)
    neg_unigram_count, neg_bigram_count, neg_unigram_total_count, neg_next_words = add_start_end_tokens(neg_train)

    define_unk(pos_unigram_count, pos_bigram_count, pos_next_words)
    define_unk(neg_unigram_count, neg_bigram_count, neg_next_words)

    #best way to do this is, given a sentence, calculate the probability for it on a positive and negative model

    pos_prob_unigram_smooth = add_plus_k_smoothing_unigram(pos_unigram_count, pos_unigram_total_count, k_pos[0])
    pos_prob_bigram_smooth = add_plus_k_smoothing_bigram(pos_bigram_count, pos_unigram_count, k_pos[1])

    neg_prob_unigram_smooth = add_plus_k_smoothing_unigram(neg_unigram_count, neg_unigram_total_count, k_neg[0])
    neg_prob_bigram_smooth = add_plus_k_smoothing_bigram(neg_bigram_count, neg_unigram_count, k_neg[1])

    sum_pos_uni = 0
    sum_pos_bi = 0
    sum_neg_uni = 0
    sum_neg_bi = 0
    #evaluate on pos validation data first
    f = open(pos_val, 'r')
    for idx,line in enumerate(f.readlines()):
        split = clean_data_and_split(line, True)

        pos_evaluated_prob_uni = 0
        pos_evaluated_prob_bi = 0
        neg_evaluated_prob_uni = 0
        neg_evaluated_prob_bi = 0

        len_split = len(split)

        for idx2, item in enumerate(split):
            pos_evaluated_prob_uni += get_prob_of_unigram(item, pos_prob_unigram_smooth)
            neg_evaluated_prob_uni += get_prob_of_unigram(item, neg_prob_unigram_smooth)
            if idx2 > 1:
                pos_evaluated_prob_bi += get_prob_of_bigram(split[idx2 - 1], split[idx2], len(split),
                                                         pos_prob_bigram_smooth, pos_unigram_count, k_pos[1])
                neg_evaluated_prob_bi += get_prob_of_bigram(split[idx2 - 1], split[idx2], len(split),
                                                         neg_prob_bigram_smooth, neg_unigram_count, k_neg[1])
        #evaluate final output on perplexity
        if math.exp(pos_evaluated_prob_uni/len(split)) > math.exp(neg_evaluated_prob_uni/len(split)):
            sum_pos_uni+=1
        else:
            sum_neg_uni+=1

        if math.exp(pos_evaluated_prob_bi/len(split)) > math.exp(neg_evaluated_prob_bi/len(split)):
            sum_pos_bi += 1
        else:
            sum_neg_bi += 1
    print "Expected all sum_pos_uni, but got %d sum_pos_uni and %d sum_neg_uni" % (sum_pos_uni, sum_neg_uni)
    print "Expected all sum_pos_bi, but got %d sum_pos_bi and %d sum_neg_bi" % (sum_pos_bi, sum_neg_bi)
    f.close()

    sum_pos_uni = 0
    sum_pos_bi = 0
    sum_neg_uni = 0
    sum_neg_bi = 0
    # evaluate on neg validation data first
    f = open(neg_val, 'r')
    for idx,line in enumerate(f.readlines()):
        split = clean_data_and_split(line, True)

        pos_evaluated_prob_uni = 0
        pos_evaluated_prob_bi = 0
        neg_evaluated_prob_uni = 0
        neg_evaluated_prob_bi = 0

        len_split = len(split)

        for idx2, item in enumerate(split):
            pos_evaluated_prob_uni += get_prob_of_unigram(item, pos_prob_unigram_smooth)
            neg_evaluated_prob_uni += get_prob_of_unigram(item, neg_prob_unigram_smooth)
            if idx2 > 1:
                pos_evaluated_prob_bi += get_prob_of_bigram(split[idx2 - 1], split[idx2], len_split,
                                                         pos_prob_bigram_smooth, pos_unigram_count, k_pos[1])
                neg_evaluated_prob_bi += get_prob_of_bigram(split[idx2 - 1], split[idx2], len_split,
                                                         neg_prob_bigram_smooth, neg_unigram_count, k_neg[1])
        if math.exp(pos_evaluated_prob_uni/len(split)) > math.exp(neg_evaluated_prob_uni/len(split)):
            sum_pos_uni+=1
        else:
            sum_neg_uni+=1

        if math.exp(pos_evaluated_prob_bi/len(split)) > math.exp(neg_evaluated_prob_bi / len(split)):
            sum_pos_bi += 1
        else:
            sum_neg_bi += 1
    print "Expected all sum_neg_uni, but got %d sum_pos_uni and %d sum_neg_uni" % (sum_pos_uni, sum_neg_uni)
    print "Expected all sum_neg_bi, but got %d sum_pos_bi and %d sum_neg_bi" % (sum_pos_bi, sum_neg_bi)
    f.close()

    #first, ingest all test data
    f = open(test_data, 'r')

    output = []

    for idx,line in enumerate(f.readlines()):
        split = clean_data_and_split(line, True)

        pos_evaluated_prob = 0
        neg_evaluated_prob = 0

        len_split = len(split)

        #first, do unigram evaluation on the line
        for idx2,item in enumerate(split):
            pos_evaluated_prob += get_prob_of_unigram(item, pos_prob_unigram_smooth)
            neg_evaluated_prob += get_prob_of_unigram(item, neg_prob_unigram_smooth)

            #then do for bigram
            # if idx2 > 1:
            #     pos_evaluated_prob += \
            #         get_prob_of_bigram(split[idx2-1], split[idx2], len_split, pos_prob_bigram_smooth, pos_unigram_count, k_pos[1])
            #     neg_evaluated_prob += \
            #         get_prob_of_bigram(split[idx2-1], split[idx2], len_split, neg_prob_bigram_smooth, neg_unigram_count, k_neg[1])


        evaluated = {}
        evaluated['Id'] = idx
        if pos_evaluated_prob < neg_evaluated_prob:
            evaluated['Prediction'] = 1
        else:
            evaluated['Prediction'] = 0
        output.append(evaluated)
    print "*" * 80

    with open('sentiment_output_bigram.csv', 'wb+') as f:
        writer = csv.writer(f, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Prediction'])
        for entry in output:
            writer.writerow([entry['Id']+1, entry['Prediction']])

    f.close()

# count = 0
#
# for i in unigram_count:
#     print "Unigram prob: %s %s" % (i, prob_unigram[i])
#     count+=1
#     if count is 10:
#         break

k_pos = test_on_corpus('pos')
k_neg = test_on_corpus('neg')

predict_sentiment(k_pos, k_neg)
# predict_sentiment([1, 1], [1, 1])

print("--- %s seconds ---" % (time.time() - start_time))
