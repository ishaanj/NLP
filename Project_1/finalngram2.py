import os
import operator
import numpy as np
import math
import numpy as np
import sklearn.metrics as metrics

#make uni and bigram
def ngram (fs):
    unigram = {}
    bigram = {}
    possible_next_words = {}
    
    #calculate counts
    total_token_count = 0
    for line in fs:
        line = line.strip()
        line = line.lstrip(".-_,")
        line = line.rstrip(".?!")
        line = "<s> " + line + " </s>"
        listtokens = line.split()
        for token in listtokens:
            total_token_count += 1
            if token in unigram:
                unigram[token] += 1.0
            else:
                unigram[token] = 1.0
                
        for idx in range(1,len(listtokens)):
            big = listtokens[idx]+"|"+listtokens[idx-1]
            if big in bigram:
                bigram[big] += 1.0
            else:
                bigram[big] = 1.0
            if listtokens[idx-1] not in possible_next_words:
                possible_next_words[listtokens[idx-1]] = set()
            possible_next_words[listtokens[idx-1]].add(listtokens[idx])
    
    return unigram, bigram, total_token_count, possible_next_words

def gen_unseen_bigrams(unigrams, bigrams, possible_next_words):
    unigram_tokens = list(unigrams.keys())
    #unigram_tokens.extend(["<s>", "</s>"])
    for i in unigram_tokens:
        for j in unigram_tokens:
            c = i+"|"+j
            if c not in bigrams:
                bigrams[c] = 0
    for i in unigram_tokens:
        for token in possible_next_words:
            if i not in possible_next_words[token]:
                possible_next_words[token].add(i)
    return unigrams, bigrams, possible_next_words

def get_probs(unigram, bigram, total_token_count, k_value = 0):
    #calculate probabilities
    unigram_prob = dict(unigram)
    bigram_prob = dict(bigram)
    vocab_size = len(unigram)
    for i in bigram:
        # if bigram[i] == 0:
        #     bigram[i] += 0.000027
        # elif bigram[i] == 1:
        #     bigram[i] -= 0.5
        # else:
        #     bigram[i] -= 0.75
        bigram_prob[i] = (bigram[i] + k_value) / (unigram[i.split("|")[1]] + (k_value * vocab_size))
    for i in unigram:
        unigram_prob[i] /= total_token_count

    return unigram_prob, bigram_prob
    
def findk(fs, k_values, unigram, bigram, total_token_count, ispos=True):
    mnprob = 1000000000
    mxk = 0
    total_dev_token_count = 0
    for k_value in k_values:
        vocab_size = len(unigram)
        unigram_prob, bigram_prob = get_probs(unigram, bigram, total_token_count, k_value)
        curprob = 0
        unigram_curprob = 0
        with open(fs, "r") as f:
            current_token = "<s>"
            for line in f:
                line = line.strip()
                line = line.lstrip(".-_,")
                line = line.rstrip(".?!")
                #line = line + " </s>"
                listtokens = line.split()
                for token in listtokens:
                    total_dev_token_count += 1
                    tkn = token
                    if token not in unigram:
                        tkn = "<unk>"
                    unigram_curprob += -math.log(unigram_prob[tkn])
                    if tkn + "|" + current_token in bigram_prob:
                        curprob += -math.log(bigram_prob[tkn + "|" + current_token])
                    else:
                        prob = (k_value) / (unigram[current_token] + (k_value * vocab_size))
                        curprob += -math.log(prob)
                    current_token = tkn
            if "</s>" + "|" + current_token in bigram_prob:
                curprob += -math.log(bigram_prob["</s>" + "|" + current_token])
            else:
                prob = (k_value) / (unigram[current_token] + (k_value * vocab_size))
                curprob += -math.log(prob)
            # print("k_value: %s curprob: %s"%(k_value, curprob))
            if curprob < mnprob:
                mnprob = curprob
                mxk = k_value
    print()
    if ispos:
        print("Positive findk results:")
    else:
        print("Negative findk results:")
    print("Best k_value: %s sum negative log prob: %s"%(mxk, mnprob))
    
    return mxk, mnprob, total_dev_token_count, unigram_curprob
    
def perplexity(mnprob, total_dev_token_count, unigram_curprob, ispos=True):
    perplexity = math.exp(mnprob / total_dev_token_count)
    unigram_perplexity = math.exp(unigram_curprob / total_dev_token_count)
    print()
    if ispos:
        print("Positive Bigram Perplexity: ",perplexity)
        print("Positive Unigram Perplexity: ",unigram_perplexity)
    else:
        print("Negative Bigram Perplexity: ",perplexity)
        print("Negative Unigram Perplexity: ",unigram_perplexity)

    return perplexity
    
def substitute_unks(unigrams, bigrams, possible_next_words):
    unk_tokens = set()
    unk_count = 0
    
    #Replace unks in unigrams
    unigram_tokens = list(unigrams.keys())
    for token in unigram_tokens:
        if unigrams[token] == 1:
            unk_tokens.add(token)
            unk_count = unk_count+1
            del unigrams[token]
    
    unigrams["<unk>"] = unk_count
    
    #Replace unks in bigrams
    bigram_tokens = list(bigrams.keys())
    for token in bigram_tokens:
        post_token = token.split("|")[0]
        pre_token = token.split("|")[1]
        cnt = bigrams[token]
        if pre_token in unk_tokens and post_token in unk_tokens:
            del bigrams[token]
            if "<unk>|<unk>" not in bigrams:
                bigrams["<unk>|<unk>"] = 0
            bigrams["<unk>|<unk>"] += cnt
        elif pre_token in unk_tokens:
            del bigrams[token]
            if post_token + "|<unk>" not in bigrams:
                bigrams[post_token + "|<unk>"] = 0
            bigrams[post_token + "|<unk>"] += cnt
        elif post_token in unk_tokens:
            del bigrams[token]
            if "<unk>|" + pre_token not in bigrams:
                bigrams["<unk>|" + pre_token] = 0
            bigrams["<unk>|" + pre_token] += cnt
            
    #Replace unks in possible_next_words
    possible_next_words_tokens = list(possible_next_words.keys())
    for token in possible_next_words_tokens:
        token_next_words = list(possible_next_words[token])
        
        unk_token_found = False
        for next_word in token_next_words:
            if next_word in unk_tokens:
                unk_token_found = True
                possible_next_words[token].remove(next_word)
        if unk_token_found:
            possible_next_words[token].add("<unk>")
        
        if token in unk_tokens:
            if "<unk>" not in possible_next_words:
                possible_next_words["<unk>"] = set()
            possible_next_words["<unk>"].update(possible_next_words[token])
            del possible_next_words[token]
        
    return unigrams, bigrams, possible_next_words

#generate unigram sentences    
def unigram_sentence_generator(num_sentences, length_bound, uniprob):
    unigram_tokens = []
    unigram_probs = []
    for token in uniprob:
        unigram_tokens.append(token)
        unigram_probs.append(uniprob[token])
    for i in range(num_sentences):
        sentence = []
        for j in range(length_bound):
            tokens = np.random.choice(unigram_tokens, 1, True, unigram_probs)
            while(tokens[0] == "<s>" or tokens[0] == "</s>"):
                tokens = np.random.choice(unigram_tokens, 1, True, unigram_probs)
            #if(tokens[0] == "</s>"):
            #    break
            sentence.append(tokens[0])
        yield " ".join(sentence)

#generate bigram sentences        
def bigram_sentence_generator(num_sentences, length_bound, biprob, possible_next_words, seed_sentence = ""):
    for i in range(num_sentences):
        if seed_sentence == "":
            current_word = "<s>"
            sentence = [current_word]
        else:
            current_word = seed_sentence.split()[-1]
            sentence = seed_sentence.split()
        for j in range(length_bound):
            probable_tokens = list(possible_next_words[current_word])
            next_probabilities = []
            for i in range(len(probable_tokens)):
                next_probabilities.append(biprob[probable_tokens[i] + "|" + current_word])
            tokens = np.random.choice(probable_tokens, 1, True, next_probabilities)
            if(tokens[0] == "</s>"):
                break
            sentence.append(tokens[0])
            current_word = tokens[0]
        sentence.append("</s>")
        yield " ".join(sentence)

def evaluni(line, unigram):
    x = 1
    words = line.split(" ")
    for word in words:
        if unigram.get(word, "") == "":
            x += math.log(unigram['<unk>'])
        else:
            x += math.log(unigram[word])
    return math.exp(x)

def evalbi(listtokens, unigram, bigram, unigram_cnts, k_value):
    """x = 1
    words = ['<s>']
    words.extend(line.split(" "))
    words.append('</s>')
    for i, w in enumerate(words):
        if unigram.get(w, '') == '':
            words[i] = '<unk>'
    for idx, word in enumerate(words[:-1]):
            x += math.log(bigram[words[idx+1]+'|'+word])
    return math.exp(x)"""
    prob = 0
    current_token = "<s>"
    vocab_size = len(unigram)
    
    for token in listtokens:
        tkn = token
        if token not in unigram:
            tkn = "<unk>"
        if tkn + "|" + current_token in bigram:
            prob += -math.log(bigram[tkn + "|" + current_token])
        else:
            prob += -math.log((k_value) / (unigram_cnts[current_token] + (k_value * vocab_size)))
        current_token = tkn
    if "</s>" + "|" + current_token in bigram:
        prob += -math.log(bigram["</s>" + "|" + current_token])
    else:
        prob += -math.log((k_value) / (unigram_cnts[current_token] + (k_value * vocab_size)))
    
    return math.exp(prob)
    
def generate_sentences(unigram, bigram, total_token_count, possible_next_words, ispos=True):

    #calculate probabilities
    uniprob, biprob = get_probs(unigram, bigram, total_token_count)
    
    #Set number of sentences to generate
    num_sentences = 3
    #Set Upper bound on the length of the sentences to generate
    length_bound = 10
    
    #Print Unigram Sentences
    print()
    if ispos:
        print ("Positive Unigram Sentences:")
    else:
        print ("Negative Unigram Sentences:")
    print()
    for sentence in unigram_sentence_generator(num_sentences, length_bound, uniprob):
        print(sentence)

    #Print Bigram Sentences
    print()
    if ispos:
        print ("Positive Bigram Sentences:")
    else:
        print ("Negative Bigram Sentences:")
    print()
    for sentence in bigram_sentence_generator(num_sentences, length_bound, biprob, possible_next_words):
        print(sentence)

    #Print Positive Bigram Sentences with seeding

    print()
    if ispos:
        print ("Positive Bigram Sentences with seeding:")
    else:
        print ("Negative Bigram Sentences with seeding:")
    print()
    for sentence in bigram_sentence_generator(1, length_bound, biprob, possible_next_words, "The movie was"):
        print(sentence)
    for sentence in bigram_sentence_generator(1, length_bound, biprob, possible_next_words, "I am"):
        print(sentence)
    for sentence in bigram_sentence_generator(1, length_bound, biprob, possible_next_words, "The film"):
        print(sentence)
        
def classification_language_model(ft, unipos, bipos, unineg, bineg, kpos, kneg, total_token_count_pos, total_token_count_neg):
    id = []
    pred = []
    unipos_prob, bipos_prob = get_probs(unipos, bipos, total_token_count_pos, kpos)
    unineg_prob, bineg_prob = get_probs(unineg, bineg, total_token_count_neg, kneg)
    for idx, line in enumerate(ft):
        id.append(idx+1)
        
        line = line.strip()
        line = line.lstrip(".-_,")
        line = line.rstrip(".?!")
        listtokens = line.split()
        
        if evalbi(listtokens, unipos_prob, bipos_prob, unipos, kpos) < evalbi(listtokens, unineg_prob, bineg_prob, unineg, kneg):
            pred.append(1)
        else:
            pred.append(0)
    # np.savetxt("unipred.csv", np.column_stack((id, pred)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')
    return pred

#open file
filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Train\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Train\\"+"neg.txt", 'r')

#get the ngrams counts
unipos, bipos, total_token_count_pos, possible_next_words_pos = ngram(fp)
unineg, bineg, total_token_count_neg, possible_next_words_neg = ngram(fn)

fp.close()
fn.close()

generate_sentences(unipos, bipos, total_token_count_pos, possible_next_words_pos, True)
generate_sentences(unineg, bineg, total_token_count_neg, possible_next_words_neg, False)

#Replace unks
unipos, bipos, possible_next_words_pos = substitute_unks(unipos, bipos, possible_next_words_pos)
unineg, bineg, possible_next_words_neg = substitute_unks(unineg, bineg, possible_next_words_neg)

#generate unseen bigrams
# unipos, bipos, possible_next_words_pos = gen_unseen_bigrams(unipos, bipos, possible_next_words_pos)
# unineg, bineg, possible_next_words_neg = gen_unseen_bigrams(unineg, bineg, possible_next_words_neg)

# fp = filepath + "\SentimentDataset\Dev\pos.txt"
# fn = filepath + "\SentimentDataset\Dev\\"+"neg.txt"

#find k for add k smoothing
kpos_values = [0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
kneg_values = [0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

kpos, mnprob_pos, total_dev_token_count_pos, unipos_curprob = findk(fp, kpos_values, unipos, bipos, total_token_count_pos, True)
kneg, mnprob_neg, total_dev_token_count_neg, unineg_curprob = findk(fn, kneg_values, unineg, bineg, total_token_count_neg, False)

#Find Perplexity
# perplexity_pos = perplexity(mnprob_pos, total_dev_token_count_pos, unipos_curprob, True)
# perplexity_neg = perplexity(mnprob_neg, total_dev_token_count_neg, unineg_curprob, False)

# ft = open(filepath + "\SentimentDataset\Test\\"+"test.txt", 'r')

# ftp = open(filepath + "\SentimentDataset\Dev\\"+"pos.txt", 'r')
# ftn = open(filepath + "\SentimentDataset\Dev\\"+"neg.txt", 'r')
# predpos = classification_language_model(ftp, unipos, bipos, unineg, bineg, kpos, kneg, total_token_count_pos, total_token_count_neg)
# predneg = classification_language_model(ftn, unipos, bipos, unineg, bineg, kpos, kneg, total_token_count_pos, total_token_count_neg)

#get probabilities
#unipos, bipos = get_probs(unipos, bipos, total_token_count_pos, kpos)
#unineg, bineg = get_probs(unineg, bineg, total_token_count_neg, kneg)

# ppposu, ppposb = perplexity(unipos, bipos, total_token_count_pos)
# ppnegu, ppnegb = perplexity(unineg, bineg, total_token_count_neg)
#
# print("Perplexity Positive Unigram: ", ppposu)
# print("Perplexity Positive Bigram: ", ppposb)
# print("Perplexity Negative Unigram: ", ppnegu)
# print("Perplexity Negative Bigram: ", ppnegb)

# yval = [0 for _ in predpos]
# yval2 = [1 for _ in predneg]
# predpos.extend(predneg)
# yval.extend(yval2)

# ftp = open(filepath + "\SentimentDataset\Dev\\"+"pos.txt", 'r')
# ftn = open(filepath + "\SentimentDataset\Dev\\"+"neg.txt", 'r')
# yval = []
# pred = []
# for line in ftp:
#     if evaluni(line, unipos) > evaluni(line, unineg):
#         pred.append(0)
#     else:
#         pred.append(1)
#     yval.append(0)
# for line in ftn:
#     if evaluni(line, unipos) > evaluni(line, unineg):
#         pred.append(0)
#     else:
#         pred.append(1)
#     yval.append(1)
#
# print("\nAccuracy: ", metrics.accuracy_score(yval, pred))
# print("Classified: ", metrics.accuracy_score(yval, pred, normalize=False), " out of ", len(yval), " samples correctly")