import os
import operator
import numpy as np
import math
import numpy as np

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
    
def perplexity(uniprob, biprob, total_token_count):
    curr = 0
    for i in uniprob:
        curr += -math.log(uniprob[i])
    unipp = math.exp((1 / total_token_count) * curr)
    curr = 0
    for i in biprob:
        curr += -math.log(biprob[i])
    bipp = math.exp((1/total_token_count)*curr)

    return unipp, bipp
    
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

def evalbi(line, unigram, bigram):
    x = 1
    words = ['<s>']
    words.extend(line.split(" "))
    words.append('</s>')
    for i, w in enumerate(words):
        if unigram.get(w, '') == '':
            words[i] = '<unk>'
    for idx, word in enumerate(words[:-1]):
            x += math.log(bigram[words[idx+1]+'|'+word])
    return math.exp(x)

#open file
filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Train\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Train\\"+"neg.txt", 'r')

#get the ngrams counts
unipos, bipos, total_token_count_pos, possible_next_words_pos = ngram(fp)
unineg, bineg, total_token_count_neg, possible_next_words_neg = ngram(fn)

fp.close()
fn.close()

#Replace unks
unipos, bipos, possible_next_words_pos = substitute_unks(unipos, bipos, possible_next_words_pos)
unineg, bineg, possible_next_words_neg = substitute_unks(unineg, bineg, possible_next_words_neg)

#generate unseen bigrams
# unipos, bipos, possible_next_words_pos = gen_unseen_bigrams(unipos, bipos, possible_next_words_pos)
# unineg, bineg, possible_next_words_neg = gen_unseen_bigrams(unineg, bineg, possible_next_words_neg)

fp = filepath + "\SentimentDataset\Dev\pos.txt"
fn = filepath + "\SentimentDataset\Dev\\"+"neg.txt"

#find k for add k smoothing
kpos = 0.1
kneg = 0.1

#get probabilities
unipos, bipos = get_probs(unipos, bipos, total_token_count_pos, kpos)
unineg, bineg = get_probs(unineg, bineg, total_token_count_neg, kneg)

# ppposu, ppposb = perplexity(unipos, bipos, total_token_count_pos)
# ppnegu, ppnegb = perplexity(unineg, bineg, total_token_count_neg)
#
# print("Perplexity Positive Unigram: ", ppposu)
# print("Perplexity Positive Bigram: ", ppposb)
# print("Perplexity Negative Unigram: ", ppnegu)
# print("Perplexity Negative Bigram: ", ppnegb)

# #Set number of sentences to generate
# num_sentences = 3
# #Set Upper bound on the length of the sentences to generate
# length_bound = 10
#
# #Print Positive Unigram Sentences
# print()
# print ("Positive Unigram Sentences:")
# print()
# for sentence in unigram_sentence_generator(num_sentences, length_bound, unipos):
#     print(sentence)
#
# #Print Negative Unigram Sentences
# print()
# print ("Negative Unigram Sentences:")
# print()
# for sentence in unigram_sentence_generator(num_sentences, length_bound, unineg):
#     print(sentence)
#
# #Print Positive Bigram Sentences
# print()
# print ("Positive Bigram Sentences:")
# print()
# for sentence in bigram_sentence_generator(num_sentences, length_bound, bipos, possible_next_words_pos):
#     print(sentence)
#
# #Print Negative Bigram Sentences
# print()
# print ("Negative Bigram Sentences:")
# print()
# for sentence in bigram_sentence_generator(num_sentences, length_bound, bineg, possible_next_words_neg):
#     print(sentence)
#
# #Print Positive Bigram Sentences with seeding
#
# print()
# print ("Positive Bigram Sentences with seeding:")
# print()
# for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "The movie was"):
#     print(sentence)
# for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "I am"):
#     print(sentence)
# for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "The film"):
#     print(sentence)
#
# #Print Negative Bigram Sentences with seeding
# print()
# print ("Negative Bigram Sentences with seeding:")
# print()
# for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "The movie was"):
#     print(sentence)
# for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "I am"):
#     print(sentence)
# for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "The film"):
#     print(sentence)

ft = open(filepath + "\SentimentDataset\Test\\"+"test.txt", 'r')
id = []
pred = []
for idx, line in enumerate(ft):
    id.append(idx+1)
    if evalbi(line, unipos, bipos) < evalbi(line, unineg, bineg):
        pred.append(1)
    else:
        pred.append(0)
np.savetxt("unipred.csv", np.column_stack((id, pred)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')