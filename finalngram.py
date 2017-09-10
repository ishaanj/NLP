import os
import operator
import numpy as np
    
#make uni and bigram
def ngram (fs):
    unigram = {}
    bigram = {}
    possible_next_words = {}
    
    #calculate counts
    total_token_count = 0
    for line in fs:
        line = line.lstrip(".-_,")
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
    
    #calculate probabilities
    for i in bigram:
        bigram[i] /= unigram[i.split("|")[1]]
    for i in unigram:
        unigram[i] /= total_token_count

    return unigram, bigram, possible_next_words

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
            while(tokens[0] == "<s>"):
                tokens = np.random.choice(unigram_tokens, 1, True, unigram_probs)
            if(tokens[0] == "</s>"):
                break
            sentence.append(tokens[0])
        yield " ".join(sentence)

#generate bigram sentences        
def bigram_sentence_generator(num_sentences, length_bound, biprob, possible_next_words):
    for i in range(num_sentences):
        current_word = "<s>"
        sentence = [current_word]
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
    

#TODO: random sentence generation
# Use . as </s> (done)
# def randuni(uni): (done along with probs)
# def randbi(bi): (done along with probs)

#open file
filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Train\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Train\\"+"neg.txt", 'r')

#get the ngrams
unipos, bipos, possible_next_words_pos = ngram(fp)
unineg, bineg, possible_next_words_neg = ngram(fn)

#Set number of sentences to generate
num_sentences = 3
#Set Upper bound on the length of the sentences to generate
length_bound = 10

#Print Positive Unigram Sentences
print()
print ("Positive Unigram Sentences:")
print()
for sentence in unigram_sentence_generator(num_sentences, length_bound, unipos):
    print(sentence)

#Print Negative Unigram Sentences
print()
print ("Negative Unigram Sentences:")
print()
for sentence in unigram_sentence_generator(num_sentences, length_bound, unineg):
    print(sentence)
    
#Print Positive Bigram Sentences
print()
print ("Positive Bigram Sentences:")
print()
for sentence in bigram_sentence_generator(num_sentences, length_bound, bipos, possible_next_words_pos):
    print(sentence)

#Print Negative Bigram Sentences
print()
print ("Negative Bigram Sentences:")
print()
for sentence in bigram_sentence_generator(num_sentences, length_bound, bineg, possible_next_words_neg):
    print(sentence)