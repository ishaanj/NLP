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
    
def get_probs(unigram, bigram, total_token_count):
    #calculate probabilities
    for i in bigram:
        bigram[i] /= unigram[i.split("|")[1]]
    for i in unigram:
        unigram[i] /= total_token_count

    return unigram, bigram
    
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
    

#TODO: random sentence generation
# Use . as </s> (done)
# def randuni(uni): (done along with probs)
# def randbi(bi): (done along with probs)

#open file
filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Train\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Train\\"+"neg.txt", 'r')

#get the ngrams counts
unipos, bipos, total_token_count_pos, possible_next_words_pos = ngram(fp)
unineg, bineg, total_token_count_neg, possible_next_words_neg = ngram(fn)

#Replace unks
unipos, bipos, possible_next_words_pos = substitute_unks(unipos, bipos, possible_next_words_pos)
unineg, bineg, possible_next_words_neg = substitute_unks(unineg, bineg, possible_next_words_neg)

#get probabilities
unipos, bipos = get_probs(unipos, bipos, total_token_count_pos)
unineg, bineg = get_probs(unineg, bineg, total_token_count_neg)


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
    
#Print Positive Bigram Sentences with seeding

print()
print ("Positive Bigram Sentences with seeding:")
print()
for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "The movie was"):
    print(sentence)
for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "I am"):
    print(sentence)
for sentence in bigram_sentence_generator(1, length_bound, bipos, possible_next_words_pos, "The film"):
    print(sentence)
    
#Print Negative Bigram Sentences with seeding
print()
print ("Negative Bigram Sentences with seeding:")
print()
for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "The movie was"):
    print(sentence)
for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "I am"):
    print(sentence)
for sentence in bigram_sentence_generator(1, length_bound, bineg, possible_next_words_neg, "The film"):
    print(sentence)