import os
import operator

#make uni and bigram
def ngram(s):
    unigram = {}
    bigram = {}

    #calculate counts
    for i in s:
        if unigram.get(i, "") == "":
            unigram[i] = 1.0
        else:
            unigram[i] += 1.0
    for idx,_ in enumerate(s[1:]):
        big = s[idx]+"|"+s[idx-1]
        if bigram.get(big, "") == "":
            bigram[big] = 1.0
        else:
            bigram[big] += 1.0

    #calculate probabilities
    for i in bigram:
        bigram[i] /= unigram[i.split("|")[1]]
    for i in unigram:
        unigram[i] /= len(unigram)

    return unigram, bigram

#TODO: random sentence generation
# Use . as </s>
# def randuni(uni):
# def randbi(bi):

#open file
filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Dev\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Dev\\"+"neg.txt", 'r')

#read corpora
pos = fp.read()
pos = pos.split()
neg = fn.read()
neg = neg.split()

#get the ngrams
unipos, bipos = ngram(pos)
unineg, bineg = ngram(neg)

#Sort the ngrams
unipos = sorted(unipos.items(), key=operator.itemgetter(1), reverse = True)
bipos = sorted(bipos.items(), key=operator.itemgetter(1), reverse = True)
unineg = sorted(unineg.items(), key=operator.itemgetter(1), reverse = True)
bineg = sorted(bineg.items(), key=operator.itemgetter(1), reverse = True)

#print for sanity
print(unipos)
print(bipos)
print(unineg)
print(bineg)

