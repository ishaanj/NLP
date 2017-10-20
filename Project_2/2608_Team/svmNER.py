import logging
import numpy as np
import os
# from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import cross_val_score
from numpy import genfromtxt
from sklearn.externals import joblib
import math

#Process input and store it into respective lists
def process_training_file(fname):
    tokens = []
    pos_tags = []
    ner_tags = []
    with open(fname) as fp:

        token_line = fp.readline()
        while token_line:
            pos_line = fp.readline()
            ner_line = fp.readline()

            tokens.extend(token_line.strip().split())
            pos_tags.extend(pos_line.strip().split())
            ner_tags.extend(ner_line.strip().split())

            token_line = fp.readline()
    return tokens, pos_tags, ner_tags

#build pos tag list
def buildPos(trainpos):
    posTags = []
    for i in trainpos:
        if i not in posTags:
            posTags.append(i)
    return posTags

#convert tag into 0-1 feature
def posToFeature(tag):
    if posTags.index(tag) >= 0:
        return posTags.index(tag)*1.0/len(posTags)
    else:
        return 0

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
                # line = line + " </s>"
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
    print("Best k_value: %s sum negative log prob: %s" % (mxk, mnprob))

    return mxk, mnprob, total_dev_token_count, unigram_curprob


#write prediction to csv
def generateCSV(test_ner_tags):
    i = 0
    tag_type_dict = {"ORG": "ORG,", "PER": "PER,", "LOC": "LOC,", "MISC": "MISC,"}
    while i < len(test_ner_tags):
        if test_ner_tags[i][0] == "B":
            tag_type = test_ner_tags[i][2:]
            j = i+1
            while j < len(test_ner_tags) and (("I-"+tag_type) == test_ner_tags[j]):
                j += 1
            tag_type_dict[tag_type] += str(i) + "-" + str(j-1) + " "
            i = j
        else:
            i += 1

    # Write to the CSV file
    with open("resultsvm.csv", "w") as fp:
        fp.write("Type,Prediction\n")
        fp.write(tag_type_dict["ORG"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["MISC"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["PER"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["LOC"].rstrip())


#process input
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
traintokens, trainpos, trainner = process_training_file('train.txt')
testtokens, testpos, testids = process_training_file('test.txt')

#load word2vec model
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#build ner and pos tag lists
ner_types = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
posTags = buildPos(trainpos)

#create input vectors from words
x = []
y = []
xtest = []

# for i, t in enumerate(traintokens):
#     print(i)
#     a = []
#     if t in model:
#         a.append(model[t])
#         if i-1 >= 0 and traintokens[i-1] in model:
#             a.append(model[traintokens[i-1]])
#     else:
#         a.append([0 for i in range(300)])
#         if i - 1 >= 0 and traintokens[i - 1] in model:
#             a.append(model[traintokens[i - 1]])
#     a = [sum(idx) / len(a) for idx in zip(*a)]
#     a.append(posToFeature(trainpos[i]))
#     x.append(a)
#     y.append(ner_types.index(trainner[i]))
#
# np.savetxt("custtest.csv", np.column_stack((x, y)), delimiter=",", fmt='%s', comments='')

#write input vectors to csv for easy read
xtrain = genfromtxt('custtest.csv', delimiter=',')
for i in xtrain:
    x.append(i[:-1])
    y.append(i[-1])

#generate test vector set
# for i, t in enumerate(testtokens):
#     print(i)
#     a = []
#     if t in model:
#         a.append(model[t])
#         if i - 1 >= 0 and testtokens[i - 1] in model:
#             a.append(model[testtokens[i - 1]])
#     else:
#         a.append([0 for i in range(300)])
#         if i - 1 >= 0 and testtokens[i - 1] in model:
#             a.append(model[testtokens[i - 1]])
#     a = [sum(idx) / len(a) for idx in zip(*a)]
#     a.append(posToFeature(testpos[i]))
#     xtest.append(a)
# np.savetxt("custtest2.csv", xtest, delimiter=",", fmt='%s', comments='')

xtest = genfromtxt('custtest2.csv', delimiter=',')
#Save and load SVM model
# print("Built test and train")
#
# msvc = svm.LinearSVC(verbose=True)
# msvc.fit(x,y)
#
# joblib.dump(msvc, 'linsvm.pkl')
# print("Saved model")
msvc = joblib.load('linsvm.pkl')
print("Loaded Model")

# predict and write to file
ytest = msvc.predict(xtest)
yner = []
print('Predicted')
print(ytest[:10])
for i, _ in enumerate(ytest):
    yner.append(ner_types[int(ytest[i])])
generateCSV(yner)

