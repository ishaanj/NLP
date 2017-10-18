import os
import random

def readData(f):
    ctr = 0
    trainSet = []
    trainPosSet = []
    trainNerSet = []
    for line in f:
        tokens = line.split()
        if ctr == 0:
            trainSet.append(tokens)
        elif ctr == 1:
            trainPosSet.append(tokens)
        elif ctr == 2:
            trainNerSet.append(tokens)
            ctr = 0
            continue
        ctr += 1
    return trainSet, trainPosSet, trainNerSet

def evalNER(set, pos):
    predNER = []
    for idx, line in enumerate(set):
        for jdx, word in enumerate(line):
            if 'NNP' in pos[idx][jdx] and word[0].isupper():
                if idx+jdx-1 >= 0 and ('B' in predNER[idx + jdx - 1] or 'I' in predNER[idx + jdx - 1]):
                    predNER.append('I-' + predNER[idx + jdx - 1][2:])
                else:
                    x = random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
                    if x == 1:
                        predNER.append('B-PER')
                    elif x == 2:
                        predNER.append('B-LOC')
                    elif x == 3:
                        predNER.append('B-MISC')
                    else:
                        predNER.append('B-ORG')
            else:
                predNER.append('O')
    return predNER

def acc(pred, ner):
    total = 0
    corr = 0
    for idx, token in enumerate(ner):
        total += 1
        if pred[idx] == token:
            corr += 1

    print("Total tokens: ", total)
    print("Correct tokens: ", corr)
    print("Accuracy: ", corr * 100 / total, '%')

filepath = os.path.abspath(".")
ftrain = open(filepath + "\Data\\"+"train.txt", 'r')
ftest = open(filepath + "\Data\\"+"test.txt", 'r')
valSplit = 0.9

trainSet, trainPOS, trainNER = readData(ftrain)
testSet, testPOS, testIDs = readData(ftest)
testIDs = [j for i in testIDs for j in i]
# valSet = trainSet[int(len(trainSet)*0.9):]
# valPOS = trainPOS[int(len(trainPOS)*0.9):]
# valNER = trainNER[int(len(trainNER)*0.9):]
# trainNER = [j for i in trainNER for j in i]
# valNER = [j for i in valNER for j in i]
# org = 0
# loc = 0
# per = 0
# misc = 0
# for ner in trainNER:
#     if 'ORG' in ner:
#         org += 1
#     elif 'LOC' in ner:
#         loc += 1
#     elif 'PER' in ner:
#         per += 1
#     elif 'MISC' in ner:
#         misc += 1
#
# print("Org ", org)
# print("Loc ", loc)
# print("Per ", per)
# print("Misc ", misc)

predtestNER = evalNER(testSet, testPOS)
cx = 0
org = []
loc = []
per = []
for i in testIDs:
    if 'B' in predtestNER[i]:
        if 'ORG' in predtestNER[i]:
            org.append(''+i+'-')

# predvalNER = evalNER(valSet, valPOS)

# print("Training")
# acc(predtestNER, trainNER)
# print()
# print("Validation")
# acc(predvalNER, valNER)
