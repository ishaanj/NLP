from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'
dev_dir =cur_dir + '\\SentimentDataset\\Dev\\'
test_dir =cur_dir + '\\SentimentDataset\\Test\\'

pos_train = train_dir + "pos.txt"
neg_train = train_dir + "neg.txt"
test_train = test_dir + "test.txt"

pos_file = open(pos_train, 'r')
neg_file = open(neg_train, 'r')
test_file = open(test_train, 'r')
# print(pos_file)

word_types = []
def split2words(f):
    for line in f:
        line = line.strip()
        line = line.rstrip(".,;-!?")
        line = line.lstrip(".,;-!?")
        word_types.append(line.split(" "))
split2words(pos_file)
split2words(neg_file)
split2words(test_file)

sizeOfModel = 100
# model = word2vec.Word2Vec(word_types, size=sizeOfModel, window=5, min_count=1, workers=4)
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)
# model.save(kk)
# model = word2vec.Word2Vec.load(kk)
b = [] # feature vectors of training data
c = [] # labels for training data for both pos and neg
d = [] # feature vectors of test data
for idx, line in enumerate(word_types):
    a = []
    for word in line:
        if word not in model:
            continue
        a.append(model[word])
    average_val = [sum(i) / len(a) for i in zip(*a)]

    if len(average_val) == 300:
        if idx >= 6559:
            d.append(average_val)
            continue
        if idx + 1 <= 3442:
            c.append(0)
        else:
            c.append(1)

        b.append(average_val)
    else:
        print("%s : %s" %(len(average_val),line))

# my_var = GradientBoostingClassifier(verbose=True)
my_var = svm.SVC(kernel="rbf",verbose=True)
my_var.fit(b,c)
e = my_var.predict(d)
import numpy as np

np.savetxt("myCsv.csv", np.column_stack((range(1,1748), e)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')
# myCsv = open("myCsv.csv",'w')
# myCsv.write("id,prediction\n")
# for idx in range(len(e)):
#     myCsv.write("%s,%s\n" %(idx+1,e[idx]))