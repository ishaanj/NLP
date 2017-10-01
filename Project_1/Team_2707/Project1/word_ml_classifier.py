from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import numpy as np

cur_dir = os.path.abspath(os.path.curdir)
train_dir = cur_dir + '\\SentimentDataset\\Train\\'
dev_dir = cur_dir + '\\SentimentDataset\\Dev\\'
test_dir = cur_dir + '\\SentimentDataset\\Test\\'

pos_train = train_dir + "pos.txt"
neg_train = train_dir + "neg.txt"
test_train = test_dir + "test.txt"

pos_dev = dev_dir + "pos.txt"
neg_dev = dev_dir + "neg.txt"

pos_file = open(pos_train, 'r')
neg_file = open(neg_train, 'r')
test_file = open(test_train, 'r')

pos_dev_file = open(pos_dev, 'r')
neg_dev_file = open(neg_dev, 'r')
word_types = []
dev_types = []

def split2words(f, append2):
    for line in f:
        line = line.strip()
        line = line.rstrip(".,;-!?")
        line = line.lstrip(".,;-!?")
        append2.append(line.split(" "))


split2words(pos_file, word_types)
split2words(neg_file, word_types)
split2words(test_file, word_types)
split2words(pos_dev_file, dev_types)
split2words(neg_dev_file, dev_types)

sizeOfModel = 100
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)
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

x = []
y = []

for idx, line in enumerate(dev_types):
    a = []
    for word in line:
        if word not in model:
            continue
        a.append(model[word])
    average_val = [sum(i) / len(a) for i in zip(*a)]

    if len(average_val) == 300:
        if idx + 1 <= 317:
            y.append(0)
        else:
            y.append(1)
        x.append(average_val)


# my_var = GradientBoostingClassifier(verbose=True)
# my_var = svm.SVC(C=100,kernel="rbf",verbose=True)
# from sklearn.naive_bayes import GaussianNB
# my_var = GaussianNB()
from sklearn.svm import LinearSVC
my_var = LinearSVC()
my_var.fit(b,c)
e = my_var.predict(d)
dev_value = my_var.predict(x)
from sklearn.metrics import accuracy_score
print("*"*80)
print(accuracy_score(y, dev_value))

# np.savetxt("myCsv.csv", np.column_stack((range(1,1748), e)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')