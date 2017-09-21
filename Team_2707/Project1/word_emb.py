from gensim.models import word2vec
import os
from sklearn import svm

cur_dir = os.path.abspath(os.path.curdir)
train_dir =cur_dir + '\\SentimentDataset\\Train\\'
dev_dir =cur_dir + '\\SentimentDataset\\Dev\\'
test_dir =cur_dir + '\\SentimentDataset\\Test\\'

pos_train = train_dir + "pos.txt"
neg_train = train_dir + "neg.txt"
test_train = test_dir + "test.txt"

pos_file = open(pos_train, 'r')
neg_file = open(pos_train, 'r')
test_file = open(test_train, 'r')
print(pos_file)

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
model = word2vec.Word2Vec(word_types, size=sizeOfModel, window=5, min_count=1, workers=4)
# model.save(kk)
# model = word2vec.Word2Vec.load(kk)
b = [] # feature vectors of training data
c = [] # labels for training data for both pos and neg
d = [] # feature vectors of test data
for idx, line in enumerate(word_types):
    a = []
    for word in line:
        a.append(model.wv[word])
    average_val = [sum(i) / len(a) for i in zip(*a)]
    b.append(average_val)
    if idx + 1 <= 3442:
        c.append(0)
    else:
        c.append(1)
    if idx >= 6559:
        d.append(average_val)

my_var = svm.LinearSVC(verbose=True)
my_var.fit(b,c)

e = my_var.predict(d)
pass