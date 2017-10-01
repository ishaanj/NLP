import logging
import numpy as np
import os
from gensim.models import word2vec
from sklearn import svm

filepath = os.path.abspath(".")
fp = open(filepath + "\SentimentDataset\Train\pos.txt", 'r')
fn = open(filepath + "\SentimentDataset\Train\\"+"neg.txt", 'r')
ft = open(filepath + "\SentimentDataset\Test\\"+"test.txt", 'r')

lines = []
linespos = 3442

for line in fp:
    line = line.strip()
    line = line.lstrip(".-_,")
    line = line.rstrip(".?!")
    lines.append(line.split(' '))
for line in fn:
    line = line.strip()
    line = line.lstrip(".-_,")
    line = line.rstrip(".?!")
    lines.append(line.split(' '))
testlines = []
for line in ft:
    line = line.strip()
    line = line.lstrip(".-_,")
    line = line.rstrip(".?!")
    testlines.append(line.split(' '))
    lines.append(line.split(' '))

num_features = 10
min_count = 1
context = 10
downsample = 1e-3
workers = 4

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_name = "300features_50minwords_10context"

# print("Loading model...")
# model = word2vec.Word2Vec.load(model_name)
# print("Loaded model...")

print("Training model...")
model = word2vec.Word2Vec(lines, workers=workers, size=num_features, min_count = min_count, window = context, sample = downsample)
model.init_sims(replace=True)
print("Training complete")
model.save(model_name)

y = []
x = []
for idx, line in enumerate(lines):
    a = []
    for word in line:
        a.append(model[word])
    a = [sum(i)/len(line) for i in zip(*a)]
    x.append(a)
    if idx<linespos:
        y.append(0)
    else:
        y.append(1)
    if idx == len(lines)-len(testlines)-1:
        break

xtest = []
for line in testlines:
    a = []
    for word in line:
        a.append(model[word])
    a = [sum(i)/len(line) for i in zip(*a)]
    xtest.append(a)

gbc = svm.SVC(C=100, kernel='rbf', verbose=True)
gbc.fit(x,y)
ytest = gbc.predict(xtest)
ids = []
for idx, _ in enumerate(ytest):
    ids.append(idx+1)
np.savetxt("svmpred.csv", np.column_stack((ids, ytest)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')