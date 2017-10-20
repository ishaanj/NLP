import sklearn
import sklearn_crfsuite

def process_training_file(fname):
    train_sents = []
    with open(fname) as fp:

        token_line = fp.readline()
        while token_line:
            train_sentence = []
            pos_line = fp.readline()
            ner_line = fp.readline()


            tokens = token_line.strip().split()
            pos_tags = pos_line.strip().split()
            ner_tags = ner_line.strip().split()
            for i in range(len(tokens)):
                train_sentence.append((tokens[i], pos_tags[i], ner_tags[i]))

            train_sents.append(train_sentence)
            token_line = fp.readline()

    return train_sents


def process_test_file(fname):
    test_sents = []
    with open(fname) as fp:

        token_line = fp.readline()
        while token_line:
            test_sentence = []
            pos_line = fp.readline()
            token_indices_line = fp.readline()

            tokens = token_line.strip().split()
            pos_tags = pos_line.strip().split()
            token_indices = token_indices_line.strip().split()
            for i in range(len(tokens)):
                test_sentence.append((tokens[i], pos_tags[i], token_indices[i]))

            test_sents.append(test_sentence)

            token_line = fp.readline()

    return test_sents


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


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
    with open("results.csv", "w") as fp:
        fp.write("Type,Prediction\n")
        fp.write(tag_type_dict["ORG"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["MISC"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["PER"].rstrip())
        fp.write("\n")
        fp.write(tag_type_dict["LOC"].rstrip())


#Process the training file
train_sents = process_training_file("train.txt")

#Process the test file
test_sents = process_test_file("test.txt")

#convert to features
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]

#build model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

#split into train and validation
trainvalx = X_train[:int(len(X_train)*0.9)]
trainvaly = y_train[:int(len(y_train)*0.9)]
trainvalx2 = X_train[int(len(X_train)*0.9):]
trainvaly2 = y_train[int(len(y_train)*0.9):]

#predict
crf.fit(trainvalx, trainvaly)
y_pred = crf.predict(trainvalx2)

#check incorrect labels
for idx, i in enumerate(trainvaly2):
    if i != y_pred[idx]:
        print("Word: ", train_sents[idx+len(trainvalx)])
        print("Predicted: ", y_pred[idx])
        print("Correct: ", i)
        print("Index: ", idx+len(trainvaly))

#test predictions
# test_ner_tags = [pred for sent in y_pred for pred in sent]
#
# print(test_ner_tags[101:114])

#Generate CSV
# generateCSV(test_ner_tags)