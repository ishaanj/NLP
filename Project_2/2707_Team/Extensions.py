import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
import numpy as np

def tokenize(split, filename="train.txt"):
    """
    Tokenizes the train file and returns
    word : NER type dictionary
    :param filename: train.txt
    :return: all_tokens
    """
    f = open(filename, 'r')
    train_sents = []
    val_sents = []

    # [('Melbourne', 'NP', 'B-LOC'),
    #  ('(', 'Fpa', 'O'),
    #  ('Australia', 'NP', 'B-LOC'),
    #  (')', 'Fpt', 'O'),
    #  (',', 'Fc', 'O'),
    #  ('25', 'Z', 'O'),
    #  ('may', 'NC', 'O'),
    #  ('(', 'Fpa', 'O'),
    #  ('EFE', 'NC', 'B-ORG'),
    #  (')', 'Fpt', 'O'),
    #  ('.', 'Fp', 'O')]
    all_lines = f.readlines()
    for i in range(int(len(all_lines)/3 * split)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i+1].split()
        line_ner = all_lines[3*i+2].split()
        train_sents.append([(line_tokens[j],line_pos[j],line_ner[j]) for j in range(len(line_tokens))]) #line_ner[j] is actually not required here
    num_lines = len(all_lines) / 3
    for i in range(int(split*num_lines), int(num_lines)):
        line_tokens = all_lines[3 * i].split()
        line_pos = all_lines[3 * i + 1].split()
        ner_result = all_lines[3 * i + 2].split()
        val_sents.append([(line_tokens[j],line_pos[j],ner_result[j]) for j in range(len(line_tokens))])

    f.close()
    return train_sents,val_sents

def validate_NER(filename="test.txt"):
    """
        Predicts NER for the test data
        :param filename: test.txt
        :param output_csv: output the file in the  required format
        :return: None
        """

    f = open(filename, 'r')
    CORRECT_PER = 0
    CORRECT_ORG = 0
    CORRECT_LOC = 0
    CORRECT_MISC = 0
    CORRECT_O = 0
    INCORRECT_PER = 0
    INCORRECT_ORG = 0
    INCORRECT_LOC = 0
    INCORRECT_MISC = 0
    INCORRECT_O = 0
    all_lines = f.readlines()
    num_lines = len(all_lines) / 3

    for i in range(int(0.8*num_lines), int(num_lines)):
        line_tokens = all_lines[3 * i].split()
        line_pos = all_lines[3 * i + 1].split()
        ner_result = all_lines[3 * i + 2].split()

        # VALID_TOKENS = set(["PER","ORG","LOC","MISC"])
        VALID_TOKENS = set(["B-PER", "B-LOC", "B-ORG", "B-MISC"])
        tagged_tokens = HMM(line_tokens, count_NER, count_NER_NER, count_wordType_NER)
        j = 1
        # New Version
        while (j < len(tagged_tokens)):
            if tagged_tokens[j] in VALID_TOKENS:
                if (tagged_tokens[j] == 'B-PER'):
                    if ner_result[j] == 'B-PER':
                        CORRECT_PER += 1
                    else:
                        INCORRECT_PER += 1

                elif (tagged_tokens[j] == 'I-PER'):
                    if ner_result[j] == 'I-PER':
                        CORRECT_PER += 1
                    else:
                        INCORRECT_PER += 1

                elif (tagged_tokens[j] == 'B-LOC'):
                    if ner_result[j] == 'B-LOC':
                        CORRECT_LOC += 1
                    else:
                        INCORRECT_LOC += 1

                elif (tagged_tokens[j] == 'I-LOC'):
                    if ner_result[j] == 'I-LOC':
                        CORRECT_LOC += 1
                    else:
                        INCORRECT_LOC += 1

                elif (tagged_tokens[j] == 'B-ORG'):
                    if ner_result[j] == 'B-ORG':
                        CORRECT_ORG += 1
                    else:
                        INCORRECT_ORG += 1

                elif (tagged_tokens[j] == 'I-ORG'):
                    if ner_result[j] == 'I-ORG':
                        CORRECT_ORG += 1
                    else:
                        INCORRECT_ORG += 1

                elif (tagged_tokens[j] == 'B-MISC'):
                    if ner_result[j] == 'B-MISC':
                        CORRECT_MISC += 1
                    else:
                        INCORRECT_MISC += 1

                elif (tagged_tokens[j] == 'I-MISC'):
                    if ner_result[j] == 'I-MISC':
                        CORRECT_MISC += 1
                    else:
                        INCORRECT_MISC += 1
                j += 1
            else:
                if ner_result[j] == 'O':
                    CORRECT_O += 1
                else:
                    INCORRECT_O += 1
                j += 1

    # print 'ST: ' + st
    print ('PER: Correct: ' + str(CORRECT_PER) + ', Incorrect: ' + str(INCORRECT_PER))
    print ('LOC: Correct: ' + str(CORRECT_LOC) + ', Incorrect: ' + str(INCORRECT_LOC))
    print ('ORG: Correct: ' + str(CORRECT_ORG) + ', Incorrect: ' + str(INCORRECT_ORG))
    print ('MISC: Correct: ' + str(CORRECT_MISC) + ', Incorrect: ' + str(INCORRECT_MISC))
    print ('O: Correct: ' + str(CORRECT_O) + ', Incorrect: ' + str(INCORRECT_O))

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        #'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    # if i > 0:
    #     word1 = sent[i-1][0]
    #     postag1 = sent[i-1][1]
    #     features.update({
    #         '-1:word.lower()': word1.lower(),
    #         '-1:word.istitle()': word1.istitle(),
    #         '-1:word.isupper()': word1.isupper(),
    #         '-1:postag': postag1,
    #         '-1:postag[:2]': postag1[:2],
    #     })
    # else:
    #     features['BOS'] = True
    #
    # if i < len(sent)-1:
    #     word1 = sent[i+1][0]
    #     postag1 = sent[i+1][1]
    #     features.update({
    #         '+1:word.lower()': word1.lower(),
    #         '+1:word.istitle()': word1.istitle(),
    #         '+1:word.isupper()': word1.isupper(),
    #         '+1:postag': postag1,
    #         '+1:postag[:2]': postag1[:2],
    #     })
    # else:
    #     features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def writeOutput(actual_preds, output_csv = "extension_output.csv"):
    PER = "PER,"
    LOC = "LOC,"
    ORG = "ORG,"
    MISC = "MISC,"
    O = "O,"

    flat_list = []

    [flat_list.extend(x) for x in actual_preds]

    j = 0
    while(j<len(flat_list)):
        if (flat_list[j] == 'B-PER' or flat_list[j] == 'I-PER'):
            first = j
            j += 1
            while j < len(flat_list) and (
                            flat_list[j] == 'I-PER' or flat_list[j] == 'B-PER'):
                j += 1

            PER = PER + str(first) + "-" + str(j - 1) + " "
            # j += 1
            continue

        if flat_list[j] == 'B-LOC' or flat_list[j] == 'I-LOC':
            first = j
            j += 1
            while j < len(flat_list) and (
                            flat_list[j] == 'I-LOC' or flat_list[j] == 'B-LOC'):
                j += 1

            LOC = LOC + str(first) + "-" + str(j - 1) + " "
            # j += 1
            continue

        if flat_list[j] == 'B-ORG' or flat_list[j] == 'I-ORG':
            first = j
            j += 1
            while j < len(flat_list) and (
                            flat_list[j] == 'I-ORG' or flat_list[j] == 'B-ORG'):
                j += 1

            ORG = ORG + str(first) + "-" + str(j - 1) + " "
            # j += 1
            continue

        if flat_list[j] == 'B-MISC' or flat_list[j] == 'I-MISC':
            first = j
            j += 1
            while j < len(flat_list) and (
                            flat_list[j] == 'I-MISC' or flat_list[j] == 'B-MISC'):
                j += 1

            MISC = MISC + str(first) + "-" + str(j - 1) + " "
            # j+=1
            continue
        else:
            j += 1

    op_csv = open(output_csv, 'w+')
    st = "Type,Prediction"
    op_csv.write(st + "\n")
    op_csv.write(PER + "\n")
    op_csv.write(LOC + "\n")
    op_csv.write(ORG + "\n")
    op_csv.write(MISC + "\n")
    # op_csv.write(O + "\n")
    op_csv.close()

def validate_NER(y_val, y_pred):
    """
        Predicts NER for the test data
        :param filename: test.txt
        :param output_csv: output the file in the  required format
        :return: None
        """

    CORRECT_B_PER = 0
    CORRECT_I_PER = 0
    CORRECT_B_ORG = 0
    CORRECT_I_ORG = 0
    CORRECT_B_LOC = 0
    CORRECT_I_LOC = 0
    CORRECT_B_MISC = 0
    CORRECT_I_MISC = 0
    CORRECT_O = 0
    INCORRECT_B_PER = 0
    INCORRECT_I_PER = 0
    INCORRECT_B_ORG = 0
    INCORRECT_I_ORG = 0
    INCORRECT_B_LOC = 0
    INCORRECT_I_LOC = 0
    INCORRECT_B_MISC = 0
    INCORRECT_I_MISC = 0
    INCORRECT_O = 0


    for j in range(len(y_pred)):

        VALID_TOKENS = set(["B-PER", "B-LOC", "B-ORG", "B-MISC", "I-PER", "I-LOC", "I-ORG", "I-MISC"])

        # New Version

        if y_pred[j] in VALID_TOKENS:
            if (y_pred[j] == 'B-PER'):
                if y_val[j] == 'B-PER':
                    CORRECT_B_PER += 1
                else:
                    INCORRECT_B_PER += 1

            elif (y_pred[j] == 'I-PER'):
                if y_val[j] == 'I-PER':
                    CORRECT_I_PER += 1
                else:
                    INCORRECT_I_PER += 1

            elif (y_pred[j] == 'B-LOC'):
                if y_val[j] == 'B-LOC':
                    CORRECT_B_LOC += 1
                else:
                    INCORRECT_B_LOC += 1

            elif (y_pred[j] == 'I-LOC'):
                if y_val[j] == 'I-LOC':
                    CORRECT_I_LOC += 1
                else:
                    INCORRECT_I_LOC += 1

            elif (y_pred[j] == 'B-ORG'):
                if y_val[j] == 'B-ORG':
                    CORRECT_B_ORG += 1
                else:
                    INCORRECT_B_ORG += 1

            elif (y_pred[j] == 'I-ORG'):
                if y_val[j] == 'I-ORG':
                    CORRECT_I_ORG += 1
                else:
                    INCORRECT_I_ORG += 1

            elif (y_pred[j] == 'B-MISC'):
                if y_val[j] == 'B-MISC':
                    CORRECT_B_MISC += 1
                else:
                    INCORRECT_B_MISC += 1
                    # print ("B-MISC wrong: " + line_tokens[j] + ' classified as: ' + ner_result[j])

            elif (y_pred[j] == 'I-MISC'):
                if y_val[j] == 'I-MISC':
                    CORRECT_I_MISC += 1
                else:
                    INCORRECT_I_MISC += 1
            j += 1
        else:
            if y_val[j] == 'O':
                CORRECT_O += 1
            else:
                INCORRECT_O += 1
            j += 1

    # print 'ST: ' + st
    print ('B:')
    print ('PER: Correct: ' + str(CORRECT_B_PER) + ', Incorrect: ' + str(INCORRECT_B_PER))
    print ('LOC: Correct: ' + str(CORRECT_B_LOC) + ', Incorrect: ' + str(INCORRECT_B_LOC))
    print ('ORG: Correct: ' + str(CORRECT_B_ORG) + ', Incorrect: ' + str(INCORRECT_B_ORG))
    print ('MISC: Correct: ' + str(CORRECT_B_MISC) + ', Incorrect: ' + str(INCORRECT_B_MISC))

    print ('I:')
    print ('PER: Correct: ' + str(CORRECT_I_PER) + ', Incorrect: ' + str(INCORRECT_I_PER))
    print ('LOC: Correct: ' + str(CORRECT_I_LOC) + ', Incorrect: ' + str(INCORRECT_I_LOC))
    print ('ORG: Correct: ' + str(CORRECT_I_ORG) + ', Incorrect: ' + str(INCORRECT_I_ORG))
    print ('MISC: Correct: ' + str(CORRECT_I_MISC) + ', Incorrect: ' + str(INCORRECT_I_MISC))
    print ('O: Correct: ' + str(CORRECT_O) + ', Incorrect: ' + str(INCORRECT_O))

#Main Calls

train_sents, val_sents = tokenize(0.8, "train.txt")
test_sents, ignore = tokenize(1, "test.txt")
#validate_NER('train.txt')

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_val = [sent2features(s) for s in val_sents]
y_val = [sent2labels(s) for s in val_sents]

X_test = [sent2features(s) for s in test_sents]

print(X_train[0], y_train[0])

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
print(labels)

y_pred = crf.predict(X_val)
actual_preds = crf.predict(X_test)
writeOutput(actual_preds)
print(metrics.flat_f1_score(y_val, y_pred,
                      average='weighted', labels=labels))
y_flat_pred = []
y_flat_val = []
[y_flat_pred.extend(x) for x in y_pred]
[y_flat_val.extend(x) for x in y_val]
validate_NER(y_flat_val, y_flat_pred)

X_new_train = []
for sent in X_train:
    new_sent = []
    for word in sent:
        new_sent.append(list(word.values()))
    X_new_train.append(new_sent)

for j in range(len(X_new_train)):
    for i in range(len(X_new_train[j])):
        X_new_train[j][i] = np.array(X_new_train[j][i])
    X_new_train[j] = np.array(X_new_train[j]).reshape(len(X_new_train[j]),8)

X_train = np.array(X_new_train)
Y_train = np.array(y_train)
#X_new_train = np.array([np.array(x.values()) for np.array(y) in X_train for x in y])

X_new_val = []
for sent in X_val:
    new_sent = []
    for word in sent:
        new_sent.append(list(word.values()))
    X_new_val.append(new_sent)

for j in range(len(X_new_val)):
    for i in range(len(X_new_val[j])):
        X_new_val[j][i] = np.array(X_new_val[j][i])
    X_new_val[j] = np.array(X_new_val[j]).reshape(len(X_new_val[j]),8)

X_val = np.array(X_new_val)
Y_val = np.array(y_val)


model = ChainCRF()
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
print(X_train[0])
print(X_train[0].shape)
ssvm.fit(X_train, y_train)
print(ssvm.score(X_val, y_val))