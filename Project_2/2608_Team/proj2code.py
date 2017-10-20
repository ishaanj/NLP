import numpy as np
import csv
import math


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

    count_word_ner = {}
    count_ner_prevner = {}
    count_ner = {}
    for i in range(len(tokens)):
        token = tokens[i]
        ner_tag = ner_tags[i]

        #calculate counts for transition probabilities
        if i>0:
            prevner_tag = ner_tags[i-1]
            if (ner_tag, prevner_tag) not in count_ner_prevner:
                count_ner_prevner[(ner_tag, prevner_tag)] = 0
            count_ner_prevner[(ner_tag, prevner_tag)] += 1

        #calculate counts for generation probabilities
        if (token, ner_tag) not in count_word_ner:
            count_word_ner[(token, ner_tag)] = 0
        count_word_ner[(token, ner_tag)] += 1

        if ner_tag not in count_ner:
            count_ner[ner_tag] = 0
        count_ner[ner_tag] += 1

    return count_word_ner, count_ner_prevner, count_ner


def process_test_file(fname):
    test_tokens = []
    test_pos_tags = []
    test_token_indices = []
    with open(fname) as fp:

        token_line = fp.readline()
        while token_line:
            pos_line = fp.readline()
            token_indices_line = fp.readline()

            test_tokens.extend(token_line.strip().split())
            test_pos_tags.extend(pos_line.strip().split())
            test_token_indices.extend(token_indices_line.strip().split())

            token_line = fp.readline()

    return test_tokens, test_pos_tags, test_token_indices

def get_generation_probability(test_token, ner_type, count_word_ner, count_ner, num_ner_tags=9, k=0.012):
    numerator = count_word_ner[(test_token, ner_type)] if (test_token, ner_type) in count_word_ner else 0
    denominator = count_ner[ner_type]
    gen_prob = float(numerator + k)/float(denominator + k*num_ner_tags)
    return gen_prob

def get_transition_probability(ner_type, prevner_type, count_ner_prevner, count_ner, num_ner_tags=9, k=0.012):
    numerator = count_ner_prevner[(ner_type, prevner_type)] if (ner_type, prevner_type) in count_ner_prevner else 0
    denominator = count_ner[prevner_type]
    trans_prob = float(numerator + k) / float(denominator + k * num_ner_tags)
    return trans_prob


def applyHMM(test_tokens, count_word_ner, count_ner_prevner, count_ner, ner_types):
    scores = [[0]]
    bkptrs = [[-1]]
    for i in range(1, len(test_tokens)+1):
        test_token = test_tokens[i-1]
        cur_token_scores = []
        cur_token_bkptrs = []
        for j in range(len(ner_types)):
            ner_type = ner_types[j]
            gen_prob = get_generation_probability(test_token, ner_type, count_word_ner, count_ner)

            #Initial state
            if i==1:
                if ner_type[0] == "B" or ner_type[0] == "O":
                    trans_prob = 0.1999999998
                else:
                    trans_prob = 0.0000000002

                cur_score = (scores[i-1][0] + math.log(gen_prob) + math.log(trans_prob))
                cur_token_scores.append(cur_score)

                cur_bkptr = 0
                cur_token_bkptrs.append(cur_bkptr)

            else:
                cur_token_prevner_scores = []
                for k in range(len(ner_types)):
                    prevner_type = ner_types[k]
                    trans_prob = get_transition_probability(ner_type, prevner_type, count_ner_prevner, count_ner)
                    cur_score = (scores[i-1][k] + math.log(gen_prob) + math.log(trans_prob))
                    cur_token_prevner_scores.append(cur_score)

                cur_score = max(cur_token_prevner_scores)
                cur_token_scores.append(cur_score)

                cur_bkptr = np.argmax(cur_token_prevner_scores)
                cur_token_bkptrs.append(cur_bkptr)
        scores.append(cur_token_scores)
        bkptrs.append(cur_token_bkptrs)

    curidx = np.argmax(scores[len(scores)-1])
    test_ner_tags = []
    for i in range(len(scores)-1, 0, -1):
        test_ner_tags.append(ner_types[curidx])
        curidx = bkptrs[i][curidx]

    #Reverse the resulting ner tags list as we go backwards
    test_ner_tags = test_ner_tags[::-1]
    return test_ner_tags


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
count_word_ner, count_ner_prevner, count_ner = process_training_file("train.txt")
ner_types = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]

#Process the test file
test_tokens, test_pos_tags, test_token_indices = process_test_file("test.txt")
train_tokens, train_pos, train_ner = process_test_file("train.txt")

#Apply HMM algo
# test_ner_tags = applyHMM(test_tokens, count_word_ner, count_ner_prevner, count_ner, ner_types)
val_ner_tags = applyHMM(train_tokens[:10000], count_word_ner, count_ner_prevner, count_ner, ner_types)

for i, _ in enumerate(train_ner[:10000]):
    if train_ner[i] != val_ner_tags[i]:
        print("Word: ", train_tokens[i])
        print("Predicted: ", val_ner_tags[i])
        print("Correct: ", train_ner[i])
        print("Index: ", i)
#Generate CSV
# generateCSV(test_ner_tags)
