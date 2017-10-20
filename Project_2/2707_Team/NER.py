#Need count(wordType,NER)
#Need count(NER)
#Need count(NER,NER)

#will keep a dummy start NER and there will be a count(start NER, NER) and a count(start NER) = #of lines


import nltk
import numpy as np
import math
def tokenize(split, filename="train.txt"):
    """
    Tokenizes the train file and returns
    word : NER type dictionary
    :param filename: train.txt
    :return: all_tokens
    """
    f = open(filename, 'r')
    all_tokens = {}
    i_tokens = {}
    o_tokens = {}
    count_wordType_NER = {}
    count_NER_NER = {}
    count_NER = {}
    count_NER["START"] = 0 #useful later - refer comments


    all_lines = f.readlines()
    #print(len(all_lines)/3)
    for i in range(int(len(all_lines)/3 * split)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i+1].split()
        line_ner = all_lines[3*i+2].split()

        for j in range(len(line_ner)):
            if "O" != line_ner[j]:
                if line_tokens[j] not in all_tokens:
                    all_tokens[line_tokens[j]] = line_ner[j]
            #leaving above unchanged and adding new code here
            wordType = line_tokens[j]
            NER = line_ner[j] if(line_ner[j] != 'O') else line_ner[j]

            if((wordType,NER) not in count_wordType_NER):
                count_wordType_NER[(wordType,NER)] = 1
            else:
                count_wordType_NER[(wordType, NER)] += 1
            if (j == 0):
                prevNER = "START"
                count_NER[prevNER] += 1 #already initialized to zero above
            else:
                prevNER = line_ner[j - 1] if(line_ner[j-1] != 'O') else line_ner[j-1]
            if ((prevNER, NER) not in count_NER_NER):
                count_NER_NER[(prevNER, NER)] = 1
            else:
                count_NER_NER[(prevNER, NER)] += 1
            if(NER not in count_NER):
                count_NER[NER] = 1
            else:
                count_NER[NER] += 1

    f.close()
    return all_tokens, count_NER, count_NER_NER, count_wordType_NER


def HMM(line_tokens, count_NER, count_NER_NER, count_wordType_NER):
    """
        Predicts NER for the test data sentence(line_tokens)
        Each row will have info about the i'th word in the sentence
        Each row will only have 5 cols. 0-PER 1-LOC 2-ORG 3-MISC 4-O
        Using +1 smoothing, so k = 1
        :param line_tokens: one line from test.txt
        :return: None
    """
    hmatrix = []
    ptrmatrix = []
    #NER_Types = ["PER","LOC","ORG","MISC", "O"]
    NER_Types = ["B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","B-MISC","I-MISC", "O"]
    word_types = len(ALL_TOKENS)
    ner = 9
    # k = 0.26
    k=0.26

    start_word = line_tokens[0]
    cur_row = []
    for NER_Type in NER_Types:
        num = count_NER_NER[("START",NER_Type)] if ("START",NER_Type) in count_NER_NER else 0
        den = count_NER["START"]
        trans_prob = float((num + k)/(den+ner*k))
        num = count_wordType_NER[(start_word,NER_Type)] if (start_word,NER_Type) in count_wordType_NER else 0
        den = count_NER[NER_Type]
        gen_prob = float((num + k)/(den+ner*k))
        if("I-" in NER_Type):
            trans_prob = 0.000000000000002
        else:
            trans_prob = 0.199999999999998
        #cur_row.append(trans_prob*gen_prob)
        cur_row.append(math.log(trans_prob)+math.log(gen_prob))
    hmatrix.append(cur_row)

    for j in range(1,len(line_tokens)):
        prev_row = hmatrix[j-1]
        cur_word = line_tokens[j]
        cur_row = []
        ptr_row = []
        for NER_Type in NER_Types:
            tempspace = []
            for p in range(len(prev_row)):
                prev_NER_Type = NER_Types[p]
                prev_score = prev_row[p]

                num = count_NER_NER[(prev_NER_Type, NER_Type)] if (prev_NER_Type, NER_Type) in count_NER_NER else 0
                den = count_NER[prev_NER_Type]
                trans_prob = float((num + k) / (den + ner * k))
                num = count_wordType_NER[(cur_word, NER_Type)] if (cur_word, NER_Type) in count_wordType_NER else 0
                den = count_NER[NER_Type]
                gen_prob = float((num + k) / (den + ner * k))
                #tempspace.append(prev_score*trans_prob*gen_prob)
                tempspace.append(prev_score+math.log(trans_prob)+math.log(gen_prob))

            cur_row.append(max(tempspace))
            ptr_row.append(np.argmax(tempspace))
        hmatrix.append(cur_row)
        ptrmatrix.append(ptr_row)

    result = []
    # idx = np.argmax(hmatrix[len(line_tokens)-1])
    # bptr = ptrmatrix[len(line_tokens)-2][idx]
    idx = np.argmax(hmatrix[-1])
    if(len(ptrmatrix)>0):
        bptr = ptrmatrix[-1][idx]

    result.append(NER_Types[idx])
    for p in range(len(ptrmatrix)-2,-1,-1):
        result.append(NER_Types[bptr])
        bptr = ptrmatrix[p][bptr]

    if(len(ptrmatrix)>0):
        result.append(NER_Types[bptr])
    result = result[::-1]

    return result

def validate_NER(filename="test.txt"):
    """
        Predicts NER for the test data
        :param filename: test.txt
        :param output_csv: output the file in the  required format
        :return: None
        """

    f = open(filename, 'r')
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
    all_lines = f.readlines()
    num_lines = len(all_lines) / 3

    states = ["B-PER", "B-LOC", "B-ORG", "B-MISC", "I-PER", "I-LOC", "I-ORG", "I-MISC", "O"]
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    states_dict = {}
    for i in range(len(states)):
        states_dict[states[i]] = counts[i]

    for i in range(int(0.8*num_lines), int(num_lines)):
        line_tokens = all_lines[3 * i].split()
        line_pos = all_lines[3 * i + 1].split()
        ner_result = all_lines[3 * i + 2].split()

        # VALID_TOKENS = set(["PER","ORG","LOC","MISC"])
        VALID_TOKENS = set(["B-PER", "B-LOC", "B-ORG", "B-MISC", "I-PER", "I-LOC", "I-ORG", "I-MISC"])

        tagged_tokens = HMM(line_tokens, count_NER, count_NER_NER, count_wordType_NER)
        j = 1
        # New Version
        while (j < len(tagged_tokens)):
            states_dict[ner_result[j]] += 1
            if tagged_tokens[j] in VALID_TOKENS:
                if (tagged_tokens[j] == 'B-PER'):
                    if ner_result[j] == 'B-PER':
                        CORRECT_B_PER += 1
                    else:
                        INCORRECT_B_PER += 1

                elif (tagged_tokens[j] == 'I-PER'):
                    if ner_result[j] == 'I-PER':
                        CORRECT_I_PER += 1
                    else:
                        INCORRECT_I_PER += 1

                elif (tagged_tokens[j] == 'B-LOC'):
                    if ner_result[j] == 'B-LOC':
                        CORRECT_B_LOC += 1
                    else:
                        INCORRECT_B_LOC += 1

                elif (tagged_tokens[j] == 'I-LOC'):
                    if ner_result[j] == 'I-LOC':
                        CORRECT_I_LOC += 1
                    else:
                        INCORRECT_I_LOC += 1

                elif (tagged_tokens[j] == 'B-ORG'):
                    if ner_result[j] == 'B-ORG':
                        CORRECT_B_ORG += 1
                    else:
                        INCORRECT_B_ORG += 1

                elif (tagged_tokens[j] == 'I-ORG'):
                    if ner_result[j] == 'I-ORG':
                        CORRECT_I_ORG += 1
                    else:
                        INCORRECT_I_ORG += 1

                elif (tagged_tokens[j] == 'B-MISC'):
                    if ner_result[j] == 'B-MISC':
                        CORRECT_B_MISC += 1
                    else:
                        INCORRECT_B_MISC += 1
                        # print ("B-MISC wrong: " + line_tokens[j] + ' classified as: ' + ner_result[j])

                elif (tagged_tokens[j] == 'I-MISC'):
                    if ner_result[j] == 'I-MISC':
                        CORRECT_I_MISC += 1
                    else:
                        INCORRECT_I_MISC += 1
                j += 1
            else:
                if ner_result[j] == 'O':
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

    # print states_dict

    correct_pred_b = CORRECT_B_PER + CORRECT_B_LOC + CORRECT_B_ORG + CORRECT_B_MISC
    correct_pred_i = CORRECT_I_PER + CORRECT_I_LOC + CORRECT_I_ORG + CORRECT_I_MISC
    incorrect_pred_b = INCORRECT_B_PER + INCORRECT_B_LOC + INCORRECT_B_ORG + INCORRECT_B_MISC
    incorrect_pred_i = INCORRECT_I_PER + INCORRECT_I_LOC + INCORRECT_I_ORG + INCORRECT_I_MISC

    correct_pred = correct_pred_b + correct_pred_i
    total_pred = correct_pred_b + correct_pred_i + incorrect_pred_b + incorrect_pred_i
    total_correct = states_dict['B-PER'] + states_dict['B-LOC'] + states_dict['B-ORG'] + states_dict['B-MISC'] + states_dict['I-PER'] + states_dict['I-LOC'] + states_dict['I-ORG'] + states_dict['I-MISC']

    print correct_pred
    print total_pred
    print total_correct

    precision = float(correct_pred) / total_pred
    recall = float(correct_pred) / total_correct
    f1 = 2.0*(precision * recall)/(precision + recall)

    print precision
    print recall
    print f1

def predict_NER(filename="test.txt", output_csv="output.csv"):
    """
    Predicts NER for the test data
    :param filename: test.txt
    :param output_csv: output the file in the  required format
    :return: None
    """
    f = open(filename, 'r')
    PER = "PER,"
    LOC = "LOC,"
    ORG = "ORG,"
    MISC = "MISC,"
    O = "O,"
    all_lines = f.readlines()
    for i in range(int(len(all_lines)/3)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i + 1].split()
        token_number = all_lines[3*i + 2].split()
        #VALID_TOKENS = set(["PER","ORG","LOC","MISC"])
        VALID_TOKENS = set(["B-PER","B-LOC","B-ORG","B-MISC","I-PER","I-LOC","I-ORG","I-MISC"])
        tagged_tokens = HMM(line_tokens, count_NER, count_NER_NER, count_wordType_NER)
        j = 0
        #commenting this to rewrite the new version
        # while(j<len(line_tokens)):
        #     if line_tokens[j] in ALL_TOKENS:
        #         if("B-PER" == ALL_TOKENS[line_tokens[j]] or "I-PER" == ALL_TOKENS[line_tokens[j]]):
        #             first = j
        #             while j<len(line_tokens) and line_tokens[j] in ALL_TOKENS and ("B-PER" == ALL_TOKENS[line_tokens[j]] or "I-PER" == ALL_TOKENS[line_tokens[j]]):
        #                 j += 1
        #
        #             PER = PER + str(token_number[first]) + "-" + str(token_number[j-1]) + " "
        #             continue
        #         if "B-LOC" == ALL_TOKENS[line_tokens[j]] or "I-LOC" == ALL_TOKENS[line_tokens[j]]:
        #             first = j
        #             while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
        #                     "B-LOC" == ALL_TOKENS[line_tokens[j]] or "I-LOC" == ALL_TOKENS[line_tokens[j]]):
        #                 j += 1
        #
        #             LOC = LOC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
        #             continue
        #         if "B-ORG" in ALL_TOKENS[line_tokens[j]] or "I-ORG" in ALL_TOKENS[line_tokens[j]]:
        #             first = j
        #             while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
        #                             "B-ORG" == ALL_TOKENS[line_tokens[j]] or "I-ORG" == ALL_TOKENS[line_tokens[j]]):
        #                 j += 1
        #
        #             ORG = ORG + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
        #             continue
        #         if "B-MISC" in ALL_TOKENS[line_tokens[j]] or "I-MISC" in ALL_TOKENS[line_tokens[j]]:
        #             first = j
        #             while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
        #                             "B-MISC" == ALL_TOKENS[line_tokens[j]] or "I-MISC" == ALL_TOKENS[line_tokens[j]]):
        #                 j += 1
        #
        #             MISC = MISC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
        #             continue
        #     else:
        #         O = O + str(token_number[j]) + "-" + str(token_number[j]) + " "
        #         j += 1

        #New Version
        while (j < len(tagged_tokens)):
            if tagged_tokens[j] in VALID_TOKENS:
                if (tagged_tokens[j] == 'B-PER' or tagged_tokens[j] == 'I-PER'):
                    first = j
                    j += 1
                    while j < len(tagged_tokens) and (
                                    tagged_tokens[j] == 'I-PER' or tagged_tokens[j] == 'B-PER'):
                        j += 1

                    PER = PER + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    # j += 1
                    continue

                if tagged_tokens[j] == 'B-LOC' or tagged_tokens[j] == 'I-LOC':
                    first = j
                    j += 1
                    while j < len(tagged_tokens) and (
                                    tagged_tokens[j] == 'I-LOC' or tagged_tokens[j] == 'B-LOC'):
                        j += 1

                    LOC = LOC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    # j += 1
                    continue

                if tagged_tokens[j] == 'B-ORG' or tagged_tokens[j] == 'I-ORG':
                    first = j
                    j += 1
                    while j < len(tagged_tokens) and (
                                    tagged_tokens[j] == 'I-ORG' or tagged_tokens[j] == 'B-ORG'):
                        j += 1

                    ORG = ORG + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    # j += 1
                    continue

                if tagged_tokens[j] == 'B-MISC' or tagged_tokens[j] == 'I-MISC':
                    first = j
                    j += 1
                    while j < len(tagged_tokens) and (
                                    tagged_tokens[j] == 'I-MISC' or tagged_tokens[j] == 'B-MISC'):
                        j += 1

                    MISC = MISC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    # j+=1
                    continue
            else:
                j += 1
                # first = j
                # while j < len(tagged_tokens) and tagged_tokens[j] not in VALID_TOKENS:
                #     j += 1
                # O = O + str(token_number[first]) + "-" + str(token_number[j-1]) + " "

    op_csv = open(output_csv,'w+')
    st = "Type,Prediction"
    op_csv.write(st + "\n")
    op_csv.write(PER + "\n")
    op_csv.write(LOC + "\n")
    op_csv.write(ORG + "\n")
    op_csv.write(MISC + "\n")
    # op_csv.write(O + "\n")
    op_csv.close()


# Main calls
ALL_TOKENS, count_NER, count_NER_NER, count_wordType_NER = tokenize(0.8, "train.txt")
validate_NER('train.txt')
predict_NER('test.txt', "output.csv")

#f = open('train.txt')
#raw = f.read()
# TRAIN_ALL_TOKENS = ALL_TOKENS[:int(len(ALL_TOKENS)*0.8)]
# VALIDATE_ALL_TOKENS = ALL_TOKENS[int(len(ALL_TOKENS)*0.8):]
#
# TRAIN_count_NER = count_NER[:int(len(ALL_TOKENS)*0.8)]
# TEST_count_NER= count_NER[int(len(ALL_TOKENS)*0.8):]
#
# TRAIN_count_NER_NER = count_NER_NER[:int(len(count_NER_NER)*0.8)]
# VALIDATE_count_NER_NER = count_NER_NER[:int(len(count_NER_NER)*0.8)]
#
# TRAIN_wordType_NER_NER = count_wordType_NER[:int(len(count_wordType_NER)*0.8)]
# VALIDATE_wordType_NER_NER = count_wordType_NER[:int(len(count_wordType_NER)*0.8)]



# tokens = nltk.word_tokenize(raw)

# #Create your bigrams
# #bgs = nltk.bigrams(tokens)
# ugs = nltk.ngrams(tokens,1)
# #tgs = nltk.trigrams(tokens)
# #fgs = nltk.ngrams(tokens,4)
# #compute frequency distribution for all the bigrams in the text
# fdist = nltk.FreqDist(ugs)
# #tdist = nltk.FreqDist(fgs)
# for k,v in fdist.items():
#     print(k,v)