#Need count(wordType,NER)
#Need count(NER)
#Need count(NER,NER)

#will keep a dummy start NER and there will be a count(start NER, NER) and a count(start NER) = #of lines


import nltk
import  numpy as np
def tokenize(filename="train.txt"):
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
    for i in range(int(len(all_lines)/3)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i+1].split()
        line_ner = all_lines[3*i+2].split()

        for j in range(len(line_ner)):
            if "O" != line_ner[j]:
                if line_tokens[j] not in all_tokens:
                    all_tokens[line_tokens[j]] = line_ner[j]
            #leaving above unchanged and adding new code here
            wordType = line_tokens[j]
            NER = line_ner[j]

            if((wordType,NER) not in count_wordType_NER):
                count_wordType_NER[(wordType,NER)] = 1
            else:
                count_wordType_NER[(wordType, NER)] += 1
            if (j == 0):
                prevNER = "START"
                count_NER[prevNER] += 1 #already initialized to zero above
            else:
                prevNER = line_ner[j - 1]
            if ((prevNER, NER) not in count_NER_NER):
                count_NER_NER[(prevNER, NER)] = 1
            else:
                count_NER_NER[(prevNER, NER)] += 1
            if(NER not in count_NER):
                count_NER[NER] = 1
            else:
                count_NER += 1

    f.close()
    return all_tokens, count_NER, count_NER_NER, count_wordType_NER


def HMM(line_tokens):
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
    NER_Types = ["PER","LOC","ORG","MISC", "O"]
    word_types = len(ALL_TOKENS)
    ner = 5
    k = 1

    start_word = line_tokens[0]
    cur_row = []
    for NER_Type in NER_Types:
        num = count_NER_NER[("START",NER_Type)] if ("START",NER_Type) in count_NER_NER else 0
        den = count_NER["START"]
        trans_prob = float((num + k)/(den+ner*k))
        num = count_wordType_NER[(start_word,NER_Type)] if (start_word,NER_Type) in count_wordType_NER else 0
        den = count_NER[NER_Type]
        gen_prob = float((num + k)/(den+ner*k))
        cur_row.append(trans_prob*gen_prob)
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
                tempspace.append(prev_score*trans_prob*gen_prob)

            cur_row.append(max(tempspace))
            ptr_row.append(np.argmax(tempspace))
        hmatrix.append(cur_row)
        ptrmatrix.append(ptr_row)

    result = []
    idx = np.argmax(hmatrix[len(line_tokens)-1])
    bptr = ptrmatrix[len(line_tokens)-1][idx]
    result.append(NER_Types[idx])
    for p in range(len(line_tokens)-2,0,-1):
        result.append(NER_Types[bptr])
        bptr = ptrmatrix[p][bptr]

    result = result[::-1]
    return result






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
        tagged_tokens = HMM(line_tokens)
        j = 0
        while(j<len(line_tokens)):
            if line_tokens[j] in ALL_TOKENS:
                if("B-PER" == ALL_TOKENS[line_tokens[j]] or "I-PER" == ALL_TOKENS[line_tokens[j]]):
                    first = j
                    while j<len(line_tokens) and line_tokens[j] in ALL_TOKENS and ("B-PER" == ALL_TOKENS[line_tokens[j]] or "I-PER" == ALL_TOKENS[line_tokens[j]]):
                        j += 1

                    PER = PER + str(token_number[first]) + "-" + str(token_number[j-1]) + " "
                    continue
                if "B-LOC" == ALL_TOKENS[line_tokens[j]] or "I-LOC" == ALL_TOKENS[line_tokens[j]]:
                    first = j
                    while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
                            "B-LOC" == ALL_TOKENS[line_tokens[j]] or "I-LOC" == ALL_TOKENS[line_tokens[j]]):
                        j += 1

                    LOC = LOC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    continue
                if "B-ORG" in ALL_TOKENS[line_tokens[j]] or "I-ORG" in ALL_TOKENS[line_tokens[j]]:
                    first = j
                    while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
                                    "B-ORG" == ALL_TOKENS[line_tokens[j]] or "I-ORG" == ALL_TOKENS[line_tokens[j]]):
                        j += 1

                    ORG = ORG + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    continue
                if "B-MISC" in ALL_TOKENS[line_tokens[j]] or "I-MISC" in ALL_TOKENS[line_tokens[j]]:
                    first = j
                    while j < len(line_tokens) and line_tokens[j] in ALL_TOKENS and (
                                    "B-MISC" == ALL_TOKENS[line_tokens[j]] or "I-MISC" == ALL_TOKENS[line_tokens[j]]):
                        j += 1

                    MISC = MISC + str(token_number[first]) + "-" + str(token_number[j - 1]) + " "
                    continue
            else:
                O = O + str(token_number[j]) + "-" + str(token_number[j]) + " "
                j += 1

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
ALL_TOKENS, count_NER, count_NER_NER, count_wordType_NER = tokenize("train.txt")
predict_NER('test.txt', "output.csv")

f = open('train.txt')
raw = f.read()

tokens = nltk.word_tokenize(raw)

#Create your bigrams
#bgs = nltk.bigrams(tokens)
ugs = nltk.ngrams(tokens,1)
#tgs = nltk.trigrams(tokens)
#fgs = nltk.ngrams(tokens,4)
#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(ugs)
#tdist = nltk.FreqDist(fgs)
for k,v in fdist.items():
    print(k,v)