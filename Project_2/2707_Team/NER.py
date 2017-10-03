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

    all_lines = f.readlines()
    for i in range(int(len(all_lines)/3)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i+1].split()
        line_ner = all_lines[3*i+2].split()

        for j in range(len(line_ner)):
            if "O" != line_ner[j]:
                if line_tokens[j] not in all_tokens:
                    all_tokens[line_tokens[j]] = line_ner[j]
    f.close()
    return all_tokens


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
ALL_TOKENS = tokenize("train.txt")
predict_NER('test.txt', "output.csv")