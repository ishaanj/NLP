def tokenize(filename):
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

ALL_TOKENS = tokenize("train.txt")

"""
Now run on test data
"""
def test_me(filename, output_csv):
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

        for j in range(len(line_tokens)):
            if line_tokens[j] in ALL_TOKENS:
                if "B-PER" == ALL_TOKENS[line_tokens[j]] or "I-PER" == ALL_TOKENS[line_tokens[j]]:
                    PER = PER + str(token_number[j]) + "-" + str(token_number[j]) + " "
                if "B-LOC" == ALL_TOKENS[line_tokens[j]] or "I-LOC" == ALL_TOKENS[line_tokens[j]]:
                    LOC = LOC + str(token_number[j]) + "-" + str(token_number[j]) + " "
                if "B-ORG" in ALL_TOKENS[line_tokens[j]] or "I-ORG" in ALL_TOKENS[line_tokens[j]]:
                    ORG = ORG + str(token_number[j]) + "-" + str(token_number[j]) + " "
                if "B-MISC" in ALL_TOKENS[line_tokens[j]] or "I-MISC" in ALL_TOKENS[line_tokens[j]]:
                    MISC = MISC + str(token_number[j]) + "-" + str(token_number[j]) + " "
            else:
                O = O + str(token_number[j]) + "-" + str(token_number[j]) + " "

    op_csv = open(output_csv,'w+')
    st = "Type,Prediction"
    op_csv.write(st + "\n")
    op_csv.write(PER + "\n")
    op_csv.write(LOC + "\n")
    op_csv.write(ORG + "\n")
    op_csv.write(MISC + "\n")
    # op_csv.write(O + "\n")
    op_csv.close()

test_me('test.txt', "output.csv")