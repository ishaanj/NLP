def tokenize(filename):
    f = open(filename, 'r')
    line_count = 0
    b_tokens = {}
    i_tokens = {}
    o_tokens = {}

    all_lines = f.readlines()
    for i in range(int(len(all_lines)/3)):
        line_tokens = all_lines[3*i].split()
        line_pos = all_lines[3*i + 1].split()
        line_ner = all_lines[3*i + 2].split()

        ner = ""
        for j in range(len(line_ner)):
            if 'B-' in line_ner[j]:
                if line_tokens[j] in b_tokens:
                    continue
                else:
                    b_tokens[line_tokens[j]] = line_ner[j]
            elif 'I-' in line_ner[j]:
                if line_tokens[j] in i_tokens:
                    continue
                else:
                    i_tokens[line_tokens[j]] = line_ner[j]
            else:
                if line_tokens[j] in o_tokens:
                    continue
                else:
                    o_tokens[line_tokens[j]] = line_ner[j]
    f.close()
    return b_tokens, i_tokens, o_tokens

B_Tokens, I_Tokens, O_Tokens = tokenize("train.txt")



