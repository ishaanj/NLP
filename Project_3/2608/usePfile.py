import time
import re
import string
import spacy
import json
import os.path
NLP = spacy.load("en_core_web_sm")


def normalize_context(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def spacy_me(sentence):
    """
    Use spacy to get sentence entity
    :param sentence:
    :return: sentence entity list
    """
    sentence_ents = [(ent.text, int(ent.start_char), int(ent.end_char), ent.label_) for ent in
                     sentence.ents]
    return sentence_ents

def rule_based_classify(questiondoc, sentence_ents, candidate_sentence):
    prediction = ''
    if "when" in str(questiondoc).lower() or "in" == str(questiondoc[0]).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "date" or ent[3].lower() == "time":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "who" in str(questiondoc).lower() or "whom" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "person" or ent[3].lower() == "org" or ent[3].lower() == "facility":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "many" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "quantity":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "much" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "money" or ent[3].lower() == "quantity" or ent[3].lower() == "percent":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "which" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "person" or ent[3].lower() == "quantity" or ent[3].lower() == "percent":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "where" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "gpe" or ent[3].lower() == "loc" or ent[3].lower() == "facility":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "whose" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "person" or ent[3].lower() == "org" or ent[3].lower() == "facility" \
                    or ent[3].lower() == "gpe" or ent[3].lower() == "loc":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    elif "percent" in str(questiondoc).lower() or "percentage" in str(questiondoc).lower():
        for ent in sentence_ents:
            if ent[3].lower() == "percent":
                prediction = candidate_sentence[ent[1]:ent[2]]
                break
    else:
        prediction = candidate_sentence
        prediction = normalize_context(prediction)

    return prediction


def classify_write_use_file(filename):
    question_type = {}

    with open(filename) as data_file:
        with open('new_2608_' + filename, 'w') as write_file:
            data = json.load(data_file)
            PREDICTIONS = {}
            for data in data['data']:
                for para in data['paragraphs']:
                    context = para['context']
                    contextdoc = NLP(context)
                    sentences = list(contextdoc.sents)
                    for qas in para['qas']:
                        question = qas['question']
                        q = question.split()[0].lower()
                        if q in question_type:
                            question_type[q] += 1
                        else:
                            question_type[q] = 1
                        id = qas['id']

                        questiondoc = NLP(question)
                        maxsim = 0
                        maxsim_sentence = ""
                        for sentence in sentences:
                            if questiondoc.similarity(sentence) >= maxsim:
                                maxsim = questiondoc.similarity(sentence)
                                maxsim_sentence = sentence

                        mxsim_sentence_doc = NLP(maxsim_sentence.text)

                        sentence_entities = spacy_me(mxsim_sentence_doc)
                        answer = rule_based_classify(questiondoc, sentence_entities, maxsim_sentence.text)

                        if answer == "":
                            answer = maxsim_sentence.text
                            answer = normalize_context(answer)

                        PREDICTIONS[id] = answer
            json.dump(PREDICTIONS, write_file)

def classify_write_use_pickle(maxsim_sentence, questiondoc):
    mxsim_sentence_doc = NLP(maxsim_sentence)
    sentence_entities = spacy_me(mxsim_sentence_doc)
    return rule_based_classify(questiondoc, sentence_entities, maxsim_sentence)


def get_pos(intent, sentence):
    doc = NLP(sentence)
    answer = []
    if intent in ["PERSON", "PRODUCT", "LOC", "ABBR", "ORG"]:
        for token in doc:
            if token.pos_ in ["PROPN", "NOUN"]:
                answer.append(token.text)
    elif intent == "QUANTITY":
        for token in doc:
            if token.pos_ in ["NUM", "NOUN"]:
                answer.append(token.text)

    return " ".join(answer)


def NER_Span(pickle_dict):
    result_dict = {}
    i = 0
    for qid in pickle_dict:
        print("DOING: %d" %i)
        i+=1
        tuple = pickle_dict[qid]
        sentence = tuple[0]
        intent = tuple[1]
        question = tuple[2]
        doc = NLP(sentence)
        answerFound = False
        if intent == 'NUM':
            intent = 'QUANTITY'
        if intent == 'HUM':
            intent = 'PERSON'
        if intent == 'ENTY':
            intent = 'PRODUCT'
        for ent in doc.ents:
            entityLabel = ent.label_
            if entityLabel == 'GPE' or entityLabel == 'FACILITY':
                entityLabel = 'LOC'
            if entityLabel == intent:
                result_dict[qid] = sentence[ent.start_char:ent.end_char+1]
                answerFound = True
                break
        if not answerFound:
            # TODO: Use rule based system
            answer = classify_write_use_pickle(sentence, question)
            if answer:
                result_dict[qid] = answer
                answerFound = True
        if not answerFound:
            # TODO: Can use POS here
            answer = get_pos(intent, sentence)
            if answer:
                result_dict[qid] = answer
                answerFound = True
        if not answerFound:
            result_dict[qid] = normalize_context(sentence)

    return result_dict


if __name__ == "__main__":
    start_time = time.time()
    print("Started %s" % str(time.ctime()))

    filename = 'PickleTest.json'
    with open(filename, 'r') as pickle_file:
        data = json.load(pickle_file)
        result_dict = NER_Span(data)

        write_file = open('2_'+ filename, 'w')
        json.dump(result_dict, write_file)

    # classify_write_use_file('testing.json')
    print("--- %s min ---" % ((time.time() - start_time) / 60))