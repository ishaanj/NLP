def scnlp():
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://corenlp.run',port=80)
    text = "Today the college, housed in the Fitzpatrick, Cushing, " \
           "and Stinson-Remick Halls of Engineering, includes five departments of study"
    t2 = "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
    tags = nlp.ner(text)
    print(tags)
import time
import re
import string
import spacy
import json
import os.path
nlp = spacy.load("en_core_web_sm")
# # from QA import normalize_context
# with open('training.json') as data_file:
#     data = json.load(data_file)
#     predictions = {}
#     for data in data['data']:
#         for paragraph in data['paragraphs']:
#             context = paragraph['context']
#             # normalized_context = normalize_context(context)
#             contextdoc = nlp(context)
#             sentences = list(contextdoc.sents)
#             for qas in paragraph['qas']:
#                 question = qas['question']
#                 id = qas['id']
#                 answers = qas['answers']
#                 # normalized_question = normalize_context(question)
#                 questiondoc = nlp(question)
#                 mxsim = 0
#                 mxsim_sentence = ""
#                 for sentence in sentences:
#                     if questiondoc.similarity(sentence) >= mxsim:
#                         mxsim = questiondoc.similarity(sentence)
#                         mxsim_sentence = sentence.text
#                 print(mxsim_sentence)

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

question_type = {}
def all_new_awesomeness(filename):
    if False:
    # if os.path.isfile('qid_'+filename):
        # don't do spacy part
        file_qid = open('qid_'+filename, 'r')
        qid_sent = json.load(file_qid)

    else:
        qid_sent = {}
        with open(filename) as data_file:
            with open('2608_'+ filename, 'w') as write_file:
                data = json.load(data_file)
                predictions = {}
                for data in data['data']:
                    for paragraph in data['paragraphs']:
                        context = paragraph['context']
                        contextdoc = nlp(context)
                        sentences = list(contextdoc.sents)
                        for qas in paragraph['qas']:
                            question = qas['question']
                            q = question.split()[0].lower()
                            if q in question_type:
                                question_type[q]+=1
                            else:
                                question_type[q] = 1
                            id = qas['id']
                            questiondoc = nlp(question)
                            mxsim = 0
                            mxsim_sentence = ""
                            for sentence in sentences:
                                if questiondoc.similarity(sentence) >= mxsim:
                                    mxsim = questiondoc.similarity(sentence)
                                    mxsim_sentence = sentence

                            mxsim_sentence_doc = nlp(mxsim_sentence.text)
                            qid_sent[id] = (str(mxsim_sentence), str(question))

                            sentence_ents = [(ent.text, int(ent.start_char), int(ent.end_char), ent.label_) for ent in
                                             mxsim_sentence_doc.ents]
                            prediction = ""
                            if "when" in str(questiondoc).lower() or "in" == str(questiondoc[0]).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "date" or ent[3].lower() == "time":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "who" in str(questiondoc).lower() or "whom" in str(questiondoc).lower() :
                                for ent in sentence_ents:
                                    if ent[3].lower() == "person" or ent[3].lower() == "org" or ent[3].lower() == "facility":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "many" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "quantity":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "much" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "money" or ent[3].lower() == "quantity" or ent[3].lower() == "percent":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "which" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "person" or ent[3].lower() == "quantity" or ent[3].lower() == "percent":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "where" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "gpe" or ent[3].lower() == "loc" or ent[3].lower() == "facility":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "whose" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "person" or ent[3].lower() == "org" or ent[3].lower() == "facility"\
                                    or ent[3].lower() == "gpe" or ent[3].lower() == "loc":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            elif "percent" in str(questiondoc).lower() or "percentage" in str(questiondoc).lower():
                                for ent in sentence_ents:
                                    if ent[3].lower() == "percent":
                                        prediction = mxsim_sentence.text[ent[1]:ent[2]]
                                        break
                            else:
                                prediction = mxsim_sentence.text
                                prediction = normalize_context(prediction)

                            if prediction == "":
                                prediction = mxsim_sentence.text
                                prediction = normalize_context(prediction)
                            predictions[id] = prediction
                json.dump(predictions, write_file)
        file_qid = open('qid_'+filename, 'w')
        json.dump(qid_sent, file_qid)


start_time = time.time()
all_new_awesomeness('testing.json')
# all_new_awesomeness('development.json')

f = open("qtype.csv",'w')

for k,v in question_type.items():
    t = str(k)+":"+str(v)+"\n"
    f.write(t)

print("--- %s min ---" % ((time.time() - start_time) / 60))