import json
import re
import string
from nltk.corpus import stopwords
import random
import spacy

#using spacy:
#https://spacy.io/docs/usage/lightning-tour#examples-pos-tags

def read_from_file(filename):
    with open(filename) as raw_data:
        json_data = json.load(raw_data)
        return json_data

def answer_questions(question_data):
    for essay in question_data['data']:
        print('next essay')
        title = essay['title']
        for paragraph in essay['paragraphs']:
            context = paragraph['context']
            normalized_context = normalize_context(context)
            normalized_context = removeStops(normalized_context.split())
            for qa in paragraph['qas']:
                # bag_of_word_vectors = get_bag_of_word_vectors(normalized_context)
                normalized_qa = normalize_context(qa['question'])
                solve_question(title, context, normalized_context, qa, normalized_qa)
                # print(answers)

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

def bag_question(question):
    bag_of_words = []

    for ele in question.split():
        if ele not in bag_of_words:
            bag_of_words.append(ele)

    return bag_of_words

def get_similarity(bag_of_words, sentence_2):
    shared_words = 0
    append_words = 0
    last = 0

    for i, ele in enumerate(sentence_2.split()):
        if ele in bag_of_words:
            shared_words+=1
            if i > last:
                last = i

    # print(last)

    for i, _ in enumerate(sentence_2.split()):
        if i > last:
            append_words+=1

    # print(append_words)

    return shared_words + (append_words/len(sentence_2.split()))

def removeStops(sent, isString = False):
    newsent = ""
    if isString:
        sent = sent.split(" ")
    for i in sent:
        if i not in stopwords.words('english'):
            newsent += i + " "
    return newsent.rstrip()

def predict(bag, sent, spacy_sent, single = False):
    # poss = []
    spacy_poss = []

    # for ele in sent.split():
    #     if ele not in bag:
    #         poss.append(ele)

    for ele in spacy_sent.split():
        if ele not in bag:
            spacy_poss.append(ele)

    # if len(poss) > 0 and len(spacy_poss) > 0:
    if len(spacy_poss) > 0:
        if single:
            # return random.choice(poss), random.choice(spacy_poss)
            return random.choice(spacy_poss)
        # return " ".join(poss), " ".join(spacy_poss)
        return " ".join(spacy_poss)
    else:
        if single:
            # return sent.split()[0], spacy_sent.split()[0]
            return spacy_sent.split()[0]
        # return sent, spacy_sent
        return spacy_sent

def solve_question(title, context, normalized_context, qa, normalized_qa):
    id = qa['id']

    spacy_highest_similarity_score = None
    spacy_highest_similarity_sentence = None

    highest_similarity_score = None
    highest_similarity_sentence = None

    # context_arr = normalized_context.split()
    window = ""

    question = removeStops(normalized_qa, True)
    # bag = bag_question(question)
    spacy_bag = spacy_nlp(question)
    window = normalized_context[0:10]
    for i in range(0, len(normalized_context) - 10, 10):
        # window = removeStops(bag_sent)
        spacy_window = spacy_nlp(window)
        # sim = get_similarity(bag, window)
        spacy_sim = spacy_bag.similarity(spacy_window)
        # print(window, ",", spacy_window)

        # if highest_similarity_score is None or highest_similarity_score < sim:
        #     highest_similarity_score = sim
        #     highest_similarity_sentence = window
        # if highest_similarity_score is not None and highest_similarity_score == sim and len(window) < len(highest_similarity_sentence):
        #     highest_similarity_score = sim
        #     highest_similarity_sentence = window

        if spacy_highest_similarity_score is None or spacy_highest_similarity_score < spacy_sim:
            spacy_highest_similarity_score = spacy_sim
            spacy_highest_similarity_sentence = window
        if spacy_highest_similarity_score is not None and spacy_highest_similarity_score == spacy_sim and len(window) < len(spacy_highest_similarity_sentence):
            spacy_highest_similarity_score = spacy_sim
            spacy_highest_similarity_sentence = window

        # del window
        window = normalized_context[i:i+10]

    #preds[id],
    spacy_preds[id] = predict(question, highest_similarity_sentence, spacy_highest_similarity_sentence, False)
    # print(id," : ", preds[id])
    # print(id, " : ", spacy_preds[id])

    # for ele in context_arr:
    #     print(ele)
    #     #this is now every sentence
    #     similarity = get_similarity(question, ele)
    #     print(similarity)
    #     if highest_similarity_score is None or highest_similarity_score < similarity:
    #         highest_similarity_score = similarity
    #         highest_similarity_sentence = ele
    #
    # print(highest_similarity_score)
    # print(highest_similarity_sentence)


    # best_word_vector_score = None
    # best_word_vector = None
    #
    # for word_vector in bag_of_word_vectors:
    #     print word_vector
    #     print qa
    #     diff_in_length = len(word_vector.values()) - len(qa.values())
    #     result = spatial.distance.cosine(qa.values().extend([0]*diff_in_length), word_vector.values())
    #     if(best_word_vector_score is None or best_word_vector_score > result):
    #         best_word_vector_score = result
    #         best_word_vector = word_vector
    #
    # return best_word_vector

def get_NER(string_data):
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://corenlp.run', port=80)
    tags = nlp.ner(string_data)
    # List of tuples
    return tags

preds = {}
spacy_preds = {}
question_data = read_from_file("development.json")

#setup spacy model:
spacy_nlp = spacy.load('en')

answer_questions(question_data)

with open('predicttest.json', 'w') as fp:
    json.dump(preds, fp, sort_keys=True)
with open('spacy_predicttest.json', 'w') as fp:
    json.dump(spacy_preds, fp, sort_keys=True)