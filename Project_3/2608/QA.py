import json
import re
import string
from nltk.corpus import stopwords
import random
# from scipy import spatial

def read_from_file(filename):
    with open(filename) as raw_data:
        json_data = json.load(raw_data)
        return json_data

def answer_questions(question_data):
    for essay in question_data['data']:
        title = essay['title']
        for paragraph in essay['paragraphs']:
            context = paragraph['context']
            normalized_context = normalize_context(context)
            for qa in paragraph['qas']:
                # bag_of_word_vectors = get_bag_of_word_vectors(normalized_context)
                solve_question(title, context, normalized_context, qa)
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

def predict(bag, sent, single = False):
    poss = []

    for ele in sent.split():
        if ele not in bag:
            poss.append(ele)


    if len(poss) > 0:
        if single:
            return random.choice(poss)
        return " ".join(poss)
    else:
        if single:
            return sent.split()[0]
        return sent

def solve_question(title, context, normalized_context, qa):

    question = normalize_context(qa['question'])
    id = qa['id']
    highest_similarity_score = None
    highest_similarity_sentence = None
    context_arr = normalized_context.split()
    window = ""
    # print('Question: ', question)
    # print('Context: ', context)
    question = removeStops(question, True)
    bag = bag_question(question)
    bag_sent = context_arr[0:10]
    for i in range(len(context_arr) - 10):
        window = removeStops(bag_sent)
        sim = get_similarity(bag, window)
        # print("Question: ", question)
        # print("Window: ", window)
        # print("Similarity: ", sim)
        if highest_similarity_score is None or highest_similarity_score < sim:
            highest_similarity_score = sim
            highest_similarity_sentence = window
        if highest_similarity_score is not None and highest_similarity_score == sim and len(window) < len(highest_similarity_sentence):
            highest_similarity_score = sim
            highest_similarity_sentence = window
        del bag_sent[0]
        bag_sent.append(context_arr[i+10])
    # print(highest_similarity_score)
    # print(highest_similarity_sentence)
    # print(id)
    preds[id] = predict(question, highest_similarity_sentence, False)
    print(id," : ", preds[id])

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


preds = {}
question_data = read_from_file("development.json")
answer_questions(question_data)

with open('predicttest.json', 'w') as fp:
    json.dump(preds, fp, sort_keys=True)