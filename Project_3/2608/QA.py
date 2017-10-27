import json
import re
import string
from scipy import spatial

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
                answers = solve_question(title, context, normalized_context, qa)
                print answers

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

def get_similarity(sentence_1, sentence_2):
    bag_of_words = set([])
    shared_words = 0

    for ele in sentence_1.split():
        # print "s1 " + ele
        bag_of_words.add(ele)

    for ele in sentence_2.split():
        # print "s2 " + ele
        if ele in bag_of_words:
            shared_words+=1

    return shared_words

def solve_question(title, context, normalized_context, qa):

    question = normalize_context(qa['question'])
    highest_similarity_score = None
    highest_similarity_sentence = None

    context_arr = normalized_context.split('.')
    for ele in context_arr:
        print "potential elements: "
        print ele
        #this is now every sentence
        similarity = get_similarity(question, ele)
        print similarity
        if highest_similarity_score is None or highest_similarity_score < similarity:
            highest_similarity_score = similarity
            highest_similarity_sentence = ele

    print highest_similarity_score
    print highest_similarity_sentence

    raise Exception()

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

question_data = read_from_file("development.json")
answer_questions(question_data)

