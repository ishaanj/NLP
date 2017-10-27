import json
import re
import string

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
                answers = solve_question(title, context, normalized_context, qa)

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

def solve_question(title, context, normalized_context, qa):
    context_arr = normalized_context.split()
    window_size = 10
    word_dict

    if len(context_arr) > window_size:

        for i in range(0, len(context_arr)-window_size):
            #now, use sliding window approach
            #window goes from i to i+10, just count up all the words in the window
            window = context_arr[i:i+window_size]




question_data = read_from_file("development.json")

