import json
from pprint import pprint

with open('training.json') as f:
    train = json.load(f)

for data in train['data']:
    print(data)
    break
    for paragraph in data['paragraph']:
        for qas in paragraph['qas']:
            print(qas)
            break