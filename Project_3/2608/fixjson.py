import json

with open('predicttest.json') as raw_data:
    json_data = json.load(raw_data)

jcopy = {}

for i in json_data:
    sent = json_data[i]
    line = ''
    for s in sent:
        line += s + ' '
    jcopy[i] = line

with open('predicttest2.json', 'w') as fp:
    json.dump(jcopy, fp, sort_keys=True)