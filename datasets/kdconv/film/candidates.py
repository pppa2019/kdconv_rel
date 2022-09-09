import json
import ipdb
f = open('dev.json', 'r', encoding='utf-8')
data = json.load(f)
test = data[0]
knowledge_pool = []
knowledge_len = []
for test in data:
    for message in test['messages']:
        if 'attrs' in message:
            knowledge_len.append(len(message['attrs']))
            for tripple in message['attrs']:
                knowledge = tripple['name']+'\t'+tripple['attrname'] +'\t' +tripple['attrvalue']
                if knowledge not in knowledge_pool:
                    knowledge_pool.append(knowledge)
        else:
            knowledge_len.append(0)
ipdb.set_trace()