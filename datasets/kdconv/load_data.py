import json
from transformers import BertTokenizer
import ipdb
test_f = open('film/test.json', 'r', encoding='utf8')
data = json.load(test_f)
kb_f = open('film/kb_film.json', 'r', encoding='utf8')
kb = json.load(kb_f)
tokenizer = BertTokenizer(vocab_file='/data/chenyijie/pretrained/CDial-GPT_LCCC-base/vocab.txt')

ipdb.set_trace()