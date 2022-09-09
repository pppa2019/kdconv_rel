import json
import os
import ipdb
from matplotlib.pyplot import polar
from numpy import tri
# load data from KdConv
dataset = 'film'
train = json.load(open(os.path.join(dataset, 'train.json'), 'r', encoding='utf-8'))
dev = json.load(open(os.path.join(dataset, 'dev.json'), 'r', encoding='utf-8'))
test = json.load(open(os.path.join(dataset, 'test.json'), 'r', encoding='utf-8'))
kb = json.load(open(os.path.join(dataset, 'kb_{}.json'.format(dataset)), 'r', encoding='utf-8'))

# convert data format
id2passage  = {}
id2query = {}
passage2id = {}
query2id = {}

def trip2passage(tripple):
    return tripple[0]+' __'+tripple[1]+'__ '+tripple[2]

def add_id2query(id2query, query2id, merge_data, tag='tr'):
    for conv_id, data in enumerate(merge_data):
        for q_id, utte in enumerate(data['messages']):
            id2query['{}_{}_{}'.format(tag, str(conv_id), str(q_id))] = utte['message']
            query2id[utte['message']] = '{}_{}_{}'.format(tag, str(conv_id), str(q_id))

def add_kg_dict(id2passage, passage2id, kb):
    id2passage['K_0'] = 'for no knowledge query'
    passage2id['for no knowledge query'] = 'K_0'
    idx = 1
    for _, tripples in kb.items():
        for tripple in tripples:
            id2passage['K_{}'.format(idx)] = tripple[0]+' __'+tripple[1]+'__ '+tripple[2]
            passage2id[tripple[0]+' __'+tripple[1]+'__ '+tripple[2]] = 'K_{}'.format(idx)
            idx += 1
    
add_id2query(id2query, query2id, train, 'tr')
add_id2query(id2query, query2id, dev, 'va')
add_id2query(id2query, query2id, test, 'te')
add_kg_dict(id2passage, passage2id, kb)

# ipdb.set_trace()

# output dukenet format file
split_f = open('kdconv_dukenet/kdconv.split', 'w', encoding='utf-8')
passage_f = open('kdconv_dukenet/kdconv.passage', 'w', encoding='utf-8')
query_f = open('kdconv_dukenet/kdconv.query', 'w', encoding='utf-8')
qrel_f = open('kdconv_dukenet/kdconv.qrel', 'w', encoding='utf-8')
pool_f = open('kdconv_dukenet/kdconv.pool', 'w', encoding='utf-8')
answer_f = open('kdconv_dukenet/kdconv.answer', 'w', encoding='utf-8')

# kdconv.split
for key, _ in id2query.items():
    if key[:2] == 'tr':
        split_f.write(key+'\t'+'train'+'\n')
    elif key[:2] == 'va':
        split_f.write(key+'\t'+'valid'+'\n')
    elif key[:2] == 'te':
        split_f.write(key+'\t'+'test'+'\n')
split_f.close()

# kdconv.query
for key, value in id2query.items():
    query_f.write(key+'\t'+value+'\n')
query_f.close()

# kdconv.passage
for key, value in id2passage.items():
    passage_f.write(key+'\t'+value+'\n')
passage_f.close()

# kdconv.pool
# TODO:确认candidate的选取方式，当前暂定为数据集给出的对话内涉及三元组。
all_data = [train, dev, test]
for dataset in all_data:
    for dial in dataset:
        # try:
        #     candidates = kb[dial['name']]
        # except:
        #     candidates = kb[dial['name'].split('（')[0]]
        # candidates = [passage2id[trip2passage(tripple)] for tripple in candidates]
        candidates = ['K_0']
        for utte in dial['messages']:
            if 'attrs' in utte.keys():
                candidates.extend([passage2id[trip2passage([tripple_dict['name'], tripple_dict['attrname'], tripple_dict['attrvalue']])] for tripple_dict in utte['attrs']])
                # for tripple_dict in utte['attrs']:
                #    candidates.extend([passage2id[trip2passage(tripple)] for tripple in kb[tripple_dict['name']]])
        # candidates = list(set(candidates))
        candidates = ["K_0", 'K_1', 'K_2']
        print('candidate length', len(candidates))
        history = []
        for utte in dial['messages']:
            q_id = query2id[utte['message']]
            hit = []
            if int(q_id.split('_')[-1])%2==0:
                if 'attrs' in utte.keys():
                    
                    hit = [passage2id[trip2passage([tripple_dict['name'], tripple_dict['attrname'], tripple_dict['attrvalue']])] for tripple_dict in utte['attrs']]
                    hit = ['K_0']
                    # for hit_id in hit:
                    #     if hit_id not in candidates:
                    #         candidates.append(hit_id)
                    #     tial_entity = id2passage[hit_id].split(' ')[-1]
                    #     if tial_entity in kb.keys():
                    #         cand_2hop = kb[tial_entity]
                    #         cand_2hop = candidates = [passage2id[trip2passage(tripple)] for tripple in cand_2hop]
                    #         candidates.extend(cand_2hop)
                    for k_id in candidates:
                        if k_id in hit:
                            pool_f.write(q_id+' '+'Q0'+' '+k_id+' 0 1 Label\n')
                            qrel_f.write(q_id+' 0 '+k_id+' 1\n')
                        else:
                            pool_f.write(q_id+' '+'Q0'+' '+k_id+' 1 0 Label\n')
                            qrel_f.write(q_id+' 0 '+k_id+' 0\n')
                else:
                    # pool_f.write(q_id+' '+'Q0'+' K_0 1 0 Label\n')
                    # qrel_f.write(q_id+' 0 '+'K_0'+' 1\n')
                    for k_id in candidates:
                        pool_f.write(q_id+' '+'Q0'+' '+k_id+' 0 1 Label\n')
                        qrel_f.write(q_id+' 0 '+k_id+' 1\n')
            if int(q_id.split('_')[-1])%2==0:
                answer_f.write(';'.join(history)+'\t'+q_id+'\t'+';'.join(hit)+'\t'+utte['message']+'\n')
            history.append(q_id)

