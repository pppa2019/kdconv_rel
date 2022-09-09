# Preprocess kdconv dataset

from multiprocessing import Pool
import argparse
import pickle
import random
import os
from typing import DefaultDict
from tqdm import tqdm
import json 
import jieba
import ipdb
from model.utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

# Tokenizer
# tokenizer = Tokenizer('spacy')
is_kg_add_query = True

tokenizer = Tokenizer('jieba')
domain = 'music'
datasets_dir = 'datasets/'
source_dir = os.path.join(datasets_dir, 'kdconv')
kdconv_dir = os.path.join(datasets_dir, f'kdconv_{domain}')
kg_path = os.path.join(source_dir, '{}/kb_{}.json'.format(domain, domain))
with open(kg_path, 'r', encoding='utf-8') as f:
    full_kg = json.load(f)


def expand_conversation(target_max_length, conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length):
    # check：expand也没有问题
    expand_conversations = []
    expand_kgs_index = []
    expand_kgs = []
    expand_kgs_hr_length = []
    expand_kgs_hrt_length = []
    for conversation, kg_index, kg, kg_hr_length, kg_hrt_length in tqdm(zip(conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length)):
        
        # check：看起来在进入下面的expand之前也没有问题。
        for i in range(2, min(len(conversation), target_max_length)):
            expand_conversations.append(conversation[:i])
            expand_kgs_index.append(kg_index[:i])
            expand_kgs.append(kg)
            expand_kgs_hr_length.append(kg_hr_length)
            expand_kgs_hrt_length.append(kg_hrt_length)
        
        for i in range(len(conversation) - target_max_length + 1):
            expand_conversations.append(conversation[i: i+target_max_length])
            expand_kgs_index.append(kg_index[i: i+target_max_length])
            expand_kgs.append(kg)
            expand_kgs_hr_length.append(kg_hr_length)
            expand_kgs_hrt_length.append(kg_hrt_length)
    # print(len(expand_conversations))
    return expand_conversations, expand_kgs_index, expand_kgs, expand_kgs_hr_length, expand_kgs_hrt_length




def load_data(domain, type):
    folder_path = os.path.join(source_dir, domain)
    data_path = os.path.join(folder_path, type+'.json')
    # official baseline process conversation, init kg_dict in every iteration.
    conversations = []
    kgs = []
    kgs_h_length = []
    kgs_hr_length = []
    kgs_hrt_length = []
    kgs_index = []

    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        for conversation in full_data:
            utterences = []
            kg_dict = {tuple((tuple(), tuple(), tuple())):0}
            kg_index = []
            kg = []
            kg_h_length = []
            kg_hr_length = []
            kg_hrt_length = []
            for item in conversation['messages']:
                utterences.append(item['message'])
                kg_index.append([])
                if 'attrs' in item.keys():
                    for attr in item['attrs']:
                        h = ['<soh>'] + jieba.lcut(attr['name'].replace('【', '').replace('】',''))
                        r = ['<sor>'] + jieba.lcut(attr['attrname'].replace('【', '').replace('】',''))
                        t = ['<sot>'] + jieba.lcut(attr['attrvalue'].replace('【', '').replace('】','')) + ['<eot>']
                        k = tuple((tuple(h), tuple(r), tuple(t)))
                        if is_kg_add_query and len(utterences)>=2:
                            utterences[-2] = utterences[-2]+' sep '+''.join(t)
                        if k not in kg_dict:
                            kg_dict[k] = len(kg)
                            # if len(kg) > 30:
                            #     print(kg_dict[k], k)
                            kg.append(list(k[0]+k[1]+k[2]))
                            kg_h_length.append(len(h))
                            kg_hr_length.append(len(h)+len(r))
                            kg_hrt_length.append(len(h)+len(r)+len(t))
                        kg_index[-1].append(kg_dict[k])
                        # print(len(kg_index), len(utterences))
                else:
                    kg_index[-1].append(0)
            conversations.append(utterences)
            kgs.append(kg)
            kgs_h_length.append(kg_h_length)
            kgs_hr_length.append(kg_hr_length)
            kgs_hrt_length.append(kg_hrt_length)
            kgs_index.append(kg_index)
            # check到这里还是没有问题的
    data = {}
    data['conversations'] = conversations
    data['kgs'] = kgs
    data['kgs_h_length'] = kgs_h_length
    data['kgs_hr_length'] = kgs_hr_length
    data['kgs_hrt_length'] = kgs_hrt_length
    data['kgs_index'] = kgs_index
    return data

def load_full_data():
    train = load_data(domain, 'train')
    test = load_data(domain, 'test')
    valid = load_data(domain, 'dev')

    print(f"Train set: {len(train['conversations'])} conversations")
    print(f"Validation set: {len(valid['conversations'])} conversations")
    print(f"Test set: {len(test['conversations'])} conversations")
    return train, valid, test


def tokenize_conversation(lines):
    sentence_list = [tokenizer(line) for line in lines]
    return sentence_list


def pad_sentences(conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length, max_sentence_length=50, max_conversation_length=28):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation, max_sentence_length=max_sentence_length):
        conversation = [pad_tokens(sentence, max_sentence_length) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []
    # print([len(conv) for conv in conversations])
    # print([len(kg_index) for kg_index in kgs_index])
    for idx, conversation in enumerate(conversations):
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
            
            kgs_index[idx] = kgs_index[idx][:max_conversation_length]
        if len(kgs[idx]) > max_conversation_length:
            kgs[idx] = kgs[idx][:max_conversation_length]
            kgs_hr_length[idx] = kgs_hr_length[idx][:max_conversation_length]
            kgs_hrt_length[idx] = kgs_hrt_length[idx][:max_conversation_length]
        
        # if len(conversation) < max_conversation_length:
        #     kgs_hr_length[idx] = kgs_hr_length[idx] + [0]*(max_conversation_length - len(kgs_hr_length[idx]))
        #     kgs_hrt_length[idx] = kgs_hrt_length[idx] + [0]*(max_conversation_length - len(kgs_hrt_length[idx]))
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        kgs[idx] = pad_conversation(kgs[idx], max_sentence_length=max_sentence_length)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


if __name__ == '__main__':

    dataset_dict = DefaultDict(list)

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=50)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=28)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=27000)
    parser.add_argument('--min_vocab_frequency', type=int, default=1)

    # Multiprocess
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())
    
    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers


    vocab = Vocab(tokenizer)
    train, valid, test = load_full_data()

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    for split_type, conv_objects in [('train', train), ('valid', valid), ('test', test)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = os.path.join(kdconv_dir,split_type)
        if not os.path.exists(split_data_dir):
            os.makedirs(split_data_dir)
        

        print(f'Tokenize.. (n_workers={n_workers})')
        def _tokenize_conversation(conv):
            return tokenize_conversation(conv)
        with Pool(n_workers) as pool:
            conversations = list(tqdm(pool.imap(_tokenize_conversation, conv_objects['conversations']),
                                     total=len(conv_objects['conversations'])))
        # ipdb.set_trace()

        kgs = conv_objects['kgs']
        kgs_hr_length = conv_objects['kgs_hr_length']
        kgs_hrt_length = conv_objects['kgs_hrt_length']
        kgs_index = conv_objects['kgs_index']
        # conversations = []
        # for conv in tqdm(conv_objects):
        #     conversations.append(tokenize_conversation(conv))

        
        print('Save Vocabulary...')
        # vocab = Vocab(tokenizer)
        vocab.add_dataframe(conversations)
        for kg in kgs:
            vocab.add_dataframe(kg)
        # 补充词表，避免过多unk的出现， 或许应该把kg的词给加入了才比较对劲儿
        # vocab_file_path = '/data/chenyijie/pretrained/tencent_vocab.txt'
        # with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
        #     vocab_add = vocab_file.readlines()
        #     for i in vocab_add:
        #         vocab.add_word(i.strip())
        vocab.update(max_size=max_vocab_size, min_freq=min_freq)

        

        
        if split_type != 'test':
            conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length = expand_conversation(8, conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length)
            # ipdb.set_trace()
        # split_conv, _, _, _, _ = expand_conversation(8, conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length)
            dataset_dict[split_type] = [[' '.join(sentence)  for sentence in conversation] for conversation in conversations]
            conversation_length = [min(len(conv), max_conv_len)
                                for conv in conversations]
        else:
            dataset_dict[split_type] = [[' '.join(sentence)  for sentence in conversation] for conversation in conversations]
            conversation_length = [min(len(conv), 50)
                                for conv in conversations]
        if split_type != 'test':
            sentences, sentence_length = pad_sentences(
                conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length,
                max_sentence_length=max_sent_len,
                max_conversation_length=max_conv_len)
        else:
            sentences, sentence_length = pad_sentences(
                conversations, kgs_index, kgs, kgs_hr_length, kgs_hrt_length,
                max_sentence_length=max_sent_len,
                max_conversation_length=50)
        print('Saving preprocessed data at', split_data_dir)
        print('total length:', len(conversation_length))
        to_pickle(conversation_length, os.path.join(split_data_dir,'conversation_length.pkl'))
        to_pickle(sentences, os.path.join(split_data_dir,('sentences.pkl')))
        to_pickle(sentence_length, os.path.join(split_data_dir,'sentence_length.pkl'))
        to_pickle(kgs, os.path.join(split_data_dir,'kgs.pkl'))
        to_pickle(kgs_hrt_length, os.path.join(split_data_dir,'kgs_hrt_length.pkl'))
        to_pickle(kgs_hr_length, os.path.join(split_data_dir,'kgs_hr_length.pkl'))
        to_pickle(kgs_index, os.path.join(split_data_dir,'kgs_index.pkl'))
    
    # with open('../CDial-GPT/kdconv.json', 'w', encoding='utf-8') as f:
    #     json.dump(dataset_dict, f, ensure_ascii=False) 

    print('Vocabulary size: ', len(vocab))
    vocab.pickle(os.path.join(kdconv_dir, 'word2id.pkl'), os.path.join(kdconv_dir, 'id2word.pkl'))