from math import pi
import pickle
import os
import ipdb
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_all(data_dir):
    conversation_length = load_pkl(os.path.join(data_dir, 'conversation_length.pkl'))# the same
    kgs = load_pkl(os.path.join(data_dir, 'kgs.pkl')) # padding is different
    sentences = load_pkl(os.path.join(data_dir, 'sentences.pkl')) # behind with knowledge
    kgs_index = load_pkl(os.path.join(data_dir, 'kgs_index.pkl')) # the same thing 
    hr_len = load_pkl(os.path.join(data_dir, 'kgs_hr_length.pkl'))
    return conversation_length, kgs, sentences, kgs_index, hr_len

dir1 = 'kdconv'
dir2 = 'kdconv_new'

train1 = os.path.join(dir1, 'train')
train2 = os.path.join(dir2, 'train')

data1 = load_all(train1)
data2 = load_all(train2)

ipdb.set_trace()