# -*- coding: utf-8 -*-
# @Time        : 2019/7/2 15:49
# @Author      : panxiaotong
# @Description : extract features and labels

import argparse
import jieba
import random

parser = argparse.ArgumentParser(description='segmentation and extract features from file')
parser.add_argument('--input', type=str, default='./nlp/mi_pv.dat', help='input file')
parser.add_argument('--trainratio', type=int, default=80, help='train set ratio')
parser.add_argument('--trainset', type=str, default='./nlp/train.dat', help='train output file')
parser.add_argument('--validationset', type=str, default='./nlp/val.dat', help='validation output file')
args = parser.parse_args()

sample_list = []
label_list = []
word_dict = {}
label_dict = {}
text_total_length = 0
with open(args.input, 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        if len(elements) < 2:
            continue
        query = elements[0]
        seg_list = [item for item in jieba.cut(query, cut_all=False)]
        id_list = []
        for word in seg_list:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            id_list.append(word_dict[word])
        sample_list.append(id_list)
        text_total_length += len(id_list)
        label = elements[1]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        label_list.append(label_dict[label])
    f.close()
print('word size is %d, label size is %d' % (len(word_dict), len(label_dict)))

'''
sample_length = len(sample_list)
avg_length = int(text_total_length / sample_length) + 1
print('avg length is %d' % avg_length)
validation_size = int((100 - args.trainratio) * sample_length / 100)
validation_idx_list = random.sample(range(sample_length), validation_size)

train_f = open(args.trainset, 'w')
validation_f = open(args.validationset, 'w')
for idx, sample in enumerate(sample_list):
    if idx in validation_idx_list:
        if len(sample) < avg_length:
            for i in range(avg_length - len(sample)):
                sample.append(0)
        validation_f.write((',').join([str(item) for item in sample[:avg_length]]) + '\t' + str(label_list[idx]) + '\n')
    else:
        if len(sample) < avg_length:
            for i in range(avg_length - len(sample)):
                sample.append(0)
        train_f.write((',').join([str(item) for item in sample[:avg_length]]) + '\t' + str(label_list[idx]) + '\n')
train_f.close()
validation_f.close()
'''