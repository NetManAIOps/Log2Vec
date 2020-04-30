#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-17 10:19
# * Last modified : 2019-01-17 12:41
# * Filename      : getSimilarityPairs.py
# * Description   :
'''
'''
# **********************************************************
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
def load_model(filename,is_binary=False):
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary = is_binary)
        return model

template_file = '../../middle/bgl_log_20w.template'
model_normal_sa = load_model('../../model/bgl_log_20w.model')
template_vector_file = '../../model/bgl_log_20w.template_vector'
template_to_index = {}
index_to_template = {}
template_to_vector = {}
template_list = []
index = 1
with open(template_file) as IN:
    for line in IN:
        template = line.strip()
        template_list.append(template)
        l = template.split()
        cur_vector = np.zeros(128)
        for word in l:
            cur_vector += model_normal_sa[word]
        cur_vector /= len(l)
        template_to_vector[template] = cur_vector
        template_to_index[template] = str(index)
        index_to_template[index] = template
        index += 1

#读取tempalte对应的向量
model_template_index = load_model(template_vector_file)
model_list = [model_template_index]#, model_no_sa, model_changed_ant_02, model_changed_a_01, model_a_01_nosyn]
row_names = ['有近义词反义词']#,'无近义词反义词','反义词学习率0.2','反义词学习率0.1','反义词学习率0.1无近义词']

c = template_to_index['RAS KERNEL error FATAL']

f = open('sim_paris.txt','w')
for i,template1 in enumerate(template_list):
    cur_dict = {}
    for j , template2 in enumerate(template_list):
        if i == j :
            continue
        index_1 = template_to_index[template1]
        index_2 = template_to_index[template2]
        sim = model_template_index.wv.similarity(index_1,index_2)
        cur_dict[template2] = sim
    sorted_tuple = sorted(cur_dict.items(),key=lambda asd:asd[1] ,reverse=True)
    f.writelines(str(template_to_index[template1])+' '+template1+'\n')
    for t in sorted_tuple[:5]:
        f.writelines('\t'+str(t[1])+' '+ str(template_to_index[t[0]])+' '+t[0]+'\n')
    f.writelines('\n')

f.close()
print('end')




