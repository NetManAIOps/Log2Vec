#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-08 16:35
# * Last modified : 2020-01-11 13:30
# * Filename      : Log2Vec.py
# * Description   :
'''
'''
# **********************************************************
import argparse


from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
def cos( vector1, vector2):
        #计算两个向量的相似度
        '''
        dot_product = 0.0;
        normA = 0.0;
        normB = 0.0;
        for a,b in zip(vector1,vector2):
            dot_product += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB==0.0:
            return None
        else:
            return dot_product / ((normA*normB)**0.5)
        '''
        return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def load_model(filename,is_binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary = is_binary)
    return model

def getLogVector(para):
    '''
        通过日志文件中的单词词向量，组合成每条日志模板的句向量，然后将日志向量保存到文件中，每个向量的index是类别号，从1开始
        return: (template_to_index, index_to_template, template_to_vector)
    '''
    template_file = para['template_file']
    model = load_model(para['word_model'])
    dimension = para['dimension']
    template_vector_file = para['template_vector_file']
    template_to_index = {}
    index_to_template = {}
    template_to_vector = {}
    template_num = 0
    with open(template_file) as IN:
        for line in IN:
                template_num += 1
    f = open(template_vector_file, 'w')
    f.writelines(str(template_num)+' '+str(para['dimension'])+'\n') #word2vec的模型格式，第一行为单词数&维度
    index = 1
    with open(template_file) as IN:
        for line in IN:
            template = line.strip()
            l = template.split()
            cur_vector = np.zeros(dimension)
            for word in l:
                cur_vector += model[word]
            cur_vector /= len(l)
            template_to_vector[template] = cur_vector
            template_to_index[template] = str(index)
            index_to_template[index] = template
            f.writelines(str(index))
            for v in cur_vector:
                f.writelines(' '+str(v))
            f.writelines('\n')
            index += 1
    return (template_to_index, index_to_template, template_to_vector)


import os
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
class Log2Vec:
    def __init__(self, model_file, is_binary=False):
        #读取现有的vec文件
        print('reading log2vec model')
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = is_binary)
    
        self.model = model
        self.dimension = len(model['1'])
        print(' Log2Vec.dimension:', self.dimension)

    def word_to_most_similar(self, y_word, topn = 1):
        '''
         input:  word
         output: tuple(template_index,similarity)
         与word最相似的，不包括word本身。
        '''
        index = self.model.most_similar(positive = y_word,topn = topn)
        return index 
    
    def vector_to_most_similar(self, y_vector, topn = 1):
        '''
         input: vector
         output: 
         与vector最相似的word，因为预先不知道vector对应的words，所以包含其本身。top1应该是vector对应的word。
        '''
        temp_dict = {}
        for t in self.vector_template_tuple:
            template_index = t[1]
            vector = t[0]
            temp_dict[template_index] = self.cos(y_vector, vector)
        sorted_final_tuple=sorted(temp_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
    
    
    def cos(self, vector1, vector2):
        #计算两个向量的相似度
        '''
        dot_product = 0.0;
        normA = 0.0;
        normB = 0.0;
        for a,b in zip(vector1,vector2):
            dot_product += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB==0.0:
            return None
        else:
            return dot_product / ((normA*normB)**0.5)
        '''
        return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        
        
        
    def get_cosine_matrix(self, _matrixB):
        '''
            矩阵矢量化操作，按行计算余弦相似度
            返回值RES为A矩阵每行对B矩阵每行向量余弦值
            RES[i,j] 表示A矩阵第i行向量与B矩阵第j行向量余弦相似度
        '''
        _matrixA = self.template_matrix
        _matrixA_matrixB = _matrixA * _matrixB.reshape(len(_matrixB),-1)
        # 按行求和，生成一个列向量, 即各行向量的模
        _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum())
        _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum())
        return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

    def vector_to_most_similar_back(self, vectorB, topn = 1):
        '''
         input: vector
         output: 
         与vector最相似的word，因为预先不知道vector对应的words，所以包含其本身。top1应该是vector对应的word。
        '''
        cosine_matrix = self.get_cosine_matrix(vectorB)
        sort_dict = {}
        for i, sim in enumerate(cosine_matrix):
            template_num = str(i+1)
            sort_dict[template_num] = sim
        sorted_final_tuple=sorted(sort_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
        
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-logs', help='log file', type=str, default='./data/BGL_without_variables.log')
    parser.add_argument('-word_model', help='word_model', type=str, default='./middle/bgl_words.model')
    parser.add_argument('-log_vector_file', help='template_vector_file', type=str, default='./middle/bgl_log.vector')
    #parser.add_argument('-template_num', help='template_num', type=int, default=373)
    parser.add_argument('-dimension', help='dimension', type=int, default=32)
    args = parser.parse_args()

    para = {
        'template_file' : args.logs,
        'word_model': args.word_model,
        'template_vector_file': args.log_vector_file,
        #'template_num':args.template_num,
        'dimension':args.dimension
    }
    print('log input:', args.logs)
    print('word vectors input:', args.word_model)
    print('log vectors output:', args.log_vector_file)
    getLogVector(para)
    print('end~~')












