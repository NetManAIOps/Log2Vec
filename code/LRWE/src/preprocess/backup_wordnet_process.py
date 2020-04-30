#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-02 18:18
# * Last modified : 2019-01-02 21:25
# * Filename      : wordnet_process_mwb.py
# * Description   :
'''
通过wordnet找模板中的近义词和反义词，并且保存到文件中
'''
# **********************************************************
from nltk.corpus import wordnet
from itertools import chain
from utils import load_vocab
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO,filename='wd.log')

def build_dictionary_wordnet(vocab_file):
    synonyms = {}
    antonyms = {}
    with open(vocab_file) as fin:
        for line in fin:
            if line.strip().startswith('<'):
                continue
            word = line.strip().split(' ')[0]
            synsets = wordnet.synsets(word)
            if len(synsets) == 0:
                logging.info('Not find: %s'%word)
                continue
            synonyms[word] = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))
            antsets = set()
            for syn in synsets:
                for lem in syn.lemmas():
                    if lem.antonyms():
                        antsets.add(lem.antonyms()[0].name())
            if len(antsets) == 0:
                logging.info('No antonyms: %s'%word)
                continue
            antonyms[word] = antsets

    fout = open('../gen_data/synonyms.wd','w')
    for key in sorted(synonyms.keys()):
        for syn in synonyms[key]:
            fout.write('%s\t%s\n'%(key,syn))
    fout.close()
    fout = open('../gen_data/antonyms.wd','w')
    for key in sorted(antonyms.keys()):
        for ant in antonyms[key]:
            fout.write('%s\t%s\n'%(key,ant))
    fout.close()

def filter_wordnet(wdfile):
    fout = open(wdfile+'.filter','w')
    with open(wdfile) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2 or len(arr[0]) < 4 or len(arr[1]) < 4:
                continue
            arr[1] = arr[1].lower()
            if arr[1].find('_') != -1:
                continue
            fout.write('%s\t%s\n'%(arr[0],arr[1]))
    fout.close()

def filter_by_freq(filename, freq=100):
    vocab = load_vocab('../gen_data/vocab.1b')
    fout = open(filename+'.f'+str(freq),'w')
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                logging.info("error line: %s"%('\t'.join(arr)))
                continue
            if arr[0] not in vocab or arr[1] not in vocab:
                logging.info("not in vocab : %s"%('\t'.join(arr)))
                continue
            if vocab[arr[0]] < freq and vocab[arr[1]] < freq:
                logging.info("low freq: %s"%('\t'.join(arr)))
            else:
                fout.write(line)
    fout.close()

def complete_pair_dictionary(filename):
    dictionary = {}
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                logging.info("error line: %s"%('\t'.join(arr)))
                continue
            if arr[0] not in dictionary:
                dictionary[arr[0]] = set()
            if arr[1] not in dictionary:
                dictionary[arr[1]] = set()
            dictionary[arr[0]].add(arr[1])
            dictionary[arr[1]].add(arr[0])
    fout = open(filename+'.pair','w')
    for key in sorted(dictionary.keys()):
        for word in dictionary[key]:
            fout.write('%s\t%s\n'%(key,word))
    fout.close()

def main():
    build_dictionary_wordnet('../gen_data/model/vocab.1b')
    filter_wordnet('../gen_data/synonyms.wd')
    filter_wordnet('../gen_data/antonyms.wd')
    filter_by_freq('../gen_data/synonyms.wd.filter',freq=100)
    filter_by_freq('../gen_data/antonyms.wd.filter',freq=100)
    complete_pair_dictionary('../gen_data/synonyms.wd.filter')
    complete_pair_dictionary('../gen_data/antonyms.wd.filter')
    logging.info("Done")

if __name__ == '__main__':
    main()
