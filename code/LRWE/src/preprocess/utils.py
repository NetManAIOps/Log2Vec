#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-02 21:25
# * Last modified : 2019-01-02 21:25
# * Filename      : utils.py
# * Description   :
'''
'''
# **********************************************************
import os
import sys
import logging

def load_vocab(filename):
    logging.info("loading vocab from %s"%filename)
    vocab = {}
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            if len(arr) != 2:
                logging.info("error line: %s"%' '.join(arr))
                continue
            vocab[arr[0]] = int(arr[1])
    return vocab

def load_vocab_for_single_col(filename):
    logging.info("loading vocab from %s"%filename)
    vocab = {}
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            vocab[arr[0]] = 1
    return vocab


def filter_by_freq(vocab, filename, freq=100):
    #vocab = load_vocab(vocab)
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
            if len(arr[0]) < 4 or len(arr[1]) < 4:
            	continue
            if vocab[arr[0]] < freq and vocab[arr[1]] < freq:
                logging.info("low freq: %s"%('\t'.join(arr)))
            else:
                fout.write(line)
    fout.close()

def build_dictionary(filename):
    dictionary = {}
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                logging.info("error line: %s"%('\t'.join(arr)))
                continue
            if arr[0] not in dictionary:
                dictionary[arr[0]] = set()
            dictionary[arr[0]].add(arr[1])
    fout = open(filename+'.di','w')
    for key in sorted(dictionary.keys()):
        for word in dictionary[key]:
            fout.write('%s\t%s\n'%(key,word))
    fout.close()

def build_dictionary_pair(filename):
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
