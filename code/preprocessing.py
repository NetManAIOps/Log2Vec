#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2017-09-13 12:32
# * Last modified : 2020-01-10 22:35
# * Filename      : preprocessing.py
# * Description   :
'''
'''
# **********************************************************

import os
import re


def getMsgFromNewSyslog(msg):

    '''
    '''
    msg = re.sub('(:(?=\s))|((?<=\s):)', '', msg)
    msg = re.sub('(\d+\.)+\d+', '', msg)
    msg = re.sub('\d{2}:\d{2}:\d{2}', '', msg)
    msg = re.sub(':?(\w+:)+', '', msg)
    msg = re.sub('\.|\(|\)|\<|\>|\/|\-|\=|\[|\]',' ',msg)
    l = msg.split()
    p = re.compile('[^(A-Za-z)]')
    new_msg = []
    for k in l:
        m = p.search(k)
        if m:
            continue
        else:
            new_msg.append(k)
    msg = ' '.join(new_msg)
    return msg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rawlog', help = 'input_file', type = str, default = './data/BGL.log')
    parser.add_argument('-o', help = 'output_file', type = str, default = None)
    args = parser.parse_args()
    input_filename = args.rawlog
    if args.o == None:
        output_filename = args.rawlog[:-4] + '_without_variables.log'
    else:
        output_filename = args.o
    f = open(output_filename,'w')
    with open(input_filename) as IN:
        for line in IN:
            nen = getMsgFromNewSyslog(line)
            if len(nen.split())<=1:
                continue
            f.writelines(nen+'\n')
    print('rawlogs:' + input_filename)
    print('variables have been removed')
    print('logs without variables:' + output_filename)
