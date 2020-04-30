#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-20 19:02
# * Last modified : 2020-01-10 23:37
# * Filename      : getTempLogs.py
# * Description   :
'''
'''
# **********************************************************
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input',help='input file name ',type = str, default ='data/BGL_without_variables.log')
parser.add_argument('-output',help='output file name ',type = str, default ='middle/BGL_without_variables_for_training.log')
arg = parser.parse_args()
input = arg.input
output = arg.output

f = open(output,'w')
with open(input) as IN:
    for line in IN:
        f.writelines(line.strip()+' ')
f.writelines('\n')

print('input:',input,'\noutput:',output)
