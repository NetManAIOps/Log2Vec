# -*- encoding: utf8 -*-
import sys
import os
import logging
from gensim.corpora import WikiCorpus
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)

'''
**************************************************************
*    file: wiki_process.py
*    author: chenbjin(chenbjin@gmail.com)
*    time: 2017年01月04日 星期三 00时29分20秒
*    brief: 
**************************************************************
'''

def parse(filename):
	OUTPATH = '../gen_data/wikicorpus'
	fout = open(OUTPATH, 'w')
	wiki = WikiCorpus(filename, lemmatize=False, dictionary={}, processes=5)
	count = 0
	for text in wiki.get_texts():
		fout.write(" ".join(text) + "\n")
		count = count + 1
		if (count % 10000 == 0):
			logging.info("Save "+str(count) + " articles")
	fout.close()
	logging.info("Finished saved "+str(count) + "articles")

if __name__ == '__main__':
	parse('/home/chenbjin/Workspaces/Thesis/data/wiki/enwiki-latest-pages-articles.xml.bz2')