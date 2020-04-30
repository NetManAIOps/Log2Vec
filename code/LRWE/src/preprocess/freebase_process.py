#-*- coding:utf8 -*-
import gzip
import sys
import re
import json
import logging
from utils import load_vocab
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

def clean(freebasefile, iszip=False):
	logging.info('BEGIN: cleaning freebasefile: %s'%freebasefile)
	fin = gzip.open(freebasefile) if iszip else open(freebasefile)
	fout = open('../gen_data/freebase/freebase.all.clean','w')
	#fname = open('../gen_data/freebase/freebase.name','w')
	count = 0
	entity_pattern = re.compile(r'\"(.*)\"@en')
	freebase_pattern = re.compile(r'<http://rdf.freebase.com/ns/(.*)>')
	for line in fin:
		arr = line.strip().split('\t')
		if len(arr) < 3:
			continue
		head = freebase_pattern.search(arr[0])
		rel  = freebase_pattern.search(arr[1])
		tail = entity_pattern.search(arr[2])

		if not head or not rel or not tail:
			logging.info('badcase:%s'%line.strip())
			continue
		head = head.groups()[0]
		rel = rel.groups()[0]
		tail = tail.groups()[0]
		fout.write('%s\t%s\t%s\n'%(head,rel,tail))
		#if rel == "type.object.name":
		#	fname.write('%s\t%s\t%s\n'%(head,rel,tail))
		if count % 1000 == 0:
			logging.info('count:%d'%count)
		count += 1
	logging.info('END: count:%d'%count)
	fout.close()
	#fname.close()

def relation_cnt(cleanfile, iszip=False):
	logging.info('BEGIN: cleaning freebasefile: %s'%cleanfile)
	fin = gzip.open(cleanfile) if iszip else open(cleanfile)
	count = {}
	for line in fin:
		arr = line.strip().split('\t')
		rel = arr[1]
		if rel not in count:
			count[rel] = 0
		count[rel] += 1
	fout = open('../gen_data/freebase/easyrelation.count.1','w')
	for key in sorted(count.iteritems(), key=lambda x:x[1], reverse=True):
		fout.write('%s\t%d\n'%(key[0],key[1]))
	fout.close()
	logging.info('END: relation count:%d'%len(count))

def tojson(cleanfile, iszip=False):
	logging.info('BEGIN: cleaning freebasefile: %s'%cleanfile)
	fin = gzip.open(cleanfile) if iszip else open(cleanfile)
	fout = open('../gen_data/freebase/freebase.json','w')
	dictionary = {}
	for line in fin:
		arr = line.strip().split('\t')
		if len(arr) != 3:
			logging.info('badcase:%s'%line.strip())
			continue
		head = arr[0]
		rel = arr[1]
		tail = arr[2]
		if head not in dictionary:
			dictionary[head] = {}
		dictionary[head]["MID"] = head
		dictionary[head][rel] = tail
	for key in sorted(dictionary.keys()):
		fout.write('%s\n'%(json.dumps(dictionary[key])))
	fout.close()

def clean_easy_freebase(filename):
	logging.info('BEGIN: cleaning freebasefile: %s'%filename)
	fout = open('../gen_data/freebase/easyfreebase.clean','w')
	count = 0
	with open(filename) as fin:
		for line in fin:
			if line.startswith('!') or line.startswith('"') or line.startswith('#'):
				continue
			arr = line.strip().split('\t')
			if len(arr) < 3:
				continue
			head = arr[0]
			if len(head.split(' ')) > 1 or head.isalpha() == False:
				continue
			tail = arr[2]
			if len(tail.split(' ')) > 1 or tail.isalpha() == False:
				continue
			rel = arr[1]
			fout.write("%s\t%s\t%s\n"%(head,rel,tail))
			count += 1
	fout.close()
	logging.info('END: relation count:%d'%count)

def filter_easy_freebase(filename):
	logging.info('BEGIN: cleaning freebasefile: %s'%filename)
	dictionary = {}
	with open(filename) as fin:
		for line in fin:
			arr = line.strip().split('\t')
			head = arr[0]
			if head not in dictionary:
				dictionary[head] = {}
			rel = arr[1].replace(' ','_')
			dictionary[head][rel] = arr[2]
	fout = open('../gen_data/freebase/easyfreebase.filter','w')
	for key in sorted(dictionary.keys()):
		for rel in dictionary[key]:
			fout.write('%s\t%s\t%s\n'%(key,rel,dictionary[key][rel]))
	fout.close()
	logging.info('END')

def filter_easy_freebase_by_vocab(filename, vocabfile):
	logging.info('BEGIN: cleaning freebasefile: %s'%filename)
	vocab = load_vocab(vocabfile)
	fout = open(filename+'.freq','w')
	with open(filename) as fin:
		for line in fin:
			arr = line.strip().split('\t')
			head = arr[0].lower()
			rel  = arr[1].lower()
			tail = arr[2].lower()
			if head not in vocab or tail not in vocab:
				continue
			if len(head) < 2 or len(tail) < 2:
				continue 
			if vocab[head] < 50 or vocab[tail] < 50:
				continue
			if rel == 'is-a' or '/' in rel or '(' in rel:
				continue
			rel = rel.replace(' ','_')
			fout.write('%s\t%s\t%s\n'%(head,rel,tail))
	fout.close()
	logging.info('END')

def filter_easy_freebase_by_vocab(

def main():
	#freebasefile = '/home/chenbjin/Workspaces/Thesis/data/freebase/en30000'
	#freebasefile = '/home/chenbjin/Workspaces/Thesis/data/freebase/freebase-rdf-latest.gz'
	#clean(freebasefile, iszip=True)
	#tojson('../gen_data/freebase/freebase.clean')
	#relation_cnt('../gen_data/freebase/freebase.clean')
	clean_easy_freebase('/home/chenbjin/Workspaces/Thesis/data/freebase/freebase-easy-latest/facts.txt')
	#relation_cnt('../gen_data/freebase/easyfreebase.clean')
	#filter_easy_freebase('../gen_data/freebase/easyfreebase.clean')
	filter_easy_freebase_by_vocab('../gen_data/freebase/easyfreebase.filter','../gen_data/model/vocab.1b')



if __name__ == '__main__':
	main()