import sys
import math
import os
import logging
import scipy.stats, scipy.spatial
from sklearn import metrics
from gensim.models.word2vec import Word2Vec
from utils import load_vocab

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_model(filename,binary=True):
	model = Word2Vec.load_word2vec_format(filename, binary=binary)
	return model

def load_questions(vocab,filename,SYN=True):
	question = []
	label = []
	with open(filename) as fin:
		for line in fin:
			arr = line.strip().split('\t')
			if arr[0].strip() not in vocab or arr[1].strip() not in vocab:
				continue
			question.append((arr[0].strip(),arr[1].strip()))
			if SYN == True:
				if arr[2].strip() == "ANT": label.append(0)
				else: label.append(1)
			else:
				if arr[2].strip() == "ANT": label.append(1)
				else: label.append(0)
	return question, label

def similarity(v1,v2):
	inner = sum([v1[i] * v2[i] for i in range(len(v1))])
	sum1 = math.sqrt(sum([v1[i] * v1[i] for i in range(len(v1))]))
	sum2 = math.sqrt(sum([v2[i] * v2[i] for i in range(len(v2))]))
	if sum1 == 0 or sum2 == 0: return -1
	return inner * 1.0 / (sum1 * sum2)

def similarity_scipy(v1, v2):
	return 1 - scipy.spatial.distance.cosine(v1, v2)

def calc_average_precision(model, label, ques):
	sims = []
	for pair in ques:
		sim = similarity(model[pair[0]],model[pair[1]])
		sims.append(sim)
	AUC = metrics.roc_auc_score(label,sims)
	AP = metrics.average_precision_score(label,sims)
	return AP,AUC

def average_precision_test(model):
	DIRNAME = 'datasets/ANT-SYN-TestSet/'
	vocab = load_vocab('../gen_data/model/vocab.1b')
	testfiles = os.listdir(DIRNAME)
	for testfile in testfiles:	
		print DIRNAME+testfile
		ques, label = load_questions(vocab,DIRNAME+testfile,SYN=False)
		print len(ques), len(label)
		ap_ant, auc_ant = calc_average_precision(model, label, ques)	
		ques, label = load_questions(vocab,DIRNAME+testfile,SYN=True)
		ap_syn, auc_syn = calc_average_precision(model, label, ques)	
		print 'ANT:{0},SYN:{1}'.format(ap_ant,ap_syn)
		print 'AUC:{0}'.format(auc_syn)

def main():
	if len(sys.argv) < 2:
		print "Usage: python synant_exam.py model"
		return
	modelfile = sys.argv[1]
	model = load_model(modelfile)
	print modelfile
	average_precision_test(model)

if __name__ == '__main__':
	main()
