import sys
import math
import logging
import scipy.stats, scipy.spatial
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_model(filename,binary=True):
	model = Word2Vec.load_word2vec_format(filename, binary=binary)
	return model

def load_GRE(filename):
	question = []
	answer = []
	with open(filename) as fin:
		for line in fin:
			words = []
			arr = line.strip().split('::')
			if len(arr) != 2:
				print "error line:%s"%line
				continue
			answer.append(arr[1].strip())
			arr = arr[0].strip().split(':')
			words.append(arr[0].strip())
			for word in arr[1].strip().split(' '):
				words.append(word.strip())
			question.append(words)
	return question, answer

def similarity(v1,v2):
	inner = sum([v1[i] * v2[i] for i in range(len(v1))])
	sum1 = math.sqrt(sum([v1[i] * v1[i] for i in range(len(v1))]))
	sum2 = math.sqrt(sum([v2[i] * v2[i] for i in range(len(v2))]))
	if sum1 == 0 or sum2 == 0: return -1
	return inner * 1.0 / (sum1 * sum2)

def similarity_scipy(v1, v2):
	return 1 - scipy.spatial.distance.cosine(v1, v2)

def main():
	if len(sys.argv) < 2:
		print "Usage: python gre_exam.py model"
		return
	modelfile = sys.argv[1]
	model = load_model(modelfile)
	ques, ans = load_GRE('datasets/GRE.txt')
	print len(ques), len(ans)
	bcnt = 0	
	cnt = 0
	for i in range(len(ans)):
		answ = ans[i]
		quew = ques[i][0]
		#print '%s\t%s\t%s'%(ques[i][0],ans[i],','.join(ques[i][1:]))
		if quew not in model:
			print 'ques not find: %s'%quew
			bcnt += 1
			continue
		sim = 1.0
		resw = None
		for word in ques[i][1:]:
			if word not in model:
				print 'ans not find: %s'%word
				continue
			new_sim = similarity(model[quew], model[word])
			#print "word: %s, sim:%f"%(word, new_sim)
			if new_sim < sim:
				sim = new_sim
				resw = word
		if resw == answ:
			cnt += 1
	#	else:
	#		print 'wrong: %s\t%s\t%s'%(ques[i][0],ans[i],','.join(ques[i][1:]))
	#		print 'ans:%s'%resw
	print "cnt: %d, total: %d, accuracy: %f" %(cnt, len(ans), cnt*1.0/len(ans))
	print bcnt

if __name__ == '__main__':
	main()
