import sys
import math
import logging
import scipy.stats, scipy.spatial
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_model(filename,binary):
	model = Word2Vec.load_word2vec_format(filename, binary=binary)
	return model

def load_wordsim(filename):
	word_pair = []
	simi = []
	with open(filename) as fin:
		for line in fin:
			if line.strip().startswith('#'): continue
			arr = line.strip().split('\t')
			if len(arr) != 3: continue
			word_pair.append((arr[0].lower(), arr[1].lower()))
			simi.append(float(arr[2]))
	return word_pair,simi

def similarity(v1,v2):
	inner = sum([v1[i] * v2[i] for i in range(len(v1))])
	sum1 = math.sqrt(sum([v1[i] * v1[i] for i in range(len(v1))]))
	sum2 = math.sqrt(sum([v2[i] * v2[i] for i in range(len(v2))]))
	if sum1 == 0 or sum2 == 0: return -1
	return inner * 1.0 / (sum1 * sum2)

def similarity_scipy(v1, v2):
	return 1 - scipy.spatial.distance.cosine(v1, v2)

def main():
	filename = sys.argv[1]
	word_pair,simi = load_wordsim('./datasets/wordsim353')
	model = load_model(filename,binary=True)
	new_simi = []
	for pair in word_pair:
		if pair[0] not in model or pair[1] not in model:
			logging.info("%s not in vocab." % pair[0] if pair[0] not in model else pair[1])
			new_simi.append(0.0)
			continue
		new_simi.append(similarity(model[pair[0]], model[pair[1]]))
	res = scipy.stats.spearmanr(simi, new_simi)
	print res


if __name__ == '__main__':
	main()
