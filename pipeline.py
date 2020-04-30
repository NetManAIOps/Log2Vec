import os
import argparse
import random
import string
import pickle
import gensim
import numpy as np
def cos( vector1, vector2):
    return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def preprocess_log(ipath, opath):
    #preprocess
    processed_log = os.path.join(opath, 'without_variables.log')
    command_for_preprocessing = "python code/preprocessing.py -rawlog %s -o %s"%(ipath, processed_log)
    os.system(command_for_preprocessing)
    return processed_log

def generate_oov(processed_log, opath):
    generate_file_path = os.path.join(opath, 'changed_log')
    if not os.path.exists(generate_file_path):
        os.mkdir(generate_file_path)
    changed_log = os.path.join(generate_file_path, "without_variables.log")
    new_vocab =  os.path.join(generate_file_path, "vocab.txt")
    old_to_new_dict = os.path.join(generate_file_path, "old_new_dict.txt")
    with open(processed_log, 'r') as file:
        logs = file.readlines()
    temp_result = []
    new_words = set()
    old_new_dict = {}
    for log in logs:
        log_in_word = log.split()
        log_length = len(log_in_word)
        target_word = random.randint(0,log_length-1) 
        if log_in_word[target_word] in old_to_new_dict:
            target_word = (target_word + 1) % log_length
        word_length = len(log_in_word[target_word])
        target_letter = random.randint(0, word_length-1)
        change_letter = random.randint(0, 25)
        if log_in_word[target_word][target_letter].lower() == string.ascii_lowercase[change_letter]:
            change_letter = (change_letter + 1) % 26
        old = log_in_word[target_word] 
        log_in_word[target_word] = (log_in_word[target_word][:target_letter] 
                                    + string.ascii_lowercase[change_letter] 
                                    + log_in_word[target_word][target_letter+1:] )
        new_words.add(log_in_word[target_word]+'\n')
        if old in old_new_dict:
            old_new_dict[old].append(log_in_word[target_word])
        else:
            old_new_dict[old] = [log_in_word[target_word]]
        temp_result.append(' '.join(log_in_word)+'\n')
    with open(changed_log, 'w') as file:
        file.writelines(temp_result)
    with open(new_vocab, 'w') as file:
        file.writelines(new_words)
    with open(old_to_new_dict, 'wb') as file:
        for key in old_new_dict:
            old_new_dict[key] = list(set(old_new_dict[key]))
        file.write(pickle.dumps(old_new_dict))
    return new_vocab, old_to_new_dict

def pipeline(processed_log, new_vocab):
    
    # Antonyms&Synonyms Extraction
    sys_output = os.path.join(opath, 'sys.txt')
    ants_output = os.path.join(opath, 'ants.txt')
    command_for_AS_extraction = '''python code/get_syn_ant.py -logs %s -ant_file %s -syn_file %s'''%(processed_log, ants_output, sys_output)
    os.system(command_for_AS_extraction)

    # Relation Triple Extraction
    triplet_log = 'triples.txt'
    command_for_triplet = '''python code/get_triplet.py %s %s'''%(processed_log, triplet_log)
    os.system(command_for_triplet)

    # Semantic Word Embedding
    train_log = os.path.join(opath, 'for_training.log')
    command_for_train = "python code/getTempLogs.py -input %s -output %s" %(processed_log, train_log)
    print(command_for_train)
    os.system(command_for_train)

    # Semantic Word Embedding
    train_model = os.path.join(opath, 'embedding.model')
    vocab = os.path.join(opath, 'embedding.vocab')
    command_for_model = ('''code/LRWE/src/lrcwe -train %s -synonym %s -antonym %s -output %s -save-vocab %s -belta-rel 0.8 -alpha-rel 0.01 -alpha-ant 0.3 -size 32 -min-count 1 -window 2 -triplet %s'''
                         %(train_log, sys_output,
                           ants_output, train_model,
                           vocab, triplet_log))
    os.system(command_for_model)
    print('------')
    print(command_for_model)

    oov_words = os.path.join(opath, 'words.pkl')
    command_for_oov = "python code/mimick/make_dataset.py --vectors %s --w2v-format --output %s"%(train_model, oov_words)
    os.system(command_for_oov)
    print('------')
    print(command_for_oov)

    oov_vector = os.path.join(opath, 'oov.vector')
    learning_rate = 0.006
    epoch = 20
    num_of_layers = 1
    dropout = -1
    hidden_dim = 250
    ch_dim = 36
    command_for_new_embedding = ("python code/mimick/model.py --dataset %s  --vocab %s --output %s --num-epochs %d --learning-rate %f --num-lstm-layers %d --cosine --dropout %f --all-from-mimick --hidden-dim %d --char-dim %d"
                                 %(oov_words, new_vocab, oov_vector, epoch, learning_rate, num_of_layers, dropout, hidden_dim, ch_dim))
    os.system(command_for_new_embedding)
    print('------')
    print(command_for_new_embedding)

    # get log2vec
    log_vector =  os.path.join(opath, 'log.vector')
    command_for_log2vec = " python code/Log2Vec.py -logs %s -word_model %s -log_vector_file %s -dimension 32"%(processed_log, train_model, log_vector)
    os.system(command_for_log2vec)
    print('------')
    print(command_for_log2vec)
    return train_model, oov_vector


def evaluate(word_model_path, old_to_new_dict, oov_vector_path, opath):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word_model_path, binary = False)
    oov_vec = gensim.models.KeyedVectors.load_word2vec_format(oov_vector_path, binary = False)
    with open(old_to_new_dict, 'rb') as file:
        old_new_dict = pickle.loads(file.read())
    total_score = 0
    count = 0
    result = []
    for key in old_new_dict:
        for new_word in old_new_dict[key]:
            if key not in word2vec:
                print(key)
                continue
            score = cos(word2vec[key], oov_vec[new_word])
            total_score += score
            count += 1
            result.append((key, new_word, score))
    score_path = os.path.join(opath, 'score')
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    with open(os.path.join(score_path, 'score'), 'w') as ofile:
        ofile.write('score: '+str(total_score/count)+'\n')
    with open(os.path.join(score_path, 'result.txt'), 'w') as ofile:
        for i in result:
            ofile.write(i[0]+ ' '+i[1]+' '+str(i[2])+'\n')
 
    return total_score/count, result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input_file')
    parser.add_argument('-o', help='output directory', type=str, default=None)
    parser.add_argument('-t', help='log type')
    args = parser.parse_args()
    ipath = args.i
    ipath = os.path.abspath(ipath)
    if args.o == None:
        output_path = 'oov_result/'
        if not os.path.exists(output_path):
            os.mkdir('oov_result')
    else:
        output_path = args.o
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    output_path = os.path.abspath(output_path) 
    log_type = args.t
    opath = os.path.join(output_path, log_type)
    if not os.path.exists(opath):
        os.mkdir(opath)
    processed_log = preprocess_log(ipath, opath)
    new_vocab, old_to_new_dict = generate_oov(processed_log, opath)
    train_model, oov_vector = pipeline(processed_log, new_vocab)
    score, result = evaluate(train_model, old_to_new_dict, oov_vector, opath)
    print('---------')
    print(score)
