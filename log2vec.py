import argparse
import os
import gensim
import numpy as np

def cos( vector1, vector2):
    return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def load_model(filename,is_binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary = is_binary)
    return model

def getLogVector(para):
    '''
        通过日志文件中的单词词向量，组合成每条日志模板的句向量，然后将日志向量保存到文件中，每个向量的index是类别号，从1开始
        return: (template_to_index, index_to_template, template_to_vector)
    '''
    template_file = para['template_file']
    model = load_model(para['word_model'])
    dimension = para['dimension']
    template_vector_file = para['template_vector_file']
    if para['oov_vector']:
        oov_vector = load_model(para['oov_vector'])
    else:
        oov_vector = None
    template_to_index = {}
    index_to_template = {}
    template_to_vector = {}
    template_num = 0
    with open(template_file) as IN:
        for line in IN:
                template_num += 1
    f = open(template_vector_file, 'w')
    f.writelines(str(template_num)+' '+str(para['dimension'])+'\n') #word2vec的模型格式，第一行为单词数&维度
    index = 1
    with open(template_file) as IN:
        for line in IN:
            template = line.strip()
            l = template.split()
            log_length = len(l)
            cur_vector = np.zeros(dimension)
            for word in l:
                if word in model:
                    cur_vector += model[word]
                elif oov_vector:
                    if word in oov_vector:
                        cur_vector += oov_vector[word]
                    else:
                        raise Exception(word + " not in w2v and oov")
                else:
                    log_length -= 1
            cur_vector /= log_length
            template_to_vector[template] = cur_vector
            template_to_index[template] = str(index)
            index_to_template[index] = template
            f.writelines(str(index))
            for v in cur_vector:
                f.writelines(' '+str(v))
            f.writelines('\n')
            index += 1
    f.close()
    return (template_to_index, index_to_template, template_to_vector)

def evaluate(output_original_path, output_withOut_path, output_oov_path):
    original_log2vec = gensim.models.KeyedVectors.load_word2vec_format(output_original_path, binary = False)
    withOut_log2vec = gensim.models.KeyedVectors.load_word2vec_format(output_withOut_path, binary = False)
    oov_log2vec = gensim.models.KeyedVectors.load_word2vec_format(output_oov_path, binary = False)
    output_file = os.path.join(output_path, 'log_similarity.txt')
    ofile = open(output_file, 'w')
    for index in range(1, len(original_log2vec.vocab)+1):
        vec = str(index)
        ofile.write(vec+' ')
        orignal_vec = original_log2vec[vec]
        withOut_vec = withOut_log2vec[vec]
        oov_vec = oov_log2vec[vec]
        ofile.write(str(cos(orignal_vec, withOut_vec))+' ')
        ofile.write(str(cos(orignal_vec, oov_vec))+'\n')
    ofile.close()
    return output_file

def statistics(similarity_result, output_path):
    log_ave_dis = 0
    count = 0
    simi_oov_sum = 0
    simi_without_sum = 0
    ifile = open(similarity_result, 'r')
    for line in ifile:
        simi_oov = float(line.split(" ")[-1])
        simi_oov_sum += simi_oov
        simi_without = float(line.split(" ")[-2])
        simi_without_sum += simi_without
        count += 1
    ifile.close()
    ofile = open(output_path, 'w')
    ofile.write('oov score: '+str(simi_oov_sum/count)+'\n')
    ofile.write('without score: '+str(simi_without_sum/count)+'\n')
    ofile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input_directory')
    parser.add_argument('-t', help='log type')
    args = parser.parse_args()
    log_type = args.t
    iopath = os.path.abspath(args.i)
    iopath = os.path.join(iopath, log_type)
    word_model_path = os.path.join(iopath, 'embedding.model')
    oov_vector_path = os.path.join(iopath, 'oov.vector')
    processed_log = os.path.join(iopath, 'without_variables.log')
    generate_file_path = os.path.join(iopath, 'changed_log')
    changed_log = os.path.join(generate_file_path, "without_variables.log")

    output_path = os.path.join(iopath, 'log2vec')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_original_path = os.path.join(output_path, 'original_log.vector')
    output_withOut_path = os.path.join(output_path, 'removeWord_log.vector')
    output_oov_path = os.path.join(output_path, 'oov_log.vector')

    para_original_log = {}
    para_original_log['template_file'] = processed_log
    para_original_log['word_model'] = word_model_path
    para_original_log['dimension'] = 32
    para_original_log['template_vector_file'] = output_original_path
    para_original_log['oov_vector'] = None

    para_withOut_log = {}
    para_withOut_log['template_file'] = changed_log
    para_withOut_log['word_model'] = word_model_path
    para_withOut_log['dimension'] = 32
    para_withOut_log['template_vector_file'] = output_withOut_path
    para_withOut_log['oov_vector'] = None

    para_oov_log = {}
    para_oov_log['template_file'] = changed_log
    para_oov_log['word_model'] = word_model_path
    para_oov_log['dimension'] = 32
    para_oov_log['template_vector_file'] = output_oov_path
    para_oov_log['oov_vector'] = oov_vector_path

    getLogVector(para_original_log)
    getLogVector(para_withOut_log)
    getLogVector(para_oov_log)
    similarity_result = evaluate(output_original_path, output_withOut_path, output_oov_path)
    statistics(similarity_result, os.path.join(output_path, 'score.txt'))
