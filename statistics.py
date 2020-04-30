import json
import argparse
import os
percent_list = [0.1*i for i in range(1,10)]


def preprocess_log(ipath, opath):
    #preprocess
    processed_log = opath
    command_for_preprocessing = "python code/preprocessing.py -rawlog %s -o %s"%(ipath, processed_log)
    os.system(command_for_preprocessing)
    return processed_log

def get_num_of_line(ipath):
    count = 0
    ifile = open(ipath, 'r')
    for line in ifile:
        count += 1
    ifile.close()
    return count

def get_statistics_for_oov(datasets, ipath_list, opath):
    result = {}
    for index in range(len(datasets)):
        dataset = datasets[index]
        path = ipath_list[index]
        result[dataset] = {}
        for train_percent in percent_list:
            result[dataset][train_percent] = {}
            result[dataset][train_percent]["test_oov_line"] = 0
            result[dataset][train_percent]["test_oov_kind_num"] = 0
            result[dataset][train_percent]["test_oov_all_num"] = 0 
            num_of_logs = get_num_of_line(path)
            statistics = {'train':{}, 'test':{}}
            result[dataset][train_percent]["train_line"] = round(train_percent * num_of_logs)
            result[dataset][train_percent]["test_line"] = num_of_logs - round(train_percent * num_of_logs)
            statistics['train']['words'] = {}
            statistics['test']['words'] = {}
            with open(path, 'r') as file:
                count = 1
                for log in file:
                    if len(log) == 0:
                        continue
                    if len(log) == 1:
                        if log[0] == '\n':
                            continue
                    words = log.split()
                    log_with_oov = False
                    if count <= result[dataset][train_percent]["train_line"]:
                        for word in words:
                            statistics['train']['words'][word] = statistics['train']['words'].get(word, 0) + 1
                    else:
                        for word in words:
                            statistics['test']['words'][word] = statistics['test']['words'].get(word, 0) + 1
                            if log_with_oov == False and word not in statistics['train']['words']:
                                log_with_oov = True 
                                result[dataset][train_percent]["test_oov_line"] += 1
                    count += 1
            result[dataset][train_percent]["train_words"] = sum(statistics['train']['words'].values())
            result[dataset][train_percent]["test_words"] = sum(statistics['test']['words'].values())
            result[dataset][train_percent]["test_oov_log_percent"] =  (
                result[dataset][train_percent]["test_oov_line"] / result[dataset][train_percent]['test_line'])
            for word in statistics['test']['words']:
                if word not in statistics['train']['words']:
                    result[dataset][train_percent]["test_oov_kind_num"] += 1
                    result[dataset][train_percent]['test_oov_all_num'] += statistics['test']['words'][word]
            result[dataset][train_percent]['test_oov_word_percent'] = (
                result[dataset][train_percent]['test_oov_all_num'] / result[dataset][train_percent]["test_words"])
            print(dataset, " ", train_percent, " finished", "count:", count)
        print(dataset, " finished")
        print('--------------------------')
    final_result = {}
    for dataset in result:
        final_result[dataset] = {}
        for percent in result[dataset]:
            final_result[dataset][round(percent, 1)] = {}
            final_result[dataset][round(percent, 1)]['train_num_of_logs'] = result[dataset][percent]['train_line']
            final_result[dataset][round(percent, 1)]['train_num_of_words'] = result[dataset][percent]['train_words']
            final_result[dataset][round(percent, 1)]['test_num_of_logs'] = result[dataset][percent]['test_line']
            final_result[dataset][round(percent, 1)]['test_num_of_words'] = result[dataset][percent]['test_words']
            final_result[dataset][round(percent, 1)]['test_oov_num_of_logs'] = result[dataset][percent]['test_oov_line']
            final_result[dataset][round(percent, 1)]['test_oov_num_of_words'] = result[dataset][percent]['test_oov_all_num']
            final_result[dataset][round(percent, 1)]['test_oov_num_of_wordKind'] = result[dataset][percent]['test_oov_kind_num']
            final_result[dataset][round(percent, 1)]['test_oov_log_percent'] = result[dataset][percent]['test_oov_log_percent']
            final_result[dataset][round(percent, 1)]['test_oov_word_percent'] = result[dataset][percent]['test_oov_word_percent']
    with open(opath, 'w') as ofile:
        ofile.write(json.dumps(final_result))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input file')
    parser.add_argument('-t', help='log type')
    parser.add_argument('-o', help='output file')
    parser.add_argument('-preprocess', help='whether to preprocess or not', default=False, type=int)
    args = parser.parse_args()
    input_list = args.i.split(',')
    log_type = args.t.split(',')
    if args.preprocess:
        if not os.path.exists('middle'):
            os.mkdir('middle')
        temp_list = []
        for index in range(len(log_type)):
            logt = log_type[index]
            temp_list.append('middle/'+logt+'.log')
            preprocess_log(input_list[index],temp_list[-1])
        input_list = temp_list
    get_statistics_for_oov(log_type, input_list, args.o)
    if args.preprocess:
        for middle_file in input_list:
            os.remove(middle_file)
