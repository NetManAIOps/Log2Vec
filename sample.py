import random
import argparse
num_of_logs = 10000
def sample_log_from_file(file_name, opath):
    result = set()
    count = 0
    skip = 0
    fin = open(file_name, 'r')
    lines_num = 0
    for line in fin:
        lines_num += 1
    print('total lines:', lines_num)
    fin.seek(0)
    for line in fin:
        if skip == 1 or random.randint(1,lines_num) < num_of_logs*1.3: 
            if line not in result:
                split_index = line.index('\t')
                result.add(line[split_index+1:])
                skip = 0
            else:
                skip = 1
        count += 1
        if(len(result) == num_of_logs):
            break
    if (len(result) < num_of_logs):
        fin.seek(0)
        for line in fin:
            split_index = line.index('\t')
            result.add(line[split_index+1:])
            if len(result) == num_of_logs:
                break
    if len(result) != num_of_logs:
        print('not enough data, num of samples:', str(len(result))) 
    fin.close()    
    with open(opath, 'w') as ofile:
        for i in result:
            log = i.strip()
            ofile.write(log+'\n')
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input file')
    parser.add_argument('-o', help='output file')
    args = parser.parse_args()
    sample_log_from_file(args.i, args.o)
