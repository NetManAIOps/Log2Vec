#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2020-01-10 22:37
# * Last modified : 2020-01-10 23:27
# * Filename      : get_triples.py
# * Description   :
'''
'''
# **********************************************************
import spacy
import re
import sys, getopt
import json
import gc
#From the dependency tree we build the triplet
object_dict = {'dobj', 'dative', 'attr', 'oprd', 'pobj'}
subject_dict = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl'}
modifier = {'appos', 'nummod', 'amod', 'poss', 'nmod', 'compound'}
record_for_different_path = {}
def record(tri_type, relation):
    if tri_type not in record_for_different_path:
        record_for_different_path[tri_type] = {}
    if relation in record_for_different_path[tri_type]:
        record_for_different_path[tri_type][relation] += 1
    else:
        record_for_different_path[tri_type][relation] = 1
def deal_noun(_object):
    obj_str = ''
    for son in _object.children:
        if son.dep_ == 'amod' or son.dep_ == 'nummod' or son.dep_ == 'compound':
            obj_str += son.text + ' '+ _object.text
    if obj_str == '':
        obj_str += _object.text
    return obj_str
def deal_ADP(_ADP, doc):
    is_DET = False
    if _ADP.text == 'from' or _ADP.text == 'to':
        if doc[_ADP.i+1].pos_ != 'DET':
            return doc[_ADP.i+1].text
    for son in _ADP.children:
        if son.dep_ in object_dict :
            return deal_noun(son)
    return None
def deal_with_verb(_object):
    for son in _object.children:
        if son.dep_ == 'prt':
            return  _object.text + " " + son.text
    return _object.text
def deal_with_root_is_verb(default_subject, root, doc):
    result_triplet = []
    dep_object = None
    dep_subject = None
    dep_prep = None
    dep_advcl = None
    dep_npmod = None
    dep_compound = None
    dep_acomp = None
    dep_auxpass = None
    dep_advmod = None
    stack = []
    already_check = {}
    for son in list(root.children):
        stack.append(son)
        already_check[son.i] = 1
        if son.dep_ in object_dict: #object
            dep_object = son
        elif son.dep_ in subject_dict: #subject
            dep_subject = son
        elif son.dep_ == 'prep': #preposition
            dep_prep = son
        elif son.dep_ == 'advcl':
            dep_advcl = son
        elif son.dep_ == 'npmod' or son.dep_ == 'npadvmod': #another kind of object
            dep_object = son
        elif son.dep_ == 'compound':
            dep_compound = son
        elif son.dep_ == 'acomp':
            dep_acomp = son
        elif son.dep_ == 'auxpass': #passive voice
            dep_auxpass = son
        elif son.dep_ == 'advmod':
            dep_advmod = son
    if dep_object :
        #subject-verb-object
        obj_str = deal_noun(dep_object)
        if dep_subject:
            record('sub-verb-obj', str((dep_subject.text, root.text, obj_str)))
            result_triplet.append((dep_subject.text, root.text, obj_str))
        else: #No subject here, but we have object, use default subject.
            record('default-verb-obj', str((default_subject, root.text, obj_str)))
            result_triplet.append((default_subject, root.text, obj_str))
    if dep_subject == None and dep_object == None:
        #We have noun but no subject and object. Use that noun as subject
        if dep_compound:
            dep_subject = dep_compound
    if dep_advmod != None:
        if dep_subject:
            record('sub-verb-adv', str((dep_subject.text, root.text, dep_advmod.text)))
            result_triplet.append((dep_subject.text, root.text, dep_advmod.text))
        else:
            record('default-verb-adv', str((default_subject, root.text, dep_advmod.text)))
            result_triplet.append((default_subject, root.text, dep_advmod.text))
    if dep_auxpass != None and dep_subject != None:
        #passive voice， get triplet with (subject, auxpass, verb)
        record('sub-auxpass-verb', str((dep_subject.text, dep_auxpass.text, deal_with_verb(root))))
        result_triplet.append((dep_subject.text, dep_auxpass.text, deal_with_verb(root)))
    if dep_acomp:
        #something like system become inactive
        if dep_subject:
            record('dep-verb-acomp', str((dep_subject.text, root.text, dep_acomp.text)))
            result_triplet.append((dep_subject.text, root.text, dep_acomp.text))
        else:
            record('default-verb-acomp', str((default_subject, root.text, dep_acomp.text)))
            result_triplet.append((default_subject, root.text, dep_acomp.text))
    if dep_subject != None :
        for son in list(dep_subject.children):
            if son.i not in already_check:
                already_check[son.i] = 1
                stack.append(son)
            #if son.dep_ in modifier:
                #record('sub-is-modifier', str((dep_subject.text, 'is', son.text)))
                #result_triplet.append((dep_subject.text, 'is', son.text))
    while(len(stack) > 0):
        token = stack[0]
        stack.pop(0)
        for son in list(token.children):
            if son.i not in already_check:
                already_check[son.i] = 1
                stack.append(son)
        if token.pos_ == 'ADP':
            a_is_b = False
            if (len(list(token.ancestors)) > 0 and len(list(token.children)) > 0):
                a = None
                b = None
                for father in token.ancestors:
                    if father.pos_ == 'NOUN':
                        a = father
                        a = deal_noun(a)
                for son in token.children:
                    if son.pos_ == 'NOUN':
                        b = son
                        b = deal_noun(b)
                #something like port XGE3/0/42 of aggregation group BAGG104
                if a != None and b != None:
                    a_is_b = True
                    record('NOUN--NOUN', str((a, token.text, b)))
                    result_triplet.append((a, token.text, b))
    return result_triplet

def get_triplet(default_subject_list, docs, nlpParser, temp_count, result_set):
    result_triplet = []
    index = 0
    docs = nlpParser.pipe(docs)
    for doc in docs:
        for sentence in doc.sents: 
            root = sentence.root
            if root.pos_ == 'VERB':
                result_triplet += deal_with_root_is_verb(default_subject_list[index], root, doc)
        index += 1
    for result in result_triplet:
        if len(result) == 0:
            continue
        result_set.add(result)
        temp_count += 1
    return temp_count, result_set

def batch_get_triplet(buffer_subject, buffer_sentence, nlp, temp_count, result_set):
    results = get_triplet(buffer_subject, buffer_sentence, nlp)
    for result in results:
        if len(result) == 0:
            continue
    for single in result:
        result_set.add(single)
        temp_count += 1
    return temp_count, result_set
def main():
    number_per_save = 10000
    try:
        argc = len(sys.argv)
        inputfile = sys.argv[1]
        outputfile = sys.argv[2]
        temp_save = 0
        if argc >= 4:
            if sys.argv[3] == "-s":
                temp_save = 1
        if argc >= 5:
            try:
                number_per_save = int(sys.argv[4])
            except:
                pass
        inputData = open(inputfile, 'r')
    except:
        print("wrong command, usage: get_triplet inputfile outputfile")
        sys.exit(2)
    log_process = []
    ip_pattern = re.compile(r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}')
    buffer_sentence = []
    buffer_subject = []
    result_set = set()
    temp_count = 0
    #### init
    nlp = spacy.load("en_core_web_md")
    my_infixes = []
    my_infixes.append('\\]')
    my_infixes.append('=')
    my_infixes.append('\\(')
    my_infixes.append('\\)')
    my_infixes.append(';')
    my_infixes.append(',')
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(tuple(my_infixes)).finditer
    disabled = nlp.disable_pipes("ner")
    for log in inputData:
        find_ip = ip_pattern.search(log)
        # blank lines are omitted
        if len(log) <= 3:
            continue
        # We only need part of the log
        if log.find(" : ") != -1:
            temp = log.split(" : ")[-1]
        elif log.find(': ') != -1:
            temp = log.split(': ')[-1]
        elif log.find(';') != -1:
            temp = log.split(';')[-1]
        else:
            if find_ip == None:
                temp = log
            else:
                temp = log[find_ip.end()+1:]
        index = temp.find(']')+1
        # replace some of the words because the model may make some mistakes.
        temp = temp.replace('logged out', 'logouted')
        temp = temp.replace('logged in', 'logined')
        # log_process is a list, with possible default subject and the processed log
        if find_ip == None:
            buffer_sentence.append(temp[index:])
            buffer_subject.append(None)
        else:
            buffer_sentence.append(temp[index:])
            buffer_subject.append(find_ip.group())
        if len(buffer_sentence) > 10000:
            temp_count, result_set = get_triplet(buffer_subject, buffer_sentence, nlp, temp_count, result_set)
            del buffer_sentence
            del buffer_subject
            gc.collect()
            buffer_sentence = []
            buffer_subject = []
        if temp_save and temp_count % number_per_save == 0:
            with open('temp_'+outputfile, 'w') as outputFile:
                for single in result_set:
                    outputFile.write(json.dumps(single))
                    outpuFile.write('\n')
            print('temp save')
    inputData.close()
    if len(buffer_sentence) > 0:
        temp_count, result_set = get_triplet(buffer_subject, buffer_sentence, nlp, temp_count, result_set)
    #output_result
    with open(outputfile, 'w') as outputFile:
        for single in result_set:
            cur_triples = json.dumps(single)[1:-1]
            #[null, "shutting", "NodeCard mCardSernummLctn mCardSernum"]
            words = cur_triples.strip().split(',')
            cur_triples = ''.join(words)
            words = cur_triples.strip().split('"')
            cur_triples = ''.join(words)
            words = cur_triples.strip().split()
            cur_triples = ' '.join(words)
            print(cur_triples)


            outputFile.write(cur_triples)
            #outputFile.write(json.dumps(single))

            outputFile.write("\n")
if __name__ == "__main__":
    main()
