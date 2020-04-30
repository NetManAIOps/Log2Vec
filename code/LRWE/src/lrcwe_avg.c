//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Modified from https://code.google.com/p/word2vec/
// @chenbingjin 2017-01-08

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <map>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define pi 3.1415926535897932384626433832795

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef pair<int,int> intpair;
typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;  //词频
  int *point;    //huffman编码对应内节点的路径
  char *word, *code, codelen; //（词，对应huffman编码，编码长度）
};

char train_file[MAX_STRING], output_file[MAX_STRING], output_theta_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char triplet_file[MAX_STRING], synonym_file[MAX_STRING], antonym_file[MAX_STRING];
struct vocab_word *vocab; //词汇表
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;  //词汇哈希表，便于快速查找，存储每个词在词汇表的索引位置。
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100; //向量维度
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
//分别对应：词的向量，内节点的向量，负采样词的向量，sigmoid函数的近似计算表
real *syn0, *syn1, *syn1neg, *expTable;

clock_t start;
// hierarchical softmax 或者NEG
int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// synonyms and antonyms @chenbingjin 2017-1-8
int flag_triplet = 0, flag_synonym = 0, flag_antonym = 0;
map<int,vector<int> > synonyms;
map<int,vector<int> > antonyms;
char buf[2*MAX_STRING];
real belta_syn = 0.7, belta_ant = 0.2, alpha_syn = 0.025, alpha_ant = 0.001;
// relation from freebase @chenbingjin 2017-1-11
map<string,int> relation2id;
map<int,string> id2relation;
map<int, vector<intpair> > triplets;
vector<vector<real> > syn2;   //relation vector
real belta_rel = 0.8, alpha_rel = 0.01;
int relation_num = 0;
// parameters vector @chenbingjin 2017-1-19
real *syn1lswe, *syn1rswe;


// 随机数
double rand(double min, double max){
  return min + (max-min)*rand()/(RAND_MAX + 1.0);
}
// 正态分布
double normal(double x, double miu,double sigma){
  return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
// 在[min,max]区间内做正态分布采样
double randn(double miu,double sigma, double min ,double max){
  double x,y,dScope;
  do {
    x=rand(min,max);
    y=normal(x,miu,sigma);
    dScope=rand(0.0,normal(miu,miu,sigma));
  }while(dScope>y);
  return x;
}
//负采样算法：带权采样思想。每个词的权重为l(w) = [counter(w)]^(3/4) / sum([counter(u)]^(3/4))，u属于词典D
//  每个词对应一个线段, 将[0,1]等距离划分成10^8，每次生成一个随机整数r，Table[r]就是一个样本。
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  // 遍历词表，统计总权重
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  // 遍历词表，为每个词分配table空间
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 从文件中读取一个词
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);  //读取一个字符
    if (ch == 13) continue; //回车符
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin); //退回一个字符，文件指针左移一位
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
// 返回词的hash值
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a]; //257进制，计算词的hash值
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 返回词在词表中的索引位置，找不到返回-1.
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
// 从文件读一个词，返回词在词汇表的索引位置
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// 将词添加到词汇表
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING; //词的长度不能超MAX_STRING
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;  //初始词频为0
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size; //如果hash值冲突，采用线性探测的开放定址法，顺序向下查找
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
// 词表排序的比较算法cmp：根据词频排序,降序
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 根据词频排序
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    // 出现次数太少的词直接丢弃，min_count 默认5
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      // 重新计算hash值
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn; //总词频
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// 缩小词汇表，移除词频过小的词
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 根据词频构建huffman树，词频越大编码越短
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {  //找第一小
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {  //找第二小
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i; // 编码长度
    vocab[a].point[0] = vocab_size - 2; //?
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}
// 从训练文件中统计每个词的词频
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab(); //如果词汇过多，先删除低频词
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}
// 保存词汇表
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}
//从文件读取词汇，该文件已经统计好每个词的词频
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c); //读取词频，换行符
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }/* 为了测试暂时注释 @chenbingjin 2017-01-08*/
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// 读入同义反义词典
void ReadSynAntonyms(int flag) {
  int i, j, cnt = 0, bcnt = 0;
  char word1[MAX_STRING], word2[MAX_STRING];
  FILE *fin;
  if (flag == 1){
    fin = fopen(synonym_file,"r");
  }else {
    fin = fopen(antonym_file,"r");
  }
  while(!feof(fin)) {
    fgets(buf, 200, fin);
    sscanf(buf,"%s\t%s\n", word1,word2);
    i = SearchVocab(word1);
    if (i == -1) {
      bcnt ++;
      continue;
    }
    j = SearchVocab(word2);
    if (j == -1) {
      continue;
    }
    if (flag == 1) {
      synonyms[i].push_back(j);
    }else {
      antonyms[i].push_back(j);
    }
    cnt ++;
  }
  if (flag == 1) {
    printf("synonyms file total line: %d, words: %d, ignore words: %d\n", cnt, int(synonyms.size()), bcnt);
  }else {
    printf("antonyms file total line: %d, words: %d, ignore words: %d\n", cnt, int(antonyms.size()), bcnt);
  }
  fclose(fin);
}

// 读入triplets
void ReadTriplets() {
  int i, j, lcnt = 0, cnt = 0;
  char word1[MAX_STRING], word2[MAX_STRING], word3[MAX_STRING];
  string relation;
  FILE *fin = fopen(triplet_file, "r");
  relation_num = 0;
  while(!feof(fin)) {
    fgets(buf, 200,fin);
    sscanf(buf,"%s\t%s\t%s\n",word1,word2,word3);
    lcnt ++;
    i = SearchVocab(word1);
    if (i == -1) {
      continue;
    }
    j = SearchVocab(word3);
    if (j == -1) {
      continue;
    }
    relation = word2;
    if (relation2id.count(relation) == 0) {
      relation2id[relation] = relation_num;
      id2relation[relation_num] = relation;
      relation_num ++;
    }
    triplets[j].push_back(make_pair(i,relation2id[relation]));
    cnt ++;
  }
  printf("triplet file total line: %d, relation num: %d, match: %d\n", lcnt, relation_num,cnt);
  fclose(fin);
}

// 初始化网络结构
void InitNet() {
  long long a, b;
  unsigned int cc;
  unsigned long long next_random = 1;
  // 分配词的向量内存，地址是128的倍数
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    // 分配huffman内部节点内存
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化为0向量
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    // 分配参数向量空间
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化为0向量
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  //chenbingjin @2016-01-08
  if (flag_synonym>0 || flag_antonym>0) {
    // 分配syn/antonym词的参数向量空间
    a = posix_memalign((void **)&syn1lswe, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1lswe == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化为0向量
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1lswe[a * layer1_size + b] = 0;
  }
  if (flag_triplet>0) {
    // 分配参数向量空间
    a = posix_memalign((void **)&syn1rswe, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1rswe == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化为0向量
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1rswe[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  // 关系向量初始化
  if (flag_triplet>0) {
    syn2.resize(relation_num);
    for (cc = 0; cc < syn2.size(); cc++) syn2[cc].resize(layer1_size);
    for (a = 0; a < relation_num; a++) {
      for (b = 0; b < layer1_size; b++)
        syn2[a][b] = randn(0,1.0/layer1_size,-6/sqrt(layer1_size),6/sqrt(layer1_size));
    }
  }
  CreateBinaryTree();
}
// 训练模型线程：训练过程
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *addr = (real *)calloc(layer1_size, sizeof(real));  //对应wi+r
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));  //对应Xw
  real *neu1e = (real *)calloc(layer1_size, sizeof(real)); //对应error累加量
  FILE *fi = fopen(train_file, "rb");
  //每个线程对应一段文本
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi); //读一个词，返回其在词汇表的索引位置
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        // 对高频词进行下采样，以概率p丢弃。p = 1-[sqrt(t/f(w))+t/f(w)].但仍保持排序不变
        // 先计算ran = sqrt(t/f(w))+t/f(w)，产生(0,1)上的随机数r，如果r>ran，则丢弃。
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        // 将1000个词当成一个句子
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    // 当前线程处理单词数超过阈值
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    for (c = 0; c < layer1_size; c++) addr[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
      // 随机产生0-5的窗口大小
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        // 上下文词进行向量加和，得到Xw
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
      	// average 向量和取平均
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        // hs，采用huffman
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size; //路径的内部节点
          // Propagate hidden -> output
          // 隐藏层到输出层，计算误差梯度
          // neu1 对应 Xw， syn1对应内部节点的向量0
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2]; //计算内积
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//sigmoid
          // 'g' is the gradient multiplied by the learning rate
          // 内部节点0的梯度(1-d-sigmoid(Xw·0))Xw，g为前面部分
          g = (1 - vocab[word].code[d] - f) * alpha;

          // Propagate errors output -> hidden
          // 反向传播误差，从huffman树传到隐藏层
          // 累加的梯度更新量
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          // 内部节点更新向量
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word; //目标词
            label = 1;   //正样本
          } else {//采样负样本
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2]; //内积
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha; //sigmoid
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2]; //累积误差梯度
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];  //参数向量更新
        }
        // hidden -> in
    	  // 更新上下文几个词语的向量。
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
        //lcwe 训练 @chenbingjin 2017-01-10
        if (flag_synonym > 0 && synonyms[word].size() > 0) {
          for (c = 0; c < layer1_size; c++) neu1[c] = 0;
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          for (unsigned int i = 0; i < synonyms[word].size(); ++i) {
            int t = synonyms[word][i];
            l3 = t * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + l3];
          }
          // average of synonyms
          if (synonyms[word].size() > 1) {
            for (c = 0; c < layer1_size; c++) neu1[c] /= synonyms[word].size();
          }
          if (negative > 0) for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
               target = word;
               label = 1;
            }else {
               next_random = next_random * (unsigned long long)25214903917 + 11;
               target = table[(next_random >> 16) % table_size];
               if (target == 0) target = next_random % (vocab_size - 1) + 1;
               if (target == word) continue;
               label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1lswe[c + l2]; //内积
            if (f > MAX_EXP) g = (label - 1) * alpha_syn;
            else if (f < -MAX_EXP) g = (label - 0) * alpha_syn;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_syn; //sigmoid
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1lswe[c + l2]; //累积误差梯度
            for (c = 0; c < layer1_size; c++) syn1lswe[c + l2] += belta_syn * g * neu1[c];  //参数向量更新
          }
          // 更新同义词集的词向量
          for (unsigned int i = 0; i < synonyms[word].size(); ++i) {
            int t = synonyms[word][i];
            l3 = t * layer1_size;
            for (c = 0; c < layer1_size; c++) syn0[c + l3] += belta_syn * neu1e[c];
          }
        }
        if (flag_antonym > 0 && antonyms[word].size() > 0) {
          for (c = 0; c < layer1_size; c++) neu1[c] = 0;
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          for (unsigned int i = 0; i < antonyms[word].size(); ++i) {
            int t = antonyms[word][i];
            l3 = t * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + l3];
          }
          // average of antonyms
          if (antonyms[word].size() > 1) {
            for (c = 0; c < layer1_size; c++) neu1[c] /= antonyms[word].size();
          }
          if (negative > 0) for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
               target = word;
               label = 1;
            }else {
               next_random = next_random * (unsigned long long)25214903917 + 11;
               target = table[(next_random >> 16) % table_size];
               if (target == 0) target = next_random % (vocab_size - 1) + 1;
               if (target == word) continue;
               label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1lswe[c + l2]; //内积
            if (f > MAX_EXP) g = (label - 1) * alpha_ant;
            else if (f < -MAX_EXP) g = (label - 0) * alpha_ant;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_ant; //sigmoid
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1lswe[c + l2]; //累积误差梯度
            for (c = 0; c < layer1_size; c++) syn1lswe[c + l2] += belta_ant * g * neu1[c];  //参数向量更新
          }
          // 更新反义词集的词向量
          for (unsigned int i = 0; i < antonyms[word].size(); ++i) {
            int t = antonyms[word][i];
            l3 = t * layer1_size;
            for (c = 0; c < layer1_size; c++) syn0[c + l3] += belta_ant * neu1e[c];
          }
        }
        //rcwe 训练 @chenbingjin 2017-01-12
        if (flag_triplet > 0 && triplets[word].size() > 0) {
          for (unsigned int i = 0; i < triplets[word].size(); ++i) {
            int t = triplets[word][i].first, rid = triplets[word][i].second;
            l3 = t * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            for (c = 0; c < layer1_size; c++) addr[c] = syn0[c + l3] + syn2[rid][c];
            for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++) f += addr[c] * syn1rswe[c + l2];
              if (f > MAX_EXP) g = (label - 1) * alpha_rel;
              else if (f < -MAX_EXP) g = (label - 0) * alpha_rel;
              else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_rel;
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1rswe[c + l2];
              for (c = 0; c < layer1_size; c++) syn1rswe[c + l2] += belta_rel * g * addr[c];
            }
            for (c = 0; c < layer1_size; c++) syn0[c + l3] += belta_rel * neu1e[c];//更新词t
            for (c = 0; c < layer1_size; c++) syn2[rid][c] += belta_rel * neu1e[c];//更新向量r
          }
        }
      }
    } else {  //train skip-gram
      //这里很神奇，利用了目标函数的对称性，p(u|w) = p(w|u), u in Context(w). 具体看 http://blog.csdn.net/mytestmy/article/details/26969149
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) { //遍历叶子节点
          f = 0;
          l2 = vocab[word].point[d] * layer1_size; //point是路径上的节点
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2]; //内积
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; //sigmoid
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha; //梯度一部分
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2]; //隐藏层的误差
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1]; //更新内部节点向量
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c]; //更新的是当前上下文的词向量

      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(addr);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}
// 训练模型
void TrainModel() {
  long a, b, c, d;
  FILE *fo, *foo;
  // 默认12个线程
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  if (triplet_file[0] != 0) { ReadTriplets(); flag_triplet = 1; }
  if (synonym_file[0] != 0) { ReadSynAntonyms(1); flag_synonym = 1; }
  if (antonym_file[0] != 0) { ReadSynAntonyms(2); flag_antonym = 1; }
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  // 启动线程
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  // 保存结果
  strcpy(output_theta_file, output_file);
  strcat(output_theta_file,".theta");
  fo = fopen(output_file, "wb");
  foo = fopen(output_theta_file,"w");
  if (classes == 0) {
    // Save the word vectors
    // Save the theta vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    fprintf(foo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      fprintf(foo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      //if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, foo);
      //else for (b = 0; b < layer1_size; b++) fprintf(foo, "%lf ", syn1neg[a * layer1_size + b]);
      for (b = 0; b < layer1_size; b++) fprintf(foo, "%lf ", syn1neg[a * layer1_size + b]);
      fprintf(fo, "\n");
      fprintf(foo, "\n");
    }
    // Save the relation vectors
    // save relation2id
    if (flag_triplet > 0) {
      ofstream rout1("relation.vec");
      ofstream rout2("relation2id");
      rout1 << relation_num << " " << layer1_size << endl;
      for (int a = 0; a < relation_num; a++) {
        rout1 << id2relation[a] << " ";
        for (int b = 0; b < layer1_size; b++) rout1 << syn2[a][b] << " ";
        rout1 << endl;
        rout2 << id2relation[a] << " " << a << endl;
      }
      rout1.close();
      rout2.close();
    }
  } else {
    // Run K-means on the word vectors
    // 对向量进行聚类
    int clcn = classes, iter = 10, closeid;
    // 该类别的数量
    int *centcn = (int *)malloc(classes * sizeof(int));
    // 每个词对应类别
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    // 每个类的中心点
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    // 初始化，每个词分配到一个类
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      // 中心点清零
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      // 计算每个类别求和值
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++; //对应类别的数量加1
      }
      // 遍历所有类别
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b]; //均值
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        // 中心点归一化
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      // 所有词重新分类
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(foo);
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    // 输入文件：已分词的语料
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    // 输出文件：词向量或词聚类
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    // 输入文件：三元组语料
    printf("\t-triplet <file>\n");
    printf("\t\tUse triplets data from <file> to train the model\n");
    // 输入文件：同义语料
    printf("\t-synonym <file>\n");
    printf("\t\tUse synonyms data from <file> to train the model\n");
    // 输入文件：反义语料
    printf("\t-antonym <file>\n");
    printf("\t\tUse antonyms data from <file> to train the model\n");
    // 词向量维度：默认100
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    // 窗口大小：默认5
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    // 词频阈值：默认0，对高频词随机下采样
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    // 采用层次softmax：默认0，不采用
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    // 采用NEG：默认5
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    // 线程数：默认12
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    // 迭代数：默认5
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    // 词频最小阈值：默认5，小于阈值则丢弃
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    // 学习率：默认是0.025(skip-gram),0.05(cbow)
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    // 聚类数：默认0
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    // debug模式：默认2
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    // 二进制存储：默认0，即保存文件时不采用二进制
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    // 保存词汇表
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    // 读取已统计好词频的词汇表
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    // 采用模型：1 CBOW，0 skip-gram，默认1
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    // 示例
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  triplet_file[0] = 0;
  synonym_file[0] = 0;
  antonym_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-triplet", argc, argv)) > 0) strcpy(triplet_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-synonym", argc, argv)) > 0) strcpy(synonym_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-antonym", argc, argv)) > 0) strcpy(antonym_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha-syn", argc, argv)) > 0) alpha_syn = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha-ant", argc, argv)) > 0) alpha_ant = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha-rel", argc, argv)) > 0) alpha_rel = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-belta-syn", argc, argv)) > 0) belta_syn = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-belta-ant", argc, argv)) > 0) belta_ant = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-belta-rel", argc, argv)) > 0) belta_rel = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  printf("alpha:%f, alpha_syn:%f, alpha_ant:%f, alpha_rel:%f\nbelta_syn:%f, belta_ant:%f, belta_rel:%f\n", alpha, alpha_syn, alpha_ant, alpha_rel, belta_syn, belta_ant,belta_rel);
  TrainModel();
  return 0;
}
