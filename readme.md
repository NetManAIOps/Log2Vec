## Paper

Our paper is published on The 29th International Conference on Computer Communications and Networks 
([ICCCN 2020](http://www.icccn.org/icccn20/),). The information can be found here:

* Weibin Meng, Ying Liu, Yuheng Huang, Shenglin Zhang, Federico Zaiter, Bingjin Chen, Dan Pei. **A Semantic-aware Representation Framework for Online Log Analysis**.  ICCCN 2020. August 3 - August 6, 2020, Honolulu, Hawaii, USA.

## Dependency

```python
1. nltk, nltk.download("wordnet")
2. spacy, spacy.load("en_core_web_md")
3. progressbar
4. dynet (python3)
```

Quick Start
---

```bash
cd code/LRWE/src/ 
make clean
make 

# prepare the middle results 
python pipeline.py -i data/HDFS.log -t HDFS -o results/
  -i input rawlog
  -t name of logs
  -o output path

# do experiments for log2vec
python log2vec.py -i results -t HDFS
  -i input path
  -t name of logs
```

Directory Structure
---

```bash
.
|-- code
|   |-- get_syn_ant.py
|   |-- get_triplet.py
|   |-- getTempLogs.py
|   |-- Log2Vec.py
|   |-- utils.py
|   |-- kmeans.py
|   |-- LRWE/
|   |-- mimick/
|   |-- preprocessing.py
|
|-- log2vec.py
|-- pipeline.py
|-- statistics.py
|-- sample.py
|-- data
|   |-- HDFS.log #sample data

```

File Descriptions
---

#### preprocessing.py

```sh
#Filter variables in the logs
python code/preprocessing.py -rawlog ./data/BGL.log

  -rawlog：raw logs
```

### Antonyms&Synonyms Extraction
```sh
#Extract antonyms and synonyms 
python code/get_syn_ant.py -logs ./data/BGL_without_variables.log -ant_file ./middle/ants.txt -syn_file ./middle/syns.txt

  -logs: logs
  -ant_file: antonyms
  -syn_file: synonyms
```

### Relation Triple Extraction

```sh
python code/get_triplet.py data/BGL_without_variables.log middle/bgl_triplet.txt

  data/BGL_without_variables.log: logs
  middle/bgl_triples.txt: triples
```

```sh
#If -s is added, temporary saving will be enabled. By default, every 10000 pieces will be saved, named "temp\_" + output\_file
python code/get_triplet.py input_file output_file -s
```

```sh
#If another parameter is added after -s, the number of bars saved per time is modified
python code/get_triplet.py input_file output_file -s 50000 
```


### Semantic Word Embedding

```shell
#Convert log file to single line for training
python code/getTempLogs.py -input data/BGL_without_variables.log -output middle/BGL_without_variables_for_training.log
```

```shell
cd code/LRWE/src/ 
make clean
make #make before you run

#The input file for training is the file obtained in the previous step
./lrcwe -train ../../middle/BGL_without_variables_for_training.log  -synonym ../../middle/syns.txt  -antonym ../../middle/ants.txt -output ../../middle/bgl_words.model -save-vocab ../../middle/bgl.vocab -belta-rel 0.8 - alpha-rel 0.01  -alpha-ant 0.3 -size 32 -min-count 1 -triplet ../../middle/bgl_triplet.txt
```


### Handle OOV Words

```shell
#Read the original vector file
python code/mimick/make_dataset.py --vectors middle/bgl_words.model --w2v-format --output middle/bgl_words.pkl

  --vectors：Results of w2v, the first row is the number of rows and dimensions (can be omitted), the format of each subsequent row is word + word vector: word d1 d2... d32
```


```shell
#Train the new embedding according to oov
python code/mimick/model.py --dataset middle/bgl_words.pkl  --vocab middle/testvocab.txt --output middle/oov.vector

  --dataset：Output of the first step
  --vocab：New words, you can write multiple words in batches, one word per line
  --output：Embedding file for new words
```

### Generate vector for logs 
```shell
python code/Log2Vec.py -logs ./data/BGL_without_variables.log -word_model ./middle/bgl_words.model -log_vector_file ./middle/bgl_log.vector -dimension 32
```



This code was completed by [@Weibin Meng](https://github.com/WeibinMeng), Yuheng Huang and Bingjin Chen in cooperation.

