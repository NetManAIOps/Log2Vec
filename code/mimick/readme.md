# 说明（temp）

### Pipeline

```shell
python mimick/make_dataset.py --vectors mimick/testdir/test.vector --w2v-format --output mimick/testdir/testoutput.pkl
```

--vectors：w2v的结果，第一行是行数和维数（可以省去），后面每一行格式为单词+词向量：word d1 d2 ... d32

--output：第一步进行inference后的对象数据集

```shell
python mimick/model.py --dataset mimick/testdir/testoutput.pkl  --vocab mimick/testdir/testvocab.txt --output mimick/testdir/modeloutput
```

--dataset：第一步的output

--vocab：新的单词，每行一个词

--output：新的单词的embedding文件

### TODO

现在output会多出来一个单词的向量，不知道原因。（例如vocab里面本来放了五个词，但是output里面会有六个。。。）







