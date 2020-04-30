python pipeline.py -i ../templateResult/dataset/hdfs/rawlog.log -t hdfs -o textRank
python pipeline.py -i ../templateResult/dataset/Zookeeper/rawlog.log -t Zookeeper -o textRank
python pipeline.py -i ../templateResult/dataset/HPC/rawlog.log -t HPC -o textRank
python pipeline.py -i ../templateResult/dataset/Proxifier/rawlog.log -t Proxifier -o textRank
python pipeline.py -i ../templateResult/dataset/bgl/rawlog.log -t bgl -o textRank
python pipeline.py -i ../templateResult/dataset/hadoop/rawlog.log -t hadoop -o textRank
python pipeline.py -i ../templateResult/dataset/spark/rawlog.log -t spark -o textRank

