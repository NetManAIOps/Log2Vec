python statistics.py -i ../templateResult/dataset/HPC/rawlog.log,../templateResult/dataset/Proxifier/rawlog.log,../templateResult/dataset/Zookeeper/rawlog.log,../templateResult/dataset/bgl/rawlog.log,../templateResult/dataset/100wExample/rawlog.log,../templateResult/dataset/hadoop/rawlog.log,../templateResult/dataset/hdfs/rawlog.log,../templateResult/dataset/spark/rawlog.log -t HPC,Proxifier,Zookeeper,bgl,switch,hadoop,hdfs,spark -o oov_result/statistics_after_preprocess.txt -preprocess 1

python statistics.py -i ../templateResult/dataset/HPC/rawlog.log,../templateResult/dataset/Proxifier/rawlog.log,../templateResult/dataset/Zookeeper/rawlog.log,../templateResult/dataset/bgl/rawlog.log,../templateResult/dataset/100wExample/rawlog.log,../templateResult/dataset/hadoop/rawlog.log,../templateResult/dataset/hdfs/rawlog.log,../templateResult/dataset/spark/rawlog.log -t HPC,Proxifier,Zookeeper,bgl,switch,hadoop,hdfs,spark -o oov_result/statistics_before_preprocess.txt -preprocess 0


