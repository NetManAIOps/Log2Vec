#!/bin/bash
TYPE=$1
IDX=$2
BINARY=1
ITER_NUM=3
VEC_NUM=200
BELTA_ANT=0.8
BELTA_SYN=$3
BELTA_REL=0.8
ALPHA_SYN=$4
ALPHA_ANT=0.025
ALPHA_REL=0.001
TRAIN_PATH="../gen_data/wikicorpus.1b"
TRIPLET_PATH="../gen_data/easyfreebase.clean.freq.rev"
#TRIPLET_PATH="../gen_data/wd18_train.txt.filter"
SYNONYM_PATH="../gen_data/synonyms.wd.ppdb.outjoin.filter"
ANTONYM_PATH="../gen_data/antonyms.wd.pair"
VOCAB_PATH="../gen_data/model/lswe.vocab.1b"

if [ $# -lt 1 ];then
	echo "Usage:./lrcwe_run.sh SAR 0"
	exit
fi

if [ $TYPE = "S" ];then
	echo "S training ..."
	MODEL_PATH="../gen_data/model/avg-lswe-cbow-${VEC_NUM}-model.s.${ITER_NUM}.${BELTA_SYN}.${ALPHA_SYN}.${IDX}.bin"
	./lrcwe -train ${TRAIN_PATH} -synonym ${SYNONYM_PATH} -output ${MODEL_PATH} -save-vocab ${VOCAB_PATH} -belta-syn ${BELTA_SYN} -alpha-syn ${ALPHA_SYN} -size ${VEC_NUM} -window 5 -sample 1e-4 -negative 5 -hs 0 -binary ${BINARY} -cbow 1 -iter ${ITER_NUM}
	exit
elif [ $TYPE = "A" ];then
	echo "A training ..."
	MODEL_PATH="../gen_data/model/lswe-cbow-${VEC_NUM}-model.a.wd2.${ITER_NUM}.${BELTA_ANT}.${ALPHA_ANT}.${IDX}.bin"
	./lrcwe -train ${TRAIN_PATH} -antonym ${ANTONYM_PATH} -output ${MODEL_PATH} -save-vocab ${VOCAB_PATH} -belta-ant ${BELTA_ANT} -alpha-ant ${ALPHA_ANT} -size ${VEC_NUM} -window 5 -sample 1e-4 -negative 5 -hs 0 -binary ${BINARY} -cbow 1 -iter ${ITER_NUM}
	exit
elif [ $TYPE = "R" ];then
	echo "R training ..."
	MODEL_PATH="../gen_data/model/lswe-cbow-${VEC_NUM}-model.r.wd18.${ITER_NUM}.${BELTA_REL}.${ALPHA_REL}.${IDX}.bin"
	./lrcwe -train ${TRAIN_PATH} -triplet ${TRIPLET_PATH} -output ${MODEL_PATH} -save-vocab ${VOCAB_PATH} -belta-rel ${BELTA_REL} -alpha-rel ${ALPHA_REL} -size ${VEC_NUM} -window 5 -sample 1e-4 -negative 5 -hs 0 -binary ${BINARY} -cbow 1 -iter ${ITER_NUM}
	exit
elif [ $TYPE = "SA" ];then
	echo "R training ..."
	MODEL_PATH="../gen_data/model/lswe-cbow-${VEC_NUM}-model.sa.${ITER_NUM}.${BELTA_SYN}.${ALPHA_SYN}.${BELTA_ANT}.${ALPHA_ANT}.${IDX}.bin"
	./lrcwe -train ${TRAIN_PATH} -synonym ${SYNONYM_PATH} -antonym ${ANTONYM_PATH} -output ${MODEL_PATH} -save-vocab ${VOCAB_PATH} -belta-syn ${BELTA_SYN} -alpha-syn ${ALPHA_SYN} -belta-ant ${BELTA_ANT} -alpha-ant ${ALPHA_ANT} -size ${VEC_NUM} -window 5 -sample 1e-4 -negative 5 -hs 0 -binary ${BINARY} -cbow 1 -iter ${ITER_NUM}
	exit
elif [ $TYPE = "SAR" ];then
	echo "SAR training ..."
	MODEL_PATH="../gen_data/model/lswe-cbow-${VEC_NUM}-model.sar.${ITER_NUM}.${BELTA_SYN}.${ALPHA_SYN}.${BELTA_ANT}.${ALPHA_ANT}.${BELTA_REL}.${ALPHA_REL}.${IDX}.bin"
	./lrcwe -train ${TRAIN_PATH} -triplet ${TRIPLET_PATH} -synonym ${SYNONYM_PATH} -antonym ${ANTONYM_PATH} -output ${MODEL_PATH} -save-vocab ${VOCAB_PATH} -belta-rel ${BELTA_REL} -alpha-rel ${ALPHA_REL} -belta-syn ${BELTA_SYN} -alpha-syn ${ALPHA_SYN} -belta-ant ${BELTA_ANT} -alpha-ant ${ALPHA_ANT} -size ${VEC_NUM} -window 5 -sample 1e-4 -negative 5 -hs 0 -binary ${BINARY} -cbow 1 -iter ${ITER_NUM}
	exit
else echo "Type error"
fi
