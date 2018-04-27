#!/usr/bin/env bash

SCRIPT=./train.py
DATASET=./dataset/
TRAINING_SET=categories_harem_I
TEST_SET=categories_mini
EMBEDDINGS_FOLDER=./embeddings/
RESULTS_FOLDER=./resultados/scripts/

declare -a tag_scheme=("iobes")
declare -a embedding=("wang2vec_s100")
declare -a cap_dim=("1")
declare -a lower=("1")
declare -a char_lstm_dim=("25")
declare -a word_lstm_dim=("100")
declare -a optimizer=("sgd")

for i in {0..9}
do

    OUTPUT=final/total/resultado_100-epochs_$TRAINING_SET\_$tag_scheme\_$embedding\_cap_dim-$cap_dim\_lower-$lower\_char_lstm_dim-$char_lstm_dim\_word_lstm_dim-$word_lstm_dim\_optimizer-$optimizer\_$i.txt

    printf "\n*** running $SCRIPT for $OUTPUT ***\n"

    ~/anaconda2/bin/python $SCRIPT --results $RESULTS_FOLDER/resultados.csv --round $i --train $DATASET$TRAINING_SET.txt --test $DATASET$TEST_SET.txt --tag_scheme=$tag_scheme --epochs=100 --all_emb=1 --pre_emb=embeddings/$embedding.txt --cap_dim=$cap_dim --lower=$lower --char_lstm_dim=$char_lstm_dim --word_lstm_dim=$word_lstm_dim --lr_method=$optimizer-lr_.01 > $RESULTS_FOLDER$OUTPUT

done
