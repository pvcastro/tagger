#!/usr/bin/env bash

SCRIPT=./train.py
DATASET=./dataset/
TRAINING_SET=categories_harem_I
TEST_SET=categories_mini
EMBEDDINGS_FOLDER=./embeddings/
RESULTS_FOLDER=./resultados/scripts/

declare -a tag_schemes=("iob")
declare -a embeddings=("wang2vec_s100")
declare -a cap_dim_values=("0" "1")
declare -a lower_values=("1")
declare -a char_lstm_dim_values=("25")
declare -a word_lstm_dim_values=("100")

for embedding in "${embeddings[@]}"
do

    for i in {4..9}
    do

        for tag_scheme in "${tag_schemes[@]}"
        do

            for cap_dim in "${cap_dim_values[@]}"
            do

                for lower in "${lower_values[@]}"
                do
                .

                    for char_lstm_dim in "${char_lstm_dim_values[@]}"
                    do

                        for word_lstm_dim in "${word_lstm_dim_values[@]}"
                        do

	                        OUTPUT=final/total/resultado_100-epochs_$TRAINING_SET\_$tag_scheme\_$embedding\_cap_dim-$cap_dim\_lower-$lower\_char_lstm_dim-$char_lstm_dim\_word_lstm_dim-$word_lstm_dim\_$i.txt

                            printf "\n*** running $SCRIPT for $OUTPUT ***\n"

                            ~/anaconda2/bin/python $SCRIPT --train $DATASET$TRAINING_SET.txt --test $DATASET$TEST_SET.txt --tag_scheme=$tag_scheme --epochs=100 --all_emb=1 --pre_emb=embeddings/$embedding.txt --cap_dim=$cap_dim --lower=$lower --char_lstm_dim=$char_lstm_dim --word_lstm_dim=$word_lstm_dim --lr_method=sgd-lr_.01 > $RESULTS_FOLDER$OUTPUT

	                    done

                    done

                done

            done

        done

    done

done
