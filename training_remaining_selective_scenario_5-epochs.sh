#!/usr/bin/env bash

SCRIPT=./train.py
DATASET=./dataset/
TRAINING_SET=filtered_harem_I
TEST_SET=filtered_mini
EMBEDDINGS_FOLDER=./embeddings/
RESULTS_FOLDER=./resultados/scripts/

declare -a tag_schemes=("iob")
declare -a embeddings=("wang2vec_s100")
declare -a cap_dim_values=("0" "1")
declare -a lower_values=("1")
declare -a char_lstm_dim_values=("25" "50")
declare -a word_lstm_dim_values=("100" "200")

for embedding in "${embeddings[@]}"
do

    for i in {0..9}
    do

        for tag_scheme in "${tag_schemes[@]}"
        do

            for cap_dim in "${cap_dim_values[@]}"
            do

                for lower in "${lower_values[@]}"
                do

                    for char_lstm_dim in "${char_lstm_dim_values[@]}"
                    do

                        for word_lstm_dim in "${word_lstm_dim_values[@]}"
                        do

	                        OUTPUT=cap_lower/resultado_5-epochs_$TRAINING_SET\_$tag_scheme\_$embedding\_cap_dim-$cap_dim\_lower-$lower\_char_lstm_dim-$char_lstm_dim\_word_lstm_dim-$word_lstm_dim\_$i.txt

                            printf "\n*** running $SCRIPT for $OUTPUT ***\n"

                            ~/anaconda2/bin/python $SCRIPT --train $DATASET$TRAINING_SET.txt --test $DATASET$TEST_SET.txt --tag_scheme=$tag_scheme --epochs=5 --all_emb=1 --pre_emb=embeddings/$embedding.txt --cap_dim=$cap_dim --lower=$lower --char_lstm_dim=$char_lstm_dim --word_lstm_dim=$word_lstm_dim > $RESULTS_FOLDER$OUTPUT

	                    done

                    done

                done

            done

        done

    done

done
