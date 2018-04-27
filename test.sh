#!/usr/bin/env bash

SCRIPT=./train.py
DATASET=./dataset/
TRAINING_SET=categories_harem_I
TEST_SET=categories_mini
EMBEDDINGS_FOLDER=./embeddings/
RESULTS_FOLDER=./resultados/scripts/

declare -a tag_schemes=("iob")
declare -a embeddings=("wang2vec_s100")
declare -a cap_dim_values=("1")
declare -a lower_values=("1")
declare -a char_lstm_dim_values=("25")
declare -a word_lstm_dim_values=("100")
declare -a optimizer=("adam")



printf "\n*** running $optimizer ***\n"

