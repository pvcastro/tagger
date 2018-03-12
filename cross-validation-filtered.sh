#!/usr/bin/env bash

for i in {0..9}
do
	printf "\n***** fold "$i" *****\n"

	FOLDER=dataset/training/fold-$i

    ~/anaconda2/bin/python train.py --train $FOLDER/filtered_train.txt --dev $FOLDER/filtered_test.txt --test dataset/filtered_mini.txt --tag_scheme=iobes --epochs=5 --all_emb=1 --pre_emb=embeddings/glove_s100.txt
done
