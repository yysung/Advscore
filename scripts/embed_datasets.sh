#!/bin/bash
source .env

datasets=("advqa" "bamboogle" "trickme" "fm2")

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    python data_prep/create_question_reps.py \
        --dataset $dataset \
        -m "cohere-embed-english-v3.0" \
        -t "classification"
done
