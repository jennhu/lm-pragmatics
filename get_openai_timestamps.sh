#!/bin/bash

# Create CSV file and add header (column names)
outfile="openai_timestamps.csv"
touch $outfile
echo "model,phenomenon,timestamp" >> $outfile

# Populate with timestamp info (just considering seed 0)
for model in text-ada-001 text-davinci-002; do
    for phen in CoherenceInference Deceits IndirectSpeech Irony Humour Maxims Metaphor; do
        timestamp=$(stat -c '%y' model_data/${phen}_${model}_seed0_examples0.csv)
        echo "${model},${phen},${timestamp}" >> $outfile
    done
done