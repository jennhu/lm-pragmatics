#!/bin/bash

# Main analyses
for seed in {0..4}; do
    for phen in Humour Irony IndirectSpeech Deceits CoherenceInference Maxims Metaphor; do  
        for model in text-ada-001 text-davinci-002; do 
            python query_openai.py -P $phen -M $model -S $seed --num_examples 0
        done
    done
done

# No-story analyses
model="text-davinci-002"
for seed in {0..4}; do
    for phen in Deceits Irony IndirectSpeech Maxims Metaphor; do  
        python query_openai.py -P $phen -M $model -S $seed --num_examples 0 --suffix no-story
    done
done
