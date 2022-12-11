#!/bin/bash

# NOTE: it's not recommended to run this as a for-loop. Run in parallel instead.

# Main analyses
for model in gpt2 google/flan-t5-base google/flan-t5-xl allenai/tk-instruct-3b-def-pos allenai/tk-instruct-11b-def-pos; do
    python query_hf.py -P $phen -M $model -S $seed
done

# No-story analyses
for model in google/flan-t5-xl allenai/tk-instruct-11b-def-pos; do
    python query_hf.py -M $model --suffix no-story -P Deceits Irony IndirectSpeech Maxims Metaphor
done