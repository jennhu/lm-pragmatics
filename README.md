# A fine-grained comparison of pragmatic language understanding in humans and language models

This repository contains code and data for "A fine-grained comparison of pragmatic language understanding in humans and language models" by Jennifer Hu, Sammy Floyd, Olessia Jouravlev, Evelina Fedorenko, and Edward Gibson.

## Prompts

All prompts can be found in the `prompts` folder. They have been generated from the original materials from Floyd, Jouravlev, et al. (in prep), which can be found under `fj_raw_data`.

## Running OpenAI models

The script `query_openai.py` is used to run OpenAI models (`text-ada-001`, `text-davinci-002`) on the prompts. Output CSV files are saved to the `model_data` directory.
To run a specific model, phenomenon (e.g., Irony), and seed, run
```
python query_openai.py -P <PHENOMENON> -M <MODEL> -S <SEED> --num_examples 0
```

You can also use the script `query_openai.sh` to run `text-ada-001` and `text-davinci-002` on all phenomena and seeds, with the following command:
```
bash query_openai.sh
```

**NOTE:** You will need an OpenAI account in order to access the models through the API.
The `query_openai.py` script expects the API key to be stored at a file called `key.txt` in the root directory;
this is not staged in the git repository for security reasons.

### OpenAI model timestamps

You can use the script `get_openai_timestamps.sh` to generate a CSV file with timestamps of OpenAI model runs.
This is done by checking the time at which the OpenAI-generated files were last modified.
An example output file can be found at `openai_timestamps.csv`, which contains timestamps from our paper.

## Running Huggingface models

The script `query_hf.py` is used to run open-source Huggingface models on the prompts.
As with the OpenAI models, the output CSV files are saved to the `model_data` directory.
To run a specific model, phenomenon (e.g., Irony), and seed, run
```
python query_hf.py -P <PHENOMENON> -M <MODEL> -S <SEED> --num_examples 0
```

Note that `query_hf.py`, by default, runs a single model on all phenomena and seeds 0-4. This is because the primary time bottleneck is the time needed to load the models.
You can omit the `-P` and `-S` arguments to run a model on all phenomena and seeds.

The script `query_hf.sh` illustrates how all the Huggingface models in our paper can be run on all tasks and seeds, using a for loop.
It is **not recommended** to run the script directly, as the loop will be excruciatingly slow.
Please run multiple jobs in parallel instead.

## Analyses

The notebook at `analysis/main.ipynb` contains code for analyzing the data and generating the figures from our paper.
All figures are rendered to `analysis/figures`.
