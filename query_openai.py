# ~~~~~~~~~~~~~~~~~~~ Script for querying OpenAI API models

import numpy as np
import pandas as pd
import openai
import argparse
from tqdm import tqdm

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def get_completion(prompt, model="text-ada-001", answer_choices=["1","2","3","4"], **kwargs):
    # Get "distribution" over answer choices
    answer_logprobs = {}
    for i, answer in enumerate(answer_choices):
        completion = openai.Completion.create(
            model=model,
            prompt=prompt + answer,
            **kwargs
        ).choices[0]
        # Get logprob assigned to answer token (e.g., "1")
        # (final token is <|endoftext|>; penultimate is answer)
        logprob = completion.logprobs.token_logprobs[-2] 
        answer_logprobs[answer] = logprob
    answer_logprobs = [answer_logprobs[answer] for answer in answer_choices] # turn into sorted list
    probs = softmax(answer_logprobs) # normalize
    probs = {answer: probs[int(answer)-1] for answer in answer_choices} # turn back into dict (answers are 1-indexed)
    
    # Get generated answers
    kwargs.update({"echo": False}) # only care about final token
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        **kwargs
    ).choices[0]
    generated_answer = completion.text

    return generated_answer, probs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-3 completion on pre-constructed prompts.")
    parser.add_argument("--phenomenon", "-P", type=str)
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--seed", "-S", type=int, default=0)
    parser.add_argument("--model", "-M", type=str, default="text-ada-001")
    parser.add_argument("--suffix", default=None, choices=["no-story"])
    args = parser.parse_args()
    
    # Random seed.
    np.random.seed(args.seed)
    
    # Secret file with API key (DO NOT commit this)
    with open("key.txt", "r") as f:
        key = f.read()
    openai.api_key = key
    
    num_answer_choices = {
        "Humour": 5, 
        "Irony": 4, 
        "IndirectSpeech": 4,
        "Deceits": 4,
        "CoherenceInference": 2,
        "Maxims": 4,
        "Metaphor": 5
    }
    answer_choices = [
        str(answer) for answer in range(1, num_answer_choices[args.phenomenon]+1, 1)
    ]
    
    # These were obtained from the tokenizer tool https://beta.openai.com/tokenizer?view=bpe
    # answer_token_ids = {
    #     "1": 16,
    #     "2": 17,
    #     "3": 18,
    #     "4": 19,
    #     "5": 20
    # }
    avoid_token_ids = {
        "\n": 198
    }
    logit_bias = {
        str(token_id): -100 for token_id in avoid_token_ids.values()
    }

    # Parameters for all completions
    completion_params = dict(
        max_tokens=1,
        logprobs=0,
        temperature=0, # equivalent to taking argmax
        echo=True,
        logit_bias=logit_bias
    )
    
    if args.suffix is not None:
        suffix = "_"+args.suffix
    else:
        suffix = ""

    scenarios = pd.read_csv(
        f"prompts/{args.phenomenon}{suffix}_prompts_seed{args.seed}_examples{args.num_examples}.csv"
    ).dropna()
    print(scenarios.head())
    
    for i, row in tqdm(scenarios.iterrows()):
        prompt = row.prompt
        generated_answer, probs = get_completion(
            prompt, 
            model=args.model, 
            answer_choices=answer_choices,
            **completion_params
        )
        # Evaluate generated text.
        scenarios.loc[i, "generation"] = generated_answer.strip()
        scenarios.loc[i, "generation_isvalid"] = (generated_answer.strip() in answer_choices)
        # Record probability distribution over valid answers.
        scenarios.loc[i, "distribution"] = str(probs)
        scenarios.loc[i, "prob_true_answer"] = probs[str(row.randomized_true_answer)]
        # Take model "answer" to be argmax of the distribution.
        sorted_probs = [probs[answer] for answer in answer_choices]
        chosen_answer = str(np.argmax(sorted_probs) + 1)
        scenarios.loc[i, "answer"] = chosen_answer
        scenarios.loc[i, "correct"] = (chosen_answer == str(row.randomized_true_answer))
        scenarios.loc[i, "answer_label_complex"] = eval(row.randomized_labels_complex)[int(chosen_answer)-1]
     
    scenarios["model"] = args.model
    scenarios.to_csv(f"model_data/{args.phenomenon}{suffix}_{args.model}_seed{args.seed}_examples{args.num_examples}.csv", index=False)
