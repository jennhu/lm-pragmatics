# ~~~~~~~~~~~~~~~~~~~ Script for querying Huggingface models

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration


def load_mt(model_name="facebook/opt-30b", cache_dir="/om2/user/jennhu/cache"):
    if "flan-t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Successfully loaded tokenizer ({model_name})")
    elif "tk-instruct" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Successfully loaded tokenizer ({model_name})")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).to(DEVICE)
        print(f"Successfully loaded model ({model_name})")
        # the fast tokenizer currently does not work correctly for OPT models
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
        
    return model, tokenizer

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def get_completion(prompt, model, tokenizer, answer_choices=[1,2,3,4,5], **kwargs):
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    answer_token_ids = tokenizer(
        [str(answer) for answer in answer_choices], 
        return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze().tolist()

    outputs = model.generate(
        **input_ids, 
        max_new_tokens=1,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
    )
    if isinstance(outputs.scores, tuple):
        logits = outputs.scores[0][0]
    else:
        logits = outputs.scores
        
    answer_logits = [logits[answer_id].item() for answer_id in answer_token_ids]
    generated_answer = str(answer_choices[np.argmax(answer_logits)])
    probs = softmax(answer_logits)
    probs = {answer: probs[int(answer)-1] for answer in answer_choices}
    
    return generated_answer, probs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-3 completion on pre-constructed prompts.")
    parser.add_argument("--phenomenon", "-P", nargs="+", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--seed", "-S", type=int, default=None)
    parser.add_argument("--model", "-M", type=str, default="text-ada-001")
    parser.add_argument("--control", default=False, action="store_true")
    parser.add_argument("--suffix", default=None, choices=["no-story"])
    args = parser.parse_args()
    
    # Constant
    num_answer_choices = {
        "Humour": 5, 
        "Irony": 4, 
        "IndirectSpeech": 4,
        "Deceits": 4,
        "CoherenceInference": 2,
        "Maxims": 4,
        "Metaphor": 5
    }
    
    # Random seed.
    if args.seed is None:
        seeds = range(5)
    else:
        seeds = [args.seed]
    print(f"Evaluating on seeds: {seeds}")
    
    # Phenomena.
    if args.phenomenon is None:
        eval_phenomena = list(num_answer_choices.keys())
    else:
        eval_phenomena = args.phenomenon
    print(f"Evaluating on phenomena: {eval_phenomena}")
    
    # Load model.
    print(f"Loading model and tokenizer ({args.model})")
    model, tokenizer = load_mt(model_name=args.model)
        
    # Prepare some variables.
    if args.control:
        suffix = "_Control"
    elif args.suffix is not None:
        suffix = "_"+args.suffix
    else:
        suffix = ""
    safe_model_name = args.model.split("/")[-1]
    
    
    # Iterate through seeds and phenomena. We do this so we can reduce the number of times 
    # we need to load the model, which is the main time bottleneck.
    for seed in seeds:
        for phenomenon in eval_phenomena:
            answer_choices = [
                str(answer) for answer in range(1, num_answer_choices[phenomenon]+1, 1)
            ]
            
            np.random.seed(seed)
            scenarios = pd.read_csv(f"prompts/{phenomenon}{suffix}_prompts_seed{seed}_examples{args.num_examples}.csv").dropna()
            print(scenarios.head())

            for i, row in tqdm(scenarios.iterrows()):
                prompt = row.prompt
                generated_answer, probs = get_completion(
                    prompt, model, tokenizer, answer_choices=answer_choices
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

            scenarios["model"] = safe_model_name
            scenarios.to_csv(f"model_data/{phenomenon}{suffix}_{safe_model_name}_seed{seed}_examples{args.num_examples}.csv", index=False)
