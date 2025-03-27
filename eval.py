from grpo_vllm_one import SYSTEM_PROMPT
import datasets  
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import re
from math_verify import parse, verify, ExprExtractionConfig

def build_prompt(messages):
    return "\n".join([msg["content"].strip() for msg in messages])

def extract_single_number_v2(text):
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")
    matches = list(re.finditer(MODEL_ANS_RE, text))
    if len(matches) > 0:
        match = matches[-1]
        return match.group()
    else:
        return None

def extract_answer_from_model_output(text):
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def prepare_dataset(data_name='gsm8k', split="train"):
    #    data = load_dataset('openai/gsm8k', 'main')[split]
    data = load_dataset('json', data_files={split:f'../cot_decoding/data/{data_name}/{split}.jsonl'})[split]
    formatted_data = []
    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)
    return formatted_data

def evaluate_model(model, tokenizer, eval_examples, batch_size, max_new_tokens, verbose=False):
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating", unit="batch")
    for batch_start in progress_bar:
        batch_end = min(batch_start + batch_size, total)
        batch_examples = eval_examples[batch_start:batch_end]

        # Prepare batch inputs
        batch_prompts = [example["prompt"] for example in batch_examples]
        batch_expected = [example["answer"] for example in batch_examples]

        # Tokenize and generate responses
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )

        # Decode responses and extract answers
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_predicted = [extract_answer_from_model_output(response) for response in responses]

        # Check correctness for each example in the batch
        for i, (predicted, expected) in enumerate(zip(batch_predicted, batch_expected)):
            is_correct = reward_correct(expected, predicted)
            # Update counter for correct answers
            if is_correct:
                correct += 1
        # Update progress bar with current accuracy
        progress_bar.set_postfix({"acc": f"{(correct / batch_end) * 100:.2f}%"})

    # Calculate and print final accuracy
    accuracy = (correct / total) * 100

    return accuracy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/mnt/local/wxy/models/simple_grpo/og/Qwen2.5-3B/step_100')
    parser.add_argument('--data_path', type=str, default='/home/wxy/project/reasoning/cot_decoding/gsm8k_data/test.jsonl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=1024)

    args = parser.parse_args()
    return args

def get_dataloader(data_path, batch_size):
    raw_data = datasets.load_dataset(path = 'json', data_files={'train':data_path})['train']
    dataloader = DataLoader(raw_data, batch_size=batch_size, shuffle=False)

    return dataloader

def extract_answer_from_dataset(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def reward_correct(gt, answer):
    # print('gt:', gt)
    # print('answer:', answer)
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    try:
        nums = re.findall(pattern, answer) 
        if len(nums) == 0: return -1.0
        lastnum = nums[-1]
        # gt = extract_answer_from_dataset(gt)
        # print('answer: ', lastnum)
        # print('gt: ', gt)
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(gt, extraction_config=[ExprExtractionConfig()])
        return 1 if verify(ans, ground_truth) else 0
    except:
        print('gt:', gt)
        print('answer:', answer)
        return 0




if __name__ == '__main__':
    

    args = get_args()
    data_path = args.data_path
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    model_path = args.model_path

    model_name = model_path.split('/')[-1]
    model = AutoModelForCausalLM.from_pretrained(model_path,
                torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

    test_data = prepare_dataset(split="test")
    after_grpo_accuracy = evaluate_model(model, tokenizer, test_data, max_new_tokens=max_new_tokens, batch_size=batch_size)

    with open('./result_og.txt', 'a') as f:
        f.write(f'{model_path}\t{after_grpo_accuracy}\n')