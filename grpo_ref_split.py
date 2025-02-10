from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/data2/Qwen/Qwen2.5-7B-Instruct"
#model_path = "step_200"
beta = 0.04
num_pre_Q = 8
Q_batch_size = 1
all_steps = 1000

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model

from datasets import load_dataset
dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question_zh'], dataset['answer'])]

from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q//2,
            pad_token_id=tokenizer.pad_token_id,
        )

system_prompt = "You are a helpful assistant, first use <think></think> to think, then answer with <answer></answer>"
def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        x += "请先用<think></think>包含思考过程，再以<answer></answer>输出带过程的回答。"
        tip_text.append(tokenizer.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers

def reward_correct(item, answer):
    nums = re.findall(r'\d+', answer)
    if len(nums) == 0: return -1.0
    lastnum = nums[-1]
    return 1.0 if item["A"] == lastnum else -0.5
def reward_format(item, answer):
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
    return 2 if re.match(pattern, answer, re.DOTALL) else 0

def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers

def generate_mode(num=10, rank=0):
    if rank == 0: print('enter generate mode')
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        if rank == 0: print('rewards:', rewards)
        if rank == 0 and ii == 5: print('answers:', answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)])
        requests.post(f"{ref_server}/upload", data=xdata)
    if rank == 0: print('exit generate mode')

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

import deepspeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)

    def get_per_token_logps(logits, input_ids):
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss

generate_mode(rank=torch.distributed.get_rank())

from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)
for step in progress:
    batch = get_batch()
    if batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()
    loss = GRPO_step(batch)

    engine.backward(loss)
    engine.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % 200 == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            save_name = f"/data2/hanjin1/ckp/step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()