from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from vllm import LLM, SamplingParams
import wandb
import logging
from datetime import datetime

SEED = 1029
temp_random = random.Random(SEED)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

log_path = f'./log/{now.strftime("%Y_%m_%d")}'
if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(level=logging.INFO,
                    filename=f'{log_path}/log_{dt_string}.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from ref_server import MODEL_PATH
gen_device_index = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 1000
batch_size = 8
num_pre_Q = 8
train_batch_size = 4
gen_update_steps = 16
save_steps = 100
compute_gen_logps = True
use_confidence = False
clip_param = 0.2
validate_step = 50
validate_data_num = 300
val_batch_size = 64
max_new_tokens = 200
output_path = 'confidence_v1' if use_confidence else 'no_confidence_v1'

from ref_server import PORT
ref_server = f"http://localhost:{PORT}"

SYSTEM_PROMPT = """You are a helpful assistant. 
A conversation between User and Assistant. 
The user asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

PROMPT = 'Question:{}\nPlease start with <think>.'

from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

wandb_key = "346b4322aa0fd6c572c8d1e1c394818bf8e256da"
os.environ["WANDB_API_KEY"] = wandb_key
wandb_project_name = f"simpple_GRPO"
os.environ["WANDB_PROJECT"] = wandb_project_name
wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
logging.info(f"Weights & Biases initialized. wandb_project_name:{wandb_project_name}")


ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
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


def gen_samples_val(inputs, model, tokenizer, num_pre_Q=1):
    prompts = [x["Q"] for x in inputs]

    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    voutputs = model.generate(
        **tip_inputs,
        temperature=0.9, 
        max_new_tokens=max_new_tokens,
        )
    answers = []
    for v in voutputs:
        answers.append(tokenizer.decode(v, skip_special_tokens=True))

    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a))

    return rewards


def validate(val_data, step, model, batch_size=64):
    all_data_count = len(val_data)

    progress_bar = tqdm(range(0, all_data_count, batch_size))
    correct_count = 0
    all_count = 0
    for i in progress_bar:
        inputs = val_data[i:i + batch_size]
        rewards = gen_samples_val(inputs, model, tokenizer)
        rewards_new = [1 if i == 1 else 0 for i in rewards]
        all_count += len(inputs)
        correct_count += sum(rewards_new)
        progress_bar.set_postfix_str(f"acc:{100*(correct_count / all_count):.2f}%")
    
    return  correct_count / all_count

    

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
    data['gen_logps'] = bytes_to_tensor(dd[4])
    if use_confidence:
        data['confidence'] = bytes_to_tensor(dd[5])

    # for k,v in data.items():
    #     if k in ['rewards', 'confidence']:
    #         print(k, v)

    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if use_confidence:
        confidence = batch['confidence'].to(engine.device).unsqueeze(1)
        advantages *= confidence

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig

def get_dataset(data_path, split='train'):
    assert split in ['train', 'val']

    dataset = load_dataset('json', data_files={"train":data_path})['train']
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

    temp_random.shuffle(QAs)
    if split == 'train':
        return QAs[validate_data_num:]
    else:
        return QAs[:validate_data_num]


def gen_answers(prompts, vllm_gen, sampling_params, tokenizer):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT.format(x)}], tokenize=False, add_generation_prompt=True))

    voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
    answers = [];  ans_token_ids = []; gen_log_probs = []
    for v in voutputs:
        tmp_bsz = []
        for z in v.outputs: 
            answers.append(z.text)
            ans_token_ids.append(z.token_ids)
            tmp = []
            for i in z.logprobs:
                tmp.append([j.logprob for j in list(i.values())[:2]])
            tmp_bsz.append(tmp)
        gen_log_probs.append(tmp_bsz) 

    return answers, ans_token_ids, gen_log_probs


def gen_samples(inputs, vllm_gen, sampling_params, tokenizer):
    prompts = [x["Q"] for x in inputs]
    answers, ans_token_ids, gen_log_probs = gen_answers(prompts, vllm_gen, sampling_params, tokenizer)
    rewards = []
    correct_rewards = []
    format_rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
            correct_rewards.append(reward_correct(inp, a))
            format_rewards.append(reward_format(inp, a))
    prompts_text = [tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x+'\n'}], tokenize=False, add_generation_prompt=True) for x in prompts]
    return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids, torch.tensor(correct_rewards, dtype=torch.float32), torch.tensor(format_rewards, dtype=torch.float32), gen_log_probs


def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

# def reward_correct(item, answer):
#     ans, _ = get_answer_and_span(answer)
#     if ans is None: return -1.0
#     ans = parse(ans, extraction_config=[ExprExtractionConfig()])
#     ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
#     return 1 if verify(ans, ground_truth) else -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count_1 = answer.count("<think>")
    think_count_2 = answer.count("</think>")
    answer_count_1 = answer.count("<answer>")
    answer_count_2 = answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count_1==1 and think_count_2 == 1 and answer_count_1 == 1 and answer_count_2==1 else -1

def decode_with_offsets(generation_ids, tokenizer):
    # I'm not aware of any more convenient way to do this. If you know, please do let me know
    tokens = tokenizer.convert_ids_to_tokens(generation_ids)

    text = ''
    offsets = []
    for i in range(len(generation_ids)):
        if tokens[i] == tokenizer.eos_token:
            break
        text = tokenizer.convert_tokens_to_string(tokens[:i + 1])
        offsets.append(len(text))
    offsets += [-1 for _ in range(len(tokens) - len(offsets))]  # add invalid offsets for EOS

    return text, offsets

def match_answer_span(answer_span, offsets):
    answer_s, answer_e = answer_span
    inds = []
    for i, offset in enumerate(offsets):
        if answer_s < offset:
            inds.append(i)
            if answer_e <= offset:
                break
    return inds

def get_answer_score(probs):
    if not isinstance(probs, list):
        probs = probs.topk(k=2, dim=-1, sorted=True).values
    score = (probs[:, 0] - probs[:, 1]).mean()
    return float(score)

def get_answer_and_span(text):
    answer_content_match = re.search(r'(?<=<answer>).*?(?=</answer>)', text)
    if answer_content_match:
        answer_content = answer_content_match.group(0)  # 获取匹配的内容
        numbers = list(re.finditer(r'\d+\.\d+|\d+/\d+|\d+', answer_content))

        if numbers:
            # 获取最后一个数字
            last_number_match = numbers[-1]
            last_number = last_number_match.group(0)  # 数字的值
            start_pos = answer_content_match.start() + last_number_match.start()  # 数字的起始位置
            end_pos = answer_content_match.start() + last_number_match.end()  # 数字的结束位置

            return last_number, (start_pos, end_pos)
        else:
            return None, None
    else:
        return None, None

def get_confidence(tokenizer, num_generations, gen_ids, gen_probs):
    confidences = []
    # Sample candidates
    for j in range(num_generations):
        curr_gen_probs = torch.tensor(gen_probs[j])
        text, offsets = decode_with_offsets(gen_ids[j], tokenizer)
        answer, answer_span = get_answer_and_span(text)
        if answer_span is None:
            confidences.append(0)
        else:
            answer_tokens = match_answer_span(answer_span, offsets)
            if len(answer_tokens) == 2:
                answer_tokens = answer_tokens[:1]

            answer_probs = curr_gen_probs[answer_tokens] 
            cot_score = get_answer_score(answer_probs)
            confidences.append(cot_score)

    return torch.tensor(confidences, dtype=torch.float32) + 1e-5


def gen_worker(Q, data_path, physics_device, tokenizer):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    
    vllm_gen = LLM(model=MODEL_PATH, gpu_memory_utilization=0.5)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=max_new_tokens, logprobs=2)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
    
    QAs = get_dataset(data_path)
    
    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 2 == 0: try_update_model()
        inputs = random.sample(QAs, batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids, correct_rewards, format_rewards, gen_log_probs = gen_samples(inputs, vllm_gen, sampling_params, tokenizer)
        # print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        # if it % 5 == 0: print('answers:', answers[0])

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_correct_rewards = correct_rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_format_rewards = format_rewards[i*num_pre_Q:(i+1)*num_pre_Q]

            curr_gen_log_probs = gen_log_probs[i]

            wandb.log({
                    "correct_rewards": curr_correct_rewards.mean().item(),
                    "format_rewards": curr_format_rewards.mean().item()
                })

            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if use_confidence:
                answer_confidence = get_confidence(tokenizer, num_pre_Q, curr_ans_ids, curr_gen_log_probs)

            if ref_server_ver == 'tensor':
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    if use_confidence:
                        sub_answer_confidence = answer_confidence[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)

                    merged_text = tokenizer.decode(merged_ids, skip_special_tokens=True)

                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                    if compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))
                    
                    if use_confidence:
                        data.append(tensor_to_bytes(sub_answer_confidence))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_name = MODEL_PATH.split('/')[-1]
    data_path = "/home/wxy/project/reasoning/cot_decoding/gsm8k_data/train.jsonl"

    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, data_path, gen_device_index, tokenizer))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    val_data = get_dataset(data_path, 'val')
    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss = GRPO_step(batch)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            # progress.set_description(f"Loss: {loss.item():.6f}")
            wandb.log({
                    "loss": loss.item()
                })
            
        if step == 1 or step % validate_step == 0:
            val_acc = validate(val_data, step, engine.module, val_batch_size)
            wandb.log({"val_acc":val_acc})

            print(f"step:{step}, batch_size:{batch_size}, val acc:{100*val_acc:.2f}%")
            with open('./val_result.txt', 'a') as f:
                f.write(f"step:{step}, batch_size:{batch_size}, val acc:{100*val_acc:.2f}%\n")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"/mnt/local/wxy/models/simple_grpo/{model_name}/{output_path}/step_{step}"
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()
