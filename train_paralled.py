import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from typing import Union, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from safetensors import safe_open
import gc
import random
from typing import List
from matplotlib import pyplot as plt
from safetensors.torch import save_file
from copy import deepcopy
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT, MCD_SYSTEM_PROMPT
from utils import PCA_DIM


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

BATCH_SIZE = 10
NUM_EPOCHES = 5



def embed_soft_prompt(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_messages: List[List[Dict[str, str]]],
    soft_prompt: torch.Tensor
):
    if soft_prompt.device != model.device:
        raise ValueError("soft_prompt must be on the same device as model")
    if soft_prompt.dtype != model.dtype:
        raise ValueError("soft_prompt must be of the same dtype as model")

    if soft_prompt.dim() != 2:
        raise ValueError("soft_prompt must be a 2D tensor")
    if any(len(messages) != 1 for messages in all_messages):
        raise ValueError("all_messages must be a list of single-message lists")
    n_prompt_tokens = soft_prompt.size(0)

    # As system message appears first, we replace the first n_prompt_tokens eos tokens with soft_prompt
    messages_with_eos_placeholder = [[{'role': 'system', 'content': toker.eos_token * n_prompt_tokens}] + e for e in all_messages]
    input_ids = [toker.apply_chat_template(e, add_generation_prompt=True, tokenize=True) for e in messages_with_eos_placeholder]
    input_lengths = [len(e) for e in input_ids]
    max_input_length = max(input_lengths)
    input_ids = [e + [toker.eos_token_id] * (max_input_length - len(e)) for e in input_ids]

    placeholder_start_index = input_ids[0].index(toker.eos_token_id) # all input_ids have the same placeholder_start_index
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds[:, placeholder_start_index:placeholder_start_index+n_prompt_tokens] = soft_prompt.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
    return inputs_embeds, input_lengths


def get_shuffled_messages_and_labels(all_messages: List[List[Dict[str, str]]], labels: torch.Tensor, seed=42):
    rng = random.Random(seed)
    assert len(all_messages) == len(labels)
    if len(all_messages) % BATCH_SIZE != 0:
        raise ValueError(f"len(all_messages) must be a multiple of {BATCH_SIZE}")
    indices = list(range(len(all_messages)))
    for epoch_idx in range(NUM_EPOCHES):
        rng.shuffle(indices)
        for idx in range(len(all_messages)//BATCH_SIZE):
            yield epoch_idx, [all_messages[indices[idx*BATCH_SIZE + i]] for i in range(BATCH_SIZE)], labels[indices[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]]


def main():
    patch_open()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'], required=True)
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short', 'mcd'], required=True)
    parser.add_argument("--output_path", type=str, default='./trained_prompts')
    parser.add_argument("--ablate_norm", action='store_true')
    parser.add_argument("--ablate_refu", action='store_true')
    parser.add_argument("--ablate_harm", action='store_true')
    parser.add_argument("--original_traindataset", action="store_true")
    parser.add_argument("--use_multilingual", action="store_true")
    parser.add_argument("--use_paralled_multilingual", action="store_true")
    
    
    args = parser.parse_args()

    if sum([args.ablate_norm, args.ablate_refu, args.ablate_harm]) >= 2:
        raise ValueError("Only one of --ablate_norm, --ablate_refu, --ablate_harm can be set to True")

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # prepare model
    model_name = args.pretrained_model_path.split('/')[-1]

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else None,
    )
    device = model.device
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    os.makedirs(f'{args.output_path}/{model_name}', exist_ok=True)

    # prepare LinearTransform
    refusal_model = nn.Linear(PCA_DIM, 1)
    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/refusal.safetensors', framework='pt') as f:
        weight = f.get_tensor('weight').mean(dim=0)
        bias = f.get_tensor('bias').mean(dim=0)
    refusal_model.load_state_dict({'weight': weight, 'bias': bias})
    refusal_model.float().to(device)
    for param in refusal_model.parameters():
        param.requires_grad = False

    harmfulness_model = nn.Linear(PCA_DIM, 1)
    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/harmfulness.safetensors', framework='pt') as f:
        weight = f.get_tensor('weight').mean(dim=0)
        bias = f.get_tensor('bias').mean(dim=0)
    harmfulness_model.load_state_dict({'weight': weight, 'bias': bias})
    harmfulness_model.float().to(device)
    for param in refusal_model.parameters():
        param.requires_grad = False

    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/transform.safetensors', framework='pt') as f:
        mean = f.get_tensor('mean').float().to(device)
        V = f.get_tensor('V').float().to(device)

    # prepare toker
    toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast='Orca-2-' not in model_name)

    if 'Llama-2-' in model_name and '-chat' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'Llama-3' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/llama-3-instruct.json'
    elif 'CodeLlama-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'Orca-2-' in model_name:
        generation_config_file = './generation_configs/orca-2.json'
    elif 'Mistral-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/mistral-instruct.json'
    elif 'vicuna-' in model_name:
        generation_config_file = './generation_configs/vicuna.json'
    elif 'openchat-' in model_name:
        generation_config_file = './generation_configs/openchat.json'
    elif 'qwen' or 'Qwen' in model_name:
        generation_config_file = './generation_configs/qwen.json'
    else:
        raise ValueError(f"Unsupported or untuned model: {model_name}")
    generation_config = json.load(open(generation_config_file))
    chat_template_file = generation_config['chat_template']
    chat_template = open(chat_template_file).read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    toker.chat_template = chat_template

    # prepare soft prompt
    if args.prompt_length == 'default':
        init_ids = toker(DEFAULT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'short':
        init_ids = toker(SHORT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'mistral':
        init_ids = toker(MISTRAL_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'mcd':
        init_ids = toker(MCD_SYSTEM_PROMPT).input_ids[1:]
    init_embeds = model.get_input_embeddings().weight.data[init_ids].detach()
    soft_prompt = nn.Parameter(init_embeds, requires_grad=True).to(model.device)

    logging.info(f"Other modules loaded")
    logging_cuda_memory_usage()

    lang_set = ['danish', 'korean', 'greek', 'irish']

    if args.original_traindataset:
        # prepare data
        dataset = 'custom'
        with open(f"./data/{dataset}.txt") as f:
            lines = f.readlines()
        with open(f"./data_harmless/{dataset}.txt") as f:
            lines_harmless = f.readlines()
    elif args.use_paralled_multilingual:
        dataset = 'custom'
        
        lines_m = []
        lines_harmless_m = []

        with open(f"./data/{dataset}.txt") as f:
            lines = f.readlines()
        with open(f"./data_harmless/{dataset}.txt") as f:
            lines_harmless = f.readlines()
        

        for lang_s in lang_set:
            with open(f"./data/{dataset}{lang_s}.txt") as f:
                lines_m.append(f.readlines())
            with open(f"./data_harmless/{dataset}{lang_s}.txt") as f:
                lines_harmless_m.append(f.readlines())

    elif args.use_multilingual:
        dataset = "multilingual"
        with open(f"./data/{dataset}.txt") as f:
            lines = f.readlines()
        with open(f"./data_harmless/{dataset}.txt") as f:
            lines_harmless = f.readlines()
    else:
        # prepare data
        dataset = 'custom'
        with open(f"./data/{dataset}.txt") as f:
            lines = f.readlines()
        with open(f"./data_harmless/{dataset}.txt") as f:
            lines_harmless = f.readlines()
    

    all_queries = [e.strip() for e in lines if e.strip()]

    n_queries = len(all_queries)

    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]

    n_queries_harmless = len(all_queries_harmless)

    all_messages = [[{'role': 'user', 'content': e.strip()}] for e in all_queries] + [[{'role': 'user', 'content': e.strip()}] for e in all_queries_harmless]
    labels = [0 for _ in range(n_queries)] + [1 for _ in range(n_queries_harmless)]
    labels = torch.tensor(labels, dtype=torch.float).to(device)

    base_hidden_states = {}
    base_transformeds = {}
    base_refusal_logits = {}
    base_harmfulness_logits = {}
    m2english = {}
    e2e = {}
    for messages in all_messages:
        query = messages[0]['content']
        e2e[query] = query
        if args.prompt_length == 'default':
            messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
        elif args.prompt_length == 'short':
            messages = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
        elif args.prompt_length == 'mistral':
            messages = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
        elif args.prompt_length == 'mcd':
            messages = [{'role': 'system', 'content': MCD_SYSTEM_PROMPT}] + messages
        
        input_ids = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)

        last_hidden_state = model(input_ids, output_hidden_states=True).hidden_states[-1][:, -1]
        transformed = torch.matmul(last_hidden_state.float() - mean, V)
        refusal_logits = refusal_model(transformed[:, :PCA_DIM]).squeeze(-1)
        harmfulness_logits = harmfulness_model(transformed[:, :PCA_DIM]).squeeze(-1)

        base_hidden_states[query] = last_hidden_state.detach()
        base_transformeds[query] = transformed.detach()
        base_refusal_logits[query] = refusal_logits.detach()
        base_harmfulness_logits[query] = harmfulness_logits.detach()
    m2english["english"] = e2e
    base_mean_multilingual_transformed = {}
    
    all_language_messages = [all_messages]
    if args.use_paralled_multilingual:
        m_queries = [ [x.strip() for x in e if x.strip()] for e in lines_m]
        # print(len(m_queries))
        
        m_queries_harmless = [ [x.strip() for x in e if x.strip()] for e in lines_harmless_m]
        # print(len(m_queries_harmless))
        
        for i in range(len(lang_set)):
            m_messages = [[{'role': 'user', 'content': e.strip()}] for e in m_queries[i]] + [[{'role': 'user', 'content': e.strip()}] for e in m_queries_harmless[i]]
            # print(f"{len(m_messages)} - {len(all_messages)}")
            all_language_messages.append(m_messages)
            m2e = {}
            for m_id in range(len(m_messages)):
                query = all_messages[m_id][0]['content']

                if query not in base_mean_multilingual_transformed.keys():
                    base_mean_multilingual_transformed[query] = []
                
                m2e[m_messages[m_id][0]['content']] = query

                messages = m_messages[m_id]
                if args.prompt_length == 'default':
                    messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
                elif args.prompt_length == 'short':
                    messages = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
                elif args.prompt_length == 'mistral':
                    messages = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
                elif args.prompt_length == 'mcd':
                    messages = [{'role': 'system', 'content': MCD_SYSTEM_PROMPT}] + messages
                
                input_ids = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)

                last_hidden_state = model(input_ids, output_hidden_states=True).hidden_states[-1][:, -1]
                transformed = torch.matmul(last_hidden_state.float() - mean, V)
                base_mean_multilingual_transformed[query].append(transformed)
            m2english[lang_set[i]] = m2e
    

    step = 0
    optimizer = torch.optim.AdamW([soft_prompt], lr=1e-3)
    seed = 42
    for lang_id in range(len(lang_set)+1):
        current_all_messages = all_language_messages[lang_id]
        
        if lang_id == 0:
            current_lang = "english"
            logging.info(f"Now we are training the language [English]...")
            current_base_transformeds = base_transformeds
            current_base_multilingual_transformed = {key:torch.stack(base_mean_multilingual_transformed[key]).mean(dim=0) for key in base_mean_multilingual_transformed.keys()}
        else:
            current_lang = lang_set[lang_id-1]
            logging.info(f"Now we are training the language [{lang_set[lang_id-1]}]...")
            current_base_transformeds = {key:base_mean_multilingual_transformed[key][lang_id-1] for key in base_mean_multilingual_transformed.keys()}
            current_base_multilingual_transformed = {key:torch.stack(base_mean_multilingual_transformed[key][:lang_id-1] + [base_transformeds[key]] + base_mean_multilingual_transformed[key][lang_id:]).mean(dim=0) for key in base_mean_multilingual_transformed.keys()}


        for epoch_idx, batch_messages, batch_labels in get_shuffled_messages_and_labels(current_all_messages, labels, seed=seed):
            batch_queries = [m2english[current_lang][e[0]['content']] for e in batch_messages]
            
            batch_base_hidden_states = torch.concat([base_hidden_states[e] for e in batch_queries], dim=0)
            batch_base_refusal_logits = torch.concat([base_refusal_logits[e] for e in batch_queries], dim=0)
            batch_base_harmfulness_logits = torch.concat([base_harmfulness_logits[e] for e in batch_queries], dim=0)
            
            
            optimizer.zero_grad()

            inputs_embeds, new_input_lengths = embed_soft_prompt(model, toker, batch_messages, soft_prompt)
            new_hidden_states = model(inputs_embeds=inputs_embeds, output_hidden_states=True).hidden_states[-1]
            new_last_hidden_states = new_hidden_states[range(len(new_input_lengths)), np.array(new_input_lengths, dtype=int)-1]

            base_transformed = torch.concat([current_base_transformeds[e] for e in batch_queries], dim=0)
            new_transformed = torch.matmul(new_last_hidden_states.float() - mean, V)

            if args.use_paralled_multilingual:
                base_multilingual_transformed = torch.concat([current_base_multilingual_transformed[e] for e in batch_queries], dim=0)
                multilingual_loss = torch.mean((base_multilingual_transformed - new_transformed)**2)

            norm_loss = torch.mean((new_transformed[:, PCA_DIM:] - base_transformed[:, PCA_DIM:])**2)
            #norm_loss = torch.mean(torch.mean(new_transformed[:, PCA_DIM:] - base_transformed[:, PCA_DIM:], dim=0)**2)
            refusal_logits = refusal_model(new_transformed[:, :PCA_DIM]).squeeze(-1) - batch_base_refusal_logits
            refusal_loss = F.binary_cross_entropy_with_logits(refusal_logits, batch_labels)
            harmfulness_logits = harmfulness_model(new_transformed[:, :PCA_DIM]).squeeze(-1) - batch_base_harmfulness_logits
            harmfulness_loss = F.binary_cross_entropy_with_logits(harmfulness_logits, batch_labels)

            if args.ablate_refu:
                total_loss = harmfulness_loss + norm_loss * 1e-3
            elif args.ablate_harm:
                total_loss = refusal_loss + norm_loss * 1e-3
            elif args.ablate_norm:
                total_loss = refusal_loss + harmfulness_loss * 1e-2
            elif args.use_paralled_multilingual:
                total_loss = refusal_loss + harmfulness_loss * 1e-2 + norm_loss * 1e-3 + multilingual_loss * 1e-3
            else:
                total_loss = refusal_loss + harmfulness_loss * 1e-2 + norm_loss * 1e-3

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(soft_prompt, 1.0)
            optimizer.step()
            step += 1

            if step % 10 == 0:
                if args.use_paralled_multilingual:
                    logging.info(f'Step {step}, total_loss {total_loss}, multilingual_loss {multilingual_loss.cpu().item()}, refusal_loss {refusal_loss.cpu().item()}, harmfulness_loss {harmfulness_loss.cpu().item()}, norm_loss {norm_loss.cpu().item()}')
                else:
                    logging.info(f'Step {step}, total_loss {total_loss}, refusal_loss {refusal_loss.cpu().item()}, harmfulness_loss {harmfulness_loss.cpu().item()}, norm_loss {norm_loss.cpu().item()}')
    soft_prompt = soft_prompt.detach()
    if args.original_traindataset:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_original_traindataset.safetensors')
    elif args.ablate_norm:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_nonorm.safetensors')
    elif args.ablate_refu:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_norefu.safetensors')
    elif args.ablate_harm:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_noharm.safetensors')
    elif args.use_paralled_multilingual:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_use_paralled_multilingual_center.safetensors')
    elif args.use_multilingual:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_use_multilingual.safetensors')
    else:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors')

    logging.info(f"Training finished")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
