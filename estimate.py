import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from utils import PCA_DIM
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from safetensors.torch import save_file
from sklearn.decomposition import PCA
from copy import deepcopy
from utils import gram_schmidt


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


  
def kmeans_smoothing(coordinates, values, k=5):  

    max_distance = torch.max(torch.norm(coordinates.unsqueeze(0) - coordinates.unsqueeze(1), dim=-1))  

    smoothed_values = []  
  
    for i in range(len(coordinates)):  

        if k == 1:  
            smoothed_value = values[i]  
        else:  
            current_coord = coordinates[i]  
    
            distances = torch.norm(coordinates - current_coord, dim=1)  
 
            _, indices = torch.topk(distances, k=k, largest=False)  
 
            weights = torch.exp(-distances[indices] / max_distance / 0.2)  
  
            smoothed_value = torch.sum(values[indices] * weights) / torch.sum(weights)  

        smoothed_values.append(smoothed_value.item())  
   
    smoothed_values = np.array(smoothed_values)  
  
    return smoothed_values  
    
MAX_EPOCHES = 10000


def Load_ElevenSharpArrows():
    data = json.load(open("data/ElevenSharpArrows/data.json"))
    # lines = []
    lines_harmless = [h["query"] for h in data['harmless']]
    lines = [h["query"] for h in data['harmful']]

    return lines, lines_harmless


def main():  
 
    patch_open()  
 
    parser = argparse.ArgumentParser()  
  

    parser.add_argument("--pretrained_model_path", type=str, required=True)  

    parser.add_argument("--config", type=str, choices=["greedy", "sampling"], required=True)  
 
    parser.add_argument("--output_path", type=str, default='./estimations')  
    
    parser.add_argument("--system_prompt_type", type=str, choices=['all'], required=True)  
  
    # 添加命令行参数 --n_splits，用于指定数据拆分的份数，默认值为10  
    parser.add_argument("--n_splits", type=int, default=10)  
  
    # 解析命令行参数  
    args = parser.parse_args()  
  
    # 使用logging模块记录命令行参数  
    for k, v in vars(args).items():  
        logging.info(f"{k}: {v}")  
    
    dataset = 'custom'
  
    # 读取'./data/custom.txt'中的数据，并去除空行  
    with open(f"./data/{dataset}.txt") as f:  
        lines = f.readlines()  
  
    # 读取'./data_harmless/custom.txt'中的数据，并去除空行  
    with open(f"./data_harmless/{dataset}.txt") as f:  
        lines_harmless = f.readlines()  
  
    # 将读取的数据转换为不含空行的列表，并获取查询数量  
    all_queries = [e.strip() for e in lines if e.strip()]  
    n_queries = len(all_queries)  
    logging.info(f"{n_queries} harmful queries")
  
    # 对无害数据集做同样处理，并获取无害查询数量  
    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]  
    n_queries_harmless = len(all_queries_harmless)  
    logging.info(f"{n_queries_harmless} harmless queries")
  
    # 断言检查，确保两个数据集中的查询数量相等  
    assert n_queries == n_queries_harmless, f"{n_queries} 不等于 {n_queries_harmless}"  
  
    # 断言检查，确保查询数量能被n_splits整除  
    assert n_queries % args.n_splits == 0, f"{n_queries} 除以 {args.n_splits} 的余数不为 0"  
  
    # 记录CUDA内存使用情况（需要logging_cuda_memory_usage函数定义）  
    logging_cuda_memory_usage()  
  
    # 清除CUDA缓存  
    torch.cuda.empty_cache()  
  
    # 触发Python的垃圾回收机制  
    gc.collect()  
  
    # 记录预训练模型路径  
    logging.info(args.pretrained_model_path)  
  
    # 准备模型  
    # 从预训练模型路径中提取模型名称  
    model_name = args.pretrained_model_path.split('/')[-1]  
  
    # 从预训练模型路径加载配置信息  
    config = AutoConfig.from_pretrained(args.pretrained_model_path)  
  
    # 获取配置中的隐藏层数量  
    num_layers = config.num_hidden_layers  
  
    # 创建输出目录（如果已存在则不报错）  
    os.makedirs(f'{args.output_path}/{model_name}_{args.system_prompt_type}', exist_ok=True) 
      
    # 标记为有害数据的处理  
    # 输出正在运行有害数据处理的日志信息  
    logging.info(f"Running harmful")  
    
    # 使用safe_open函数（可能是自定义的）打开存储隐藏状态的safetensors文件  
    # 获取模型在最后一层（num_layers-1）的隐藏状态  
    hidden_states = safe_open(f'hidden_states_baseline/{model_name}_{dataset}.safetensors',  
                                framework='pt', device=0)  
    
    # 类似地，获取带有默认设置的隐藏状态  
    hidden_states_with_default = safe_open(f'hidden_states_baseline/{model_name}_with_default_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 获取带有“short”设置的隐藏状态  
    hidden_states_with_short = safe_open(f'hidden_states_baseline/{model_name}_with_short_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 获取带有“mistral”设置的隐藏状态  
    hidden_states_with_mistral = safe_open(f'hidden_states_baseline/{model_name}_with_mistral_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 初始化列表来存储所有查询的隐藏状态  
    all_hidden_states = []  
    all_hidden_states_with_default = []  
    all_hidden_states_with_short = []  
    all_hidden_states_with_mistral = []  
    
    # 遍历所有查询  
    for idx, query in enumerate(all_queries):  
        # 从每个safetensors文件中获取指定索引和层的隐藏状态  
        # 注意：这里假设每个查询的隐藏状态存储在'sample.{idx}_layer.{num_layers-1}'的键下  
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
    
        # 将每个查询的隐藏状态添加到相应的列表中  
        all_hidden_states.append(tmp_hidden_states)  
        all_hidden_states_with_default.append(tmp_hidden_states_with_default)  
        all_hidden_states_with_short.append(tmp_hidden_states_with_short)  
        all_hidden_states_with_mistral.append(tmp_hidden_states_with_mistral) 
    
    
    # 无害数据处理  
    # 输出正在运行无害数据处理的日志信息  
    logging.info(f"Running harmless")  
    
    # 使用safe_open函数（可能是自定义的）打开存储无害隐藏状态的safetensors文件  
    # 这里的文件名路径和文件名后缀表示是无害的（harmless）数据  
    hidden_states = safe_open(f'hidden_states_baseline_harmless/{model_name}_{dataset}.safetensors',  
                                framework='pt', device=0)  
    
    # 类似地，获取带有默认设置的无害隐藏状态  
    hidden_states_with_default = safe_open(f'hidden_states_baseline_harmless/{model_name}_with_default_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 获取带有“short”设置的无害隐藏状态  
    hidden_states_with_short = safe_open(f'hidden_states_baseline_harmless/{model_name}_with_short_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 获取带有“mistral”设置的无害隐藏状态  
    hidden_states_with_mistral = safe_open(f'hidden_states_baseline_harmless/{model_name}_with_mistral_{dataset}.safetensors',  
                                        framework='pt', device=0)  
    
    # 初始化列表来存储所有无害查询的隐藏状态  
    all_hidden_states_harmless = []  
    all_hidden_states_with_default_harmless = []  
    all_hidden_states_with_short_harmless = []  
    all_hidden_states_with_mistral_harmless = []  
    
    # 遍历所有无害查询  
    for idx, query_harmless in enumerate(all_queries_harmless):  
        # 从每个safetensors文件中获取指定索引和层的无害隐藏状态  
        # 注意：这里假设每个查询的隐藏状态存储在'sample.{idx}_layer.{num_layers-1}'的键下  
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
        tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]  
    
        # 将每个无害查询的隐藏状态添加到相应的列表中  
        all_hidden_states_harmless.append(tmp_hidden_states)  
        all_hidden_states_with_default_harmless.append(tmp_hidden_states_with_default)  
        all_hidden_states_with_short_harmless.append(tmp_hidden_states_with_short)  
        all_hidden_states_with_mistral_harmless.append(tmp_hidden_states_with_mistral)  
    
    # 将所有的隐藏状态列表转换为PyTorch张量，并堆叠到一起  
    # 这里的堆叠操作假设所有隐藏状态的形状都是一致的，以便可以沿一个新的维度堆叠  
    all_hidden_states = torch.stack(all_hidden_states)  # 堆叠原始隐藏状态  
    all_hidden_states_with_default = torch.stack(all_hidden_states_with_default)  # 堆叠带有默认设置的隐藏状态  
    all_hidden_states_with_short = torch.stack(all_hidden_states_with_short)  # 堆叠带有“short”设置的隐藏状态  
    all_hidden_states_with_mistral = torch.stack(all_hidden_states_with_mistral)  # 堆叠带有“mistral”设置的隐藏状态  
    
    # 对于无害数据，也执行相同的堆叠操作  
    all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)  # 堆叠无害的原始隐藏状态  
    all_hidden_states_with_default_harmless = torch.stack(all_hidden_states_with_default_harmless)  # 堆叠无害且带有默认设置的隐藏状态  
    all_hidden_states_with_short_harmless = torch.stack(all_hidden_states_with_short_harmless)  # 堆叠无害且带有“short”设置的隐藏状态  
    all_hidden_states_with_mistral_harmless = torch.stack(all_hidden_states_with_mistral_harmless)  # 堆叠无害且带有“mistral”设置的隐藏状态  
    
    # 调用get_following_indices函数来获取不同设置下的分数  
    # 这些分数可能是基于隐藏状态或其他模型输出计算得出的  
    # 注意：get_following_indices函数的具体实现和参数意义在这里没有给出，但我们可以从参数猜测其功能  
    
    # 获取不使用无害数据的分数  
    scores = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_harmless=False, return_only_scores=True)  
    
    # 获取使用无害数据的分数  
    scores_harmless = get_following_indices(  
        model_name, dataset=dataset,config=args.config, use_harmless=True, return_only_scores=True)  
    
    # 获取使用默认提示且不使用无害数据的分数  
    scores_with_default = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_default_prompt=True, use_harmless=False, return_only_scores=True)  
    
    # 获取使用默认提示且使用无害数据的分数  
    scores_with_default_harmless = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_default_prompt=True, use_harmless=True, return_only_scores=True)  
    
    # 获取使用“short”提示且不使用无害数据的分数  
    scores_with_short = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_short_prompt=True, use_harmless=False, return_only_scores=True)  
    
    # 获取使用“short”提示且使用无害数据的分数  
    scores_with_short_harmless = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_short_prompt=True, use_harmless=True, return_only_scores=True)  
    
    # 获取使用“mistral”提示且不使用无害数据的分数  
    scores_with_mistral = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_mistral_prompt=True, use_harmless=False, return_only_scores=True)  
    
    # 获取使用“mistral”提示且使用无害数据的分数  
    scores_with_mistral_harmless = get_following_indices(  
        model_name, dataset=dataset, config=args.config, use_mistral_prompt=True, use_harmless=True, return_only_scores=True)  
    
    # 注意：这里假设get_following_indices函数根据提供的参数返回相应的分数  
    # 具体的计算逻辑和分数的意义取决于该函数的实现

    # 将计算得到的分数转换为PyTorch张量，并移动到CUDA设备上，同时指定数据类型为float  
    scores = torch.tensor(scores, device='cuda', dtype=torch.float)  
    scores_harmless = torch.tensor(scores_harmless, device='cuda', dtype=torch.float)  
    scores_with_default = torch.tensor(scores_with_default, device='cuda', dtype=torch.float)  
    scores_with_default_harmless = torch.tensor(scores_with_default_harmless, device='cuda', dtype=torch.float)  
    scores_with_short = torch.tensor(scores_with_short, device='cuda', dtype=torch.float)  
    scores_with_short_harmless = torch.tensor(scores_with_short_harmless, device='cuda', dtype=torch.float)  
    scores_with_mistral = torch.tensor(scores_with_mistral, device='cuda', dtype=torch.float)  
    scores_with_mistral_harmless = torch.tensor(scores_with_mistral_harmless, device='cuda', dtype=torch.float)  
    
    # 将所有隐藏状态按照无害和有害条件进行合并，并在第一个维度（即batch维度）上进行拼接  
    # 注意，这里将所有无害的隐藏状态放在了前面  
    hidden_states = torch.cat([  
        all_hidden_states_harmless,  
        all_hidden_states_with_default_harmless,  
        all_hidden_states_with_short_harmless,  
        all_hidden_states_with_mistral_harmless,  
        all_hidden_states,  
        all_hidden_states_with_default,  
        all_hidden_states_with_short,  
        all_hidden_states_with_mistral,  
    ], dim=0).float()  
    
    # 创建一个PCA对象，并指定PCA的维度和随机状态  
    pca = PCA(PCA_DIM, random_state=42)  
    # 使用CPU上的数据来拟合PCA模型  
    pca.fit(hidden_states.cpu().numpy())  
    # 记录PCA解释方差的比例以及它们的总和  
    logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, sum: {np.sum(pca.explained_variance_ratio_)}")  
    
    # 将PCA的均值转换为PyTorch张量，并移动到CUDA设备上  
    mean = torch.tensor(pca.mean_, dtype=torch.float, device='cuda')  
    # 将PCA的主成分（即转换矩阵的转置）转换为PyTorch张量，并移动到CUDA设备上  
    V = torch.tensor(pca.components_.T, dtype=torch.float, device='cuda')  
    
    # 获取转换矩阵V的行数（即PCA的主成分数量）  
    n = V.size(0)  
    
    # 创建一个正交基础列表，这里虽然列出了V的每一列，但接下来并没有直接使用这个列表  
    basis = [V[:, i] for i in range(V.size(1))]  
    
    # 设置随机种子以确保结果的可复现性  
    set_seed(42)  
    # 生成一个随机的张量all_vectors，其形状为(n*2, n)，用于后续的正交化过程  
    all_vectors = torch.randn((n*2, n), device='cuda', dtype=torch.double)  
    
    # 使用Gram-Schmidt过程来生成一个正交的基础，但注意这里实际上并没有使用原始的basis列表，而是直接使用了all_vectors  
    # 这可能是一个逻辑错误，因为通常Gram-Schmidt过程会使用一组线性无关的向量作为输入来生成正交基  
    orthogonal_basis = gram_schmidt(all_vectors, basis, n)  
    
    # 将PCA的均值和正交基础保存到文件中  
    save_file({'mean': mean, 'V': orthogonal_basis}, f'{args.output_path}/{model_name}_{args.system_prompt_type}/transform.safetensors')  
    

    def train_model(model, train_X, train_Y, test_X=None, test_Y=None):  
        # 使用AdamW优化器，并设置学习率为1e-4  
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  
    
        # 初始化最小epoch为0，初始化最小的测试BCE损失和MSE损失为一个较大的值  
        min_epoch = 0  
        min_test_bce_loss = 1e5  
        min_test_mse_loss_by_bce_loss = 1e5  
    
        # 初始化当前epoch为0  
        epoch = 0  
    
        # 初始化测试损失连续未降低的次数为0  
        test_loss_drop_times = 0  
    
        # 复制模型以存储最佳模型  
        best_model_copy = deepcopy(model)  
    
        # 无限循环，直到满足停止条件  
        while True:  
            # 将模型设置为训练模式  
            model.train()  
    
            # 清零梯度  
            optimizer.zero_grad()  
    
            # 前向传播，计算训练数据的logits（未经过sigmoid激活）  
            train_logits = model(train_X).squeeze(-1)  
    
            # 计算二元交叉熵损失  
            bce_loss = F.binary_cross_entropy_with_logits(train_logits, train_Y)  
    
            # 梯度裁剪，防止梯度爆炸  
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
    
            # 反向传播  
            bce_loss.backward()  
    
            # 更新参数  
            optimizer.step()  
    
            # 更新epoch  
            epoch += 1  
    
            # 如果有测试集，则在测试集上进行评估  
            if test_X is not None:  
                # 设置模型为评估模式  
                with torch.no_grad():  
                    model.eval()  
    
                    # 计算测试数据的logits  
                    test_logits = model(test_X).squeeze(-1)  
    
                    # 计算测试集的二元交叉熵损失  
                    test_bce_loss = F.binary_cross_entropy_with_logits(test_logits, test_Y).detach()  
    
                    # 计算测试集的MSE损失（通过sigmoid激活后的logits与真实标签比较）  
                    test_mse_loss = F.mse_loss(test_logits.sigmoid(), test_Y).detach()  
            else:  
                # 如果没有测试集，则在训练集上计算损失作为替代  
                test_bce_loss = bce_loss.detach()  
                test_mse_loss = F.mse_loss(train_logits.sigmoid(), train_Y).detach()  
    
            # 检查是否找到了更小的测试BCE损失  
            if min_test_bce_loss is None or test_bce_loss < min_test_bce_loss - 1e-4:  
                # 更新最小的测试BCE损失和MSE损失  
                min_test_bce_loss = test_bce_loss  
                min_test_mse_loss_by_bce_loss = test_mse_loss  
    
                # 复制当前模型作为最佳模型  
                best_model_copy = deepcopy(model)  
    
                # 重置测试损失连续未降低的次数  
                test_loss_drop_times = 0  
    
                # 更新最小epoch  
                min_epoch = epoch  
            else:  
                # 否则，增加测试损失连续未降低的次数  
                test_loss_drop_times += 1  
    
            # 检查是否达到最大epoch数或测试损失连续未降低次数达到3次  
            if epoch == MAX_EPOCHES or test_loss_drop_times == 3:  
                # 记录最终的epoch、BCE损失和MSE损失  
                logging.info(f"最终epoch {min_epoch}: BCE损失 {min_test_bce_loss.item()}, MSE损失 {min_test_mse_loss_by_bce_loss.item()}")  
                # 跳出循环  
                break  
    
        # 返回最佳模型  
        return best_model_copy


    # 创建一个包含从0到8*n_queries的索引列表  n_queries=1100
    total_indices = list(range(8*n_queries))  
    
    # 设置随机种子，以便结果可复现  
    set_seed(42)  
    # 打乱total_indices中的索引  
    random.shuffle(total_indices)  
    
    # 记录信息，表示开始训练拒绝模型  
    logging.info(f"Training refusal model")  
    
    # 初始化拒绝模型的线性权重和偏置的列表  
    refusal_linear_weights = []  
    refusal_linear_biases = []  
    
    # 对于每一个分割（n_splits定义了分割的数量）  
    for split_idx in range(args.n_splits):  
        # 如果分割数量大于1  
        if args.n_splits > 1:  
            # 计算当前分割的测试集索引  
            test_indices = total_indices[split_idx * 8*n_queries//args.n_splits: (split_idx+1) * 8*n_queries//args.n_splits]  
            # 计算当前分割的训练集索引（除了当前分割的测试集外的所有索引）  
            train_indices = total_indices[:split_idx * 8*n_queries//args.n_splits] + total_indices[(split_idx+1) * 8*n_queries//args.n_splits:]  
        else:  
            # 如果没有分割（即只训练一个模型），则测试集索引为None，训练集索引为所有索引  
            test_indices = None  
            train_indices = total_indices  
    
        # 将多个隐藏状态数据拼接起来，并按照训练集索引进行筛选  
        train_X = torch.cat([  
            all_hidden_states,  
            all_hidden_states_with_default,  
            all_hidden_states_with_short,  
            all_hidden_states_with_mistral,  
            all_hidden_states_harmless,  
            all_hidden_states_with_default_harmless,  
            all_hidden_states_with_short_harmless,  
            all_hidden_states_with_mistral_harmless,  
        ], dim=0).float()[train_indices]  
    
        # 对train_X进行标准化处理（这里假设mean和V是预先计算好的均值和变换矩阵）  
        train_X = torch.matmul(train_X - mean, V)  


        # 将多个分数数据拼接起来，并按照训练集索引进行筛选  
        train_Y = torch.cat([  
            scores,  
            scores_with_default,  
            scores_with_short,  
            scores_with_mistral,  
            scores_harmless,  
            scores_with_default_harmless,  
            scores_with_short_harmless,  
            scores_with_mistral_harmless,  
        ], dim=0)[train_indices]  

        # 如果存在测试集索引（即存在分割的情况）  
        if test_indices is not None:  
            # 将多个隐藏状态数据拼接起来，并按照测试集索引进行筛选  
            test_X = torch.cat([  
                all_hidden_states,  
                all_hidden_states_with_default,  
                all_hidden_states_with_short,  
                all_hidden_states_with_mistral,  
                all_hidden_states_harmless,  
                all_hidden_states_with_default_harmless,  
                all_hidden_states_with_short_harmless,  
                all_hidden_states_with_mistral_harmless,  
            ], dim=0).float()[test_indices]  
        
            # 对test_X进行标准化处理  
            test_X = torch.matmul(test_X - mean, V)  
        
            # 将多个分数数据拼接起来，并按照测试集索引进行筛选  
            test_Y = torch.cat([  
                scores,  
                scores_with_default,  
                scores_with_short,  
                scores_with_mistral,  
                scores_harmless,  
                scores_with_default_harmless,  
                scores_with_short_harmless,  
                scores_with_mistral_harmless,  
            ], dim=0)[test_indices]  
        else:  
            # 如果没有测试集索引（即不进行分割，使用全部数据作为训练集），则设置test_X和test_Y为None  
            test_X = None  
            test_Y = None  
        
        # 创建一个线性模型，输入维度为PCA_DIM（可能是之前某个PCA步骤得到的特征维度），输出维度为1  
        model_refusal = nn.Linear(PCA_DIM, 1).to('cuda')  
        
        # 使用train_model函数训练模型，并将训练好的模型赋值给model_refusal  
        # 注意：train_model函数不是PyTorch的标准函数，而是自定义的，它应该包含模型训练的逻辑  
        model_refusal = train_model(model_refusal, train_X, train_Y, test_X, test_Y)  
        
        # 将训练好的模型的权重和偏置从计算图中分离（即不再参与梯度计算），并添加到列表中  
        refusal_linear_weights.append(model_refusal.weight.detach())  
        refusal_linear_biases.append(model_refusal.bias.detach())

    # 日志信息，表明开始训练有害性模型  
    logging.info(f"Training harmfulness model")  
    
    # 初始化存储有害性模型线性层权重的列表  
    harmfulness_linear_weights = []  
    # 初始化存储有害性模型线性层偏置的列表  
    harmfulness_linear_biases = []  
    
    # 根据参数args.n_splits指定的分割次数进行循环  
    for split_idx in range(args.n_splits):  
        # 如果分割次数大于1，则进行数据的分割  
        if args.n_splits > 1:  
            # 计算当前分割的测试集索引范围  
            # 假设total_indices包含了所有数据的索引，8*n_queries可能是数据总量的一个倍数  
            test_indices = total_indices[split_idx * 8*n_queries//args.n_splits: (split_idx+1) * 8*n_queries//args.n_splits]  
            # 计算当前分割的训练集索引（除了当前分割的测试集外的所有索引）  
            train_indices = total_indices[:split_idx * 8*n_queries//args.n_splits] + total_indices[(split_idx+1) * 8*n_queries//args.n_splits:]  
        else:  
            # 如果不进行分割（即args.n_splits为1），则没有测试集，所有数据都是训练集  
            test_indices = None  
            train_indices = total_indices  
    
        # 将多个隐藏状态数据拼接起来，并按照训练集索引进行筛选  
        train_X = torch.cat([  
            all_hidden_states,  
            all_hidden_states_with_default,  
            all_hidden_states_with_short,  
            all_hidden_states_with_mistral,  
            all_hidden_states_harmless,  
            all_hidden_states_with_default_harmless,  
            all_hidden_states_with_short_harmless,  
            all_hidden_states_with_mistral_harmless,  
        ], dim=0).float()[train_indices]  
    

        # 对train_X进行标准化处理  
        train_X = torch.matmul(train_X - mean, V)  
        logging.info("harmlessness train_X shape")
        logging.info(train_X.shape)
    
        train_Y = torch.tensor([0 for _ in range(4*n_queries)] + [1 for _ in range(4*n_queries)], device=train_X.device, dtype=torch.float)[train_indices]  
        logging.info("harmlessness train_Y shape")
        logging.info(train_Y.shape)

    # logging.info("Test_Indices")
    # logging.info(test_indices)
    # 如果定义了测试集索引，则准备测试数据  
    
    if test_indices is not None:  
        # 将多个隐藏状态数据拼接起来，并按照测试集索引进行筛选  
        test_X = torch.cat([  
            all_hidden_states,  
            all_hidden_states_with_default,  
            all_hidden_states_with_short,  
            all_hidden_states_with_mistral,  
            all_hidden_states_harmless,  
            all_hidden_states_with_default_harmless,  
            all_hidden_states_with_short_harmless,  
            all_hidden_states_with_mistral_harmless,  
        ], dim=0).float()[test_indices]  
        
        # 对test_X进行标准化处理  
        test_X = torch.matmul(test_X - mean, V)  
    
        # 创建测试标签test_Y（这里假设与训练集标签生成方式相同，但在实际场景中可能需要根据真实数据来创建）  
        test_Y = torch.tensor([0 for _ in range(4*n_queries)] + [1 for _ in range(4*n_queries)], device=train_X.device, dtype=torch.float)[test_indices]  

    else:  
        # 如果没有定义测试集索引（即只有训练集），则不准备测试数据  
        test_X = None  
        test_Y = None  
    
    # 初始化有害性模型，这是一个线性层，输入维度为PCA_DIM，输出维度为1  
    model_harmfulness = nn.Linear(PCA_DIM, 1).to('cuda')  
    
    # 使用train_model函数来训练模型，该函数可能需要自定义，包括训练循环、优化器和损失函数等  
    # 这里同时传入训练集和测试集（如果有的话）  
    model_harmfulness = train_model(model_harmfulness, train_X, train_Y, test_X, test_Y)  
    
    # 将训练后的模型权重和偏置添加到列表中  
    harmfulness_linear_weights.append(model_harmfulness.weight.detach())  
    harmfulness_linear_biases.append(model_harmfulness.bias.detach())  
    
    # 将之前收集的权重和偏置堆叠成一个张量，并转移到CPU上  
    refusal_linear_weight = torch.stack(refusal_linear_weights).cpu()  
    refusal_linear_bias = torch.stack(refusal_linear_biases).cpu()  
    harmfulness_linear_weight = torch.stack(harmfulness_linear_weights).cpu()  
    harmfulness_linear_bias = torch.stack(harmfulness_linear_biases).cpu()  
    
    # 将权重和偏置保存到文件中，文件名包含模型名称和系统提示类型  
    save_file({'weight': refusal_linear_weight, 'bias': refusal_linear_bias},  
            f'{args.output_path}/{model_name}_{args.system_prompt_type}/refusal.safetensors')  
    save_file({'weight': harmfulness_linear_weight, 'bias': harmfulness_linear_bias},  
            f'{args.output_path}/{model_name}_{args.system_prompt_type}/harmfulness.safetensors')  
    
    # 记录日志，表示训练已完成  
    logging.info(f"Training finished")  
    
    # 记录并输出当前CUDA内存使用情况（这个函数需要自定义）  
    logging_cuda_memory_usage()  
    
    # 清除CUDA缓存  
    torch.cuda.empty_cache()  
    
    # 强制Python的垃圾回收器运行，回收未使用的内存  
    gc.collect()

if __name__ == "__main__":
    main()
