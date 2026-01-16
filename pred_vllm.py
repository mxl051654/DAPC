import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp

from datasets import load_dataset
from transformers import AutoTokenizer
from model_openai import OpenAI_Model
import concurrent.futures
import threading
import queue
from consts import (
    GENERATE_SIZE, IDNAME, DATASET_LIST,
    get_method_para, prepare_query
)

MAX_LENGTH = 32768 - 2 * 4096


def read_jsonlines(path, column=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if column is not None:
                    data.append(json.loads(line)[column])
                else:
                    data.append(json.loads(line))  # 每一行都是一个 JSON 字符串
            except Exception as e:
                pass
                # print(f"Json parse error: {e}")
    return data


def file_writer(q, out_path):
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            item = q.get()
            if item is None:
                break
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            q.task_done()


def process_request(args, json_obj, message, llm, q=None):
    max_token = GENERATE_SIZE[args.bench].get(json_obj['dataset'], 128)

    json_obj['pred'] = llm.generate(message, temperature=0.5, max_tokens=max_token)
    for k, v in json_obj.items():
        if type(v) == str:
            json_obj[k] = v.replace('"', ' ')  # .replace('\n', ' ')

    if q is not None:
        q.put(json_obj)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class BudgetAllocator:
    def __init__(self, model_name, budget_policy='global', compress_length=None, compress_ratio=None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert budget_policy in ['chunk', 'global']
        self.budget_policy = budget_policy
        self.compress_length = compress_length
        self.compress_ratio = compress_ratio

    def get_compressed(self, data):
        print(f'Construct Compressed Dataset \nBudget_policy:{self.budget_policy} '
              f'Target_len: {self.compress_length} Ratio: {self.compress_ratio}')

        new_data = []
        for item in data:

            tokens = item['original_tokens']  # list[float]
            scores = item['scores']  # list[list[int]] score for each chunk
            assert sum([len(s) for s in scores]) == len(tokens)

            if self.budget_policy == 'chunk':
                compressed_list = []
                shift = 0
                for chunk_score in scores:

                    # 计算每个chunk需要保留的 token数量
                    chunk_len = len(chunk_score)
                    if self.compress_ratio is not None:
                        k = int(chunk_len * self.compress_ratio)
                    elif self.compress_length is not None:
                        total_len = len(tokens)
                        k = int(chunk_len / total_len * self.compress_length) if total_len > 0 else 0
                    else:
                        k = chunk_len

                    # TODO 基于分数，筛选 token
                    chunk_tokens = tokens[shift: shift + chunk_len]

                    if k < chunk_len:
                        top_indices = sorted(range(chunk_len), key=lambda i: chunk_score[i], reverse=True)[:k]
                        top_indices.sort()
                        select_ids = [chunk_tokens[i] for i in top_indices]
                    else:
                        select_ids = chunk_tokens

                    compressed_list.append(self.tokenizer.decode(select_ids, skip_special_tokens=False))
                    shift += chunk_len

                item['compressed_prompt'] = '\n\n'.join(compressed_list)

            elif self.budget_policy == 'global':

                # 直接拼接分数，然后筛选
                flat_scores = []
                for chunk in scores:
                    flat_scores.extend(chunk)

                total_len = len(tokens)
                if self.compress_ratio is not None:
                    k = int(total_len * self.compress_ratio)
                elif self.compress_length is not None:
                    k = self.compress_length
                else:
                    k = total_len

                if k < total_len:
                    top_indices = sorted(range(total_len), key=lambda i: flat_scores[i], reverse=True)[:k]
                    top_indices.sort()
                    select_ids = [tokens[i] for i in top_indices]
                else:
                    select_ids = tokens

                item['compressed_prompt'] = self.tokenizer.decode(select_ids, skip_special_tokens=False)

            new_data.append(item)

        return new_data


def load_data(args, dataset):
    data_dir = '/data/hf'
    print(f"\n===== {dataset} =====\n")
    data_name = f"{dataset}_e" if args.bench == 'longbench-e' else dataset
    if args.method in ["original", 'zero-shot']:
        data = load_dataset(f'{data_dir}/THUDM/LongBench', data_name, split='test')
    else:
        method_str = get_method_para(args)
        score_path = f"{args.compress_model.split('/')[-1]}_compressed/{args.bench}/{data_name}_{method_str}.jsonl"
        print(f"Load compressed data from {score_path}")
        if os.path.exists(score_path):
            data = read_jsonlines(score_path)
        else:
            data = None
            print(f"!!! no compressed files")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/data/hf/Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--compress_model', type=str, default='/data/hf/Qwen/Qwen2.5-7B-Instruct')
    # NOTE  dataset parameter
    parser.add_argument('--bench', type=str, default='longbench')
    parser.add_argument('--dataset', type=str, default=None)
    # NOTE  compress parameter
    parser.add_argument('--method', type=str,
                        # default='dac'
                        # default='lrp-qa',
                        # default='original',
                        # default='zero-shot',
                        default='kvzip'
                        )
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--contrast_alpha', type=float, default=1.0)
    parser.add_argument('--rollout_m', type=int, default=1)  # 1 ,2,  3
    parser.add_argument('--compress_rate', type=float, default=None)
    parser.add_argument('--target_len', type=int, default=2000, )
    parser.add_argument('--budget_policy', type=str, default='chunk')

    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--port', default='8002')
    parser.add_argument('--rerun', action='store_true', help="Rerun and overwrite existing results")
    args = parser.parse_args()

    seed_everything(42)
    mp.set_start_method('spawn', force=True)

    model_name = args.model.split('/')[-1]
    llm = OpenAI_Model(args.model, base_url=f"http://localhost:{args.port}/v1")
    tokenizer = AutoTokenizer.from_pretrained(args.compress_model)  # NOTE corresponding to compress model

    assert args.bench in ['longbench', 'longbench-e', 'infinitebench']
    if args.dataset:
        datasets = [args.dataset]
    elif args.bench == 'longbench-e':
        datasets = DATASET_LIST['longbench-e']
    elif args.bench == 'longbench':
        datasets = DATASET_LIST['longbench']

    for dataset in datasets:
        print(f"==={args.compress_model}====")
        data = load_data(args, dataset)
        method_str = get_method_para(args, infer=True)
        out_dir = f"{args.compress_model.split('/')[-1]}_pred/{args.bench}_{model_name}_{method_str}/result"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = f"{out_dir}/{dataset}.jsonl"

        # NOTE 加载已经处理好的记录
        finished_ids = []
        # if args.rerun and os.path.exists(out_path):
        #     os.remove(out_path)

        if os.path.exists(out_path):
            finished_ids = read_jsonlines(out_path, IDNAME[args.bench])
        print(f"Have processed {len(finished_ids)} @ {out_path}")

        # NOTE 构建数据
        if args.method in ['original', 'zero-shot']:
            pass
        elif 'llmlingua' in args.method:
            pass
        else:
            bc = BudgetAllocator(model_name=args.compress_model,
                                 budget_policy=args.budget_policy,
                                 compress_length=args.target_len,
                                 compress_ratio=args.compress_rate)
            data = bc.get_compressed(data)  # NOTE set new compressed_prompt

        todos = []
        for json_obj in data:
            if json_obj.get("compressed_prompt"):
                json_obj["context"] = json_obj["compressed_prompt"]  # NOTE !!!
            if json_obj[IDNAME[args.bench]] in finished_ids:
                continue
            # NOTE 裁剪上下文
            tokenized_prompt = tokenizer(json_obj["context"], truncation=False, return_tensors="pt").input_ids[0]
            while len(tokenized_prompt) > MAX_LENGTH:
                half = int(0.9 * MAX_LENGTH)
                json_obj["context"] = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                tokenized_prompt = tokenizer(json_obj["context"], truncation=False, return_tensors="pt").input_ids[0]

            prompt = prepare_query(json_obj, args.bench, dataset, infer=True, method=method_str)
            message = [{'role': 'user', 'content': prompt}]
            todos.append((json_obj, message))

        # NOTE 并发执行模型预测请求
        print(f"TODO num : {len(todos)}")
        result_queue = queue.Queue()
        writer = threading.Thread(target=file_writer, args=(result_queue, out_path))
        writer.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # 提交任务并等待执行完成
            futures = [executor.submit(process_request, args, json_obj, message, llm, result_queue)
                       for json_obj, message in todos]
            # 使用 tqdm 显示任务进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()  # 获取返回值，若有异常会抛出
        result_queue.put(None)
        writer.join()


if __name__ == '__main__':
    main()
