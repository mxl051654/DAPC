import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Iterable

from llmlingua import (
    PromptCompressor,
    DACPromptCompressor,
    EHPCPromptCompressor,
    LRPPromptCompressor
)

from transformers import Qwen2ForCausalLM
from consts import my_load_dataset, get_method_para, prepare_query, IDNAME


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
                continue
    return data


class CompressorFactory:
    def __init__(
            self,
            args,
            method: str = "llmlingua",
            model_name: Optional[str] = None,
            compress_rate: Optional[float] = 0.5,  # 0-1
            target_token: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        self.args = args
        self.method = method
        self.compress_rate = compress_rate
        self.target_token = target_token

        init_kwargs: Dict[str, Any] = {}
        if model_name is not None:
            init_kwargs["model_name"] = model_name  # 模型名称

        if method in [
            'llmlingua',
            'llmlingua-2',
            'longllmlingua',
        ]:
            if method == "llmlingua-2":
                init_kwargs["use_llmlingua2"] = True  # 是否启用 LLMLingua-2
            if method == 'llmlingua':
                init_kwargs["model_name"] = "/data/hf/NousResearch/Llama-2-7b-hf"
                # model_name="/data/hf/microsoft/phi-2"
                # model_name="TheBloke/Llama-2-7b-Chat-GPTQ"
            elif method == 'llmlingua-2':
                init_kwargs["model_name"] = "/data/hf/microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
                # model_name="/data/hf/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
            elif method == 'longllmlingua':
                # NOTE pip install transformers==4.38.2
                # NOTE LRP transformer-lens  transformers>=4.51
                init_kwargs["model_name"] = "/data/hf/NousResearch/Llama-2-7b-hf"
            self.compressor = PromptCompressor(**{**kwargs, **init_kwargs})

        elif method in [
            'dac',
            'attn',
            'ppl'
        ]:
            self.compressor = DACPromptCompressor(model_name=args.model)

        elif method in [
            "attn-qa",
            "ehpc",
            "kvzip"
            'p-contrast',
            'p-contrast-qa',
            'rollout',
            'rollout-qa',
            'contrast-rollout-qa',
            'rollout-contrast-qa',
        ]:
            self.compressor = EHPCPromptCompressor(model_name=args.model)

        elif method in [
            'lrp-qa'
        ]:
            self.compressor = LRPPromptCompressor(model_name=args.model)
        else:
            print(f"No implementation for method {method}")

    def __str__(self) -> str:
        if self.target_token is not None:
            return f"{self.method}_t{self.target_token}"
        else:
            return f"{self.method}_r{self.compress_rate:.2f}"

    def compress(
            self,
            context: List[str],
            instruction: str = "",
            question: str = "",
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        query-agnostic
        - LLMLingua ：不切块，直接对整串文本做“迭代窗口 + KV-Cache”压缩，适合因果 LM，能在超长场景下流式推进。
        - LLMLingua-2 ：必须切块（编码器 512 限制），在块内做词级分类压缩，并可先做块/段的上下文级筛选，天然适合极长输入的分段处理与加速。
        query-aware
        - LongLLMLingua ：先粗筛与重排（上下文/句子），大幅缩短需要细压的范围，再用 LLMLingua 的迭代窗口做细压，专门解决长上下文的相关性与位置信息问题。
        """
        call_kwargs: Dict[str, Any] = dict(kwargs)
        if self.compress_rate is not None:
            call_kwargs["rate"] = self.compress_rate  # 压缩率
            call_kwargs.setdefault("target_token", -1)
        else:
            call_kwargs["target_token"] = self.target_token  # 目标 token 数
            call_kwargs.setdefault("rate", 0.5)

        if self.method == "llmlingua":
            return self.compressor.compress_prompt(
                instruction=instruction,  # 指令
                context=context,  # 上下文
                question=question,  # 问题
                **call_kwargs,  # 其他参数（如 iterative_size, context_budget 等）
            )
        elif self.method == "llmlingua-2":
            return self.compressor.compress_prompt(
                instruction=instruction,
                context=context,
                question=question,
                **call_kwargs,
            )
        elif self.method == "longllmlingua":
            """
            Task(Question)-Aware Compression, 需要提供question
            """
            call_kwargs.setdefault("use_sentence_level_filter", False)
            call_kwargs.setdefault("condition_in_question", "after_condition")
            call_kwargs.setdefault("reorder_context", "sort")
            call_kwargs.setdefault("dynamic_context_compression_ratio", 0.3)
            call_kwargs.setdefault("condition_compare", True)
            call_kwargs.setdefault("context_budget", "+100")
            call_kwargs.setdefault("rank_method", "longllmlingua")
            return self.compressor.compress_prompt(
                instruction=instruction,
                context=context,
                question=question,
                **call_kwargs,
            )
        elif self.method in [
            'dac',
            'attn',
            'ppl'
        ]:
            return self.compressor.compress(
                context=context,
                compress_ratio=self.compress_rate,
                target_token=self.target_token,
                method="attn_ppl" if self.method == 'dac' else self.method,
                fusion="additive",
                alpha=0.8,
                dyn_time=None,
                preserve_punct=False,
                return_info=True
            )

        elif self.method in [
            'attn-qa',
            "ehpc", 
            "kvzip",
            'p-contrast', 
            'p-contrast-qa', 
            'rollout',
            'rollout-qa', 
            'contrast-rollout-qa',
            'rollout-contrast-qa',
        ]:
            # 基于 复现任务提示的注意力识别, 迁移到PC
            return self.compressor.compress(
                context=context,
                question=question,
                compress_ratio=self.compress_rate,
                target_token=self.target_token,
                method=self.method,
                preserve_punct=False,
                return_info=True,
                # NOTE
                rollout_m=self.args.rollout_m,
                contrast_alpha=self.args.contrast_alpha,
                chunk_size=self.args.chunk_size,
            )
        elif self.method == 'lrp-qa':
            return self.compressor.compress(
                context=context,
                question=question,
                compress_ratio=self.compress_rate,
                target_token=self.target_token,
                method=self.method,
                preserve_punct=False,
                return_info=True
            )
        else:
            raise NotImplementedError(f"未知的压缩方法: {self.method}")

    def compress_dataset(
            self,
            bench,
            task,
            data: Any,
            save_path: str,
            have_processed: list,
            context_key: str = "context",
            instruction_key: Optional[str] = None,
            question_key: Optional[str] = "input",
            **kwargs: Any,
    ) -> None:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:

            for json_obj in progress_bar(data, desc="Compressing"):

                if json_obj.get(IDNAME[bench]) in have_processed:  # NOTE if have_processed '_id'
                    continue

                ctx = json_obj.get(context_key, "")
                inst = json_obj.get(instruction_key, "") if instruction_key else ""
                ques = prepare_query(json_obj, bench, task, infer=False)

                comp = self.compress(
                    context=ctx,
                    instruction=inst,
                    question=ques,
                    **kwargs,
                )
                new_obj = dict(json_obj)
                if comp.get("compressed_prompt"):
                    for k, v in comp.items():
                        new_obj[k] = v
                    f.write(json.dumps(new_obj, ensure_ascii=False))
                    f.write("\n")
                else:
                    print('Key name error')


def progress_bar(iterable: Iterable, desc: str = "", total: Optional[int] = None):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=False,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {rate_fmt}",
    )


def run_compress() -> None:
    parser = argparse.ArgumentParser()
    # NOTE  compress parameter
    parser.add_argument('--method', type=str,
                        # default='dac',
                        # default='ehpc',
                        # default='p-contrast-qa',
                        # default='rollout-qa',
                        default='kvzip',
                        )
    parser.add_argument('--model', type=str, default='/data/hf/Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--contrast_alpha', type=float, default=1.0)
    parser.add_argument('--rollout_m', type=int, default=1)

    parser.add_argument('--target_token', type=int, default=2000)
    parser.add_argument('--compress_rate', type=float, default=None)
    # NOTE  dataset parameter
    parser.add_argument('--bench', type=str,
                        # default='longbench-e'
                        default='longbench'
                        )
    parser.add_argument('--dataset', type=str,
                        default='narrativeqa'
                        # default='lcc',
                        )
    parser.add_argument('--data_dir', type=str, default='/data/hf')
    parser.add_argument('--save_dir', type=str, default='compressed')
    args = parser.parse_args()

    # NOTE ============= get compressed data ====================
    args.save_dir = f"{args.model.split('/')[-1]}_compressed"  # set dir for compress model
    method_str = get_method_para(args, infer=False)
    data_name = f"{args.dataset}_e" if args.bench == 'longbench-e' else args.dataset
    compressed_path = os.path.join(args.save_dir, args.bench, f"{data_name}_{method_str}.jsonl")
    have_processed = []
    if os.path.exists(compressed_path):
        compressed_data = read_jsonlines(compressed_path)
        if len(compressed_data) > 0 and not compressed_data[0].get("scores"):
            os.remove(compressed_path)
        else:
            have_processed = [x[IDNAME[args.bench]] for x in compressed_data]
    else:
        have_processed = []
    print(f"Have processed {len(have_processed)} ({compressed_path})")

    # NOTE ============= get original data ====================
    original_data = my_load_dataset(args.bench, args.dataset)
    if len(original_data) == len(have_processed):
        return

    factory = CompressorFactory(
        args=args,
        method=args.method, target_token=args.target_token, compress_rate=args.compress_rate
    )
    factory.compress_dataset(
        args.bench,
        args.dataset,
        original_data,
        compressed_path,
        have_processed=have_processed,
        context_key="context",
        instruction_key=None,
        question_key="input",
    )
    print(f"Have saved to {compressed_path}")


def synthetic_compress(
        bench: str,
        dataset: str,
        methods: List[str],
        alphas: List[float]
):
    assert len(methods) == len(alphas), "Methods and alphas must have the same length."

    contrast_path = f"compressed/{bench}/{dataset}_p-contrast-qa_cs1024_ca1.0.jsonl"
    lrp_path = f"compressed/{bench}/{dataset}_lrp-qa_cs1024.jsonl"
    # e.g. synthetic_p-rollout-qa_0.5_p-contrast-qa_0.5
    mix_name = "-".join([f"{m}-{a}" for m, a in zip(methods, alphas)])
    # synthetic method [add multiple max min]
    method_name = f"add-{mix_name}"
    save_path = f"compressed/{bench}/{dataset}_{method_name}_cs1024_ca1.0.jsonl"

    data_dicts = []
    for fp in [contrast_path, lrp_path]:
        print(f"Reading {fp}...")
        if not os.path.exists(fp):
            print(f"File not found: {fp}")
            return
        data = read_jsonlines(fp)
        data_dicts.append({item['_id']: item for item in data if '_id' in item})

    # Intersection of IDs
    common_ids = set(data_dicts[0].keys())
    for d in data_dicts[1:]:
        common_ids &= set(d.keys())
    print(f"Found {len(common_ids)} common samples.")

    if len(common_ids) == 0:
        print("No common IDs found. Please check file contents.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with (open(save_path, "w", encoding="utf-8") as f):
        for _id in tqdm(common_ids, desc="Mixing scores"):
            items = [d[_id] for d in data_dicts]

            tokens_0 = items[0].get("original_tokens", [])
            scores_0 = []
            _ = [scores_0.extend(x) for x in items[0].get("scores", [])]  # Flatten
            scores_1 = []  # Flatten
            _ = [scores_1.extend(x) for x in items[1].get("scores", [])]

            assert len(scores_0) == len(scores_1)

            scores_0 = np.array(scores_0)
            scores_1 = np.array(scores_1)

            def normal(arr):
                a_max = np.max(arr)
                a_min = np.min(arr)

                return (arr - a_min) / (a_max - a_min)

            norm_scores_0 = normal(scores_0)
            norm_scores_1 = normal(scores_1)

            new_scores = alphas[0] * norm_scores_0 + alphas[1] * norm_scores_1
            new_scores_flat = new_scores.tolist()  # Convert back to list for JSON serialization

            # Re-chunk scores
            new_scores_chunked = []
            current_idx = 0
            original_chunks = items[0].get("scores", [])
            for chunk in original_chunks:
                chunk_len = len(chunk)
                new_scores_chunked.append(new_scores_flat[current_idx: current_idx + chunk_len])
                current_idx += chunk_len

            result = {
                "_id": _id,
                "method": method_name,
                "original_tokens": tokens_0,
                "scores": new_scores_chunked,
                **{k: v for k, v in items[0].items() if
                   k not in ["method", "compressed_prompt", "actual_ratio", "scores", "processing_time",
                             "original_tokens"]}
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved synthetic results to {save_path}")


if __name__ == '__main__':
    run_compress()
