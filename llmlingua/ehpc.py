"""

2501_EHPC_Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference(NeurIPS 2025)

1. 识别评估头：首先，通过上述的“大海捞针”试点实验，为目标LLM模型 f 确定其评估头集合 Cf。这是一个一次性的离线过程。

    ameyhengle/Multilingual-Needle-in-a-Haystack
    Multilingual Needle in a Haystack: 
    Investigating Long-Context Behavior of Multilingual Large Language Models (ACl 25)
    # To iterate through all available configurations and splits
    for config in ["4k", "8k", "16k", "32k"]:
        for lang in ["en", "es", "de", "zh", "vi", "ar", "hi"]:
            dataset = load_dataset("ameyhengle/Multilingual-Needle-in-a-Haystack", config, split=lang)
            print(f"Loaded config: {config}, language: {lang}, size: {len(dataset)}")

    answer_start_index, answer_sentence, prompt
        

2. 计算Token效用分数：对于一个需要压缩的新输入提示 x=(x1,x2,...,xN)：
  - 将输入提示送入模型 f，但只运行前几层（为了效率，并非所有层）。
  - 提取这些层中已识别的评估头 CfCf 的注意力分数。
  - 对于每个token xi，计算其最终的效用分数 s。计算公式如下：
  - si=Pool(concat(ahl[N,:]),r),∀(h,l)∈Cf
    - concat($a_{hl}[N, :]$)：将所有评估头在最后一个位置的注意力分数向量拼接起来。
    - Pool(•, r)：一个池化操作，如平均池化，r 是核大小。这个操作非常重要，它可以将分数相近的相邻token分组，确保被选中的token是连续的，从而提升压缩后文本的可读性和语义连贯性。
3. 选择与压缩：
  - 根据计算出的效用分数 s，对所有token进行排序。
  - 根据预设的压缩率（例如，压缩到2048个token），选择分数最高的token。
  - 将这些被选中的token按照原始顺序组合，形成压缩后的提示 x′x′，而其余token则被直接丢弃。
"""
import os.path

import jsonlines
from tqdm import tqdm
import torch
import numpy as np
from typing import List, Optional, Dict, Union, Tuple

# from brotlicffi import Compressor
from torch import Tensor
import time
import re
import json
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import is_bitsandbytes_available
from datasets import load_dataset

# CHUNK_SIZE = 1024


class EHPCPromptCompressor:
    """
    A comprehensive prompt compressor that supports compression 
    using entropy and Attention scores with additive/multiplicative fusion.
    """

    def __init__(
            self, model_name,
            device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
            model_config: dict = {},
    ):
        self.model_name = model_name  # The model used for compress
        self.pos_suffix = model_config.get("pos_suffix", "\n Focus on key information.")
        self.neg_suffix = model_config.get("neg_suffix", "\n Focus on redundant information.")
        self.contrast_alpha = float(model_config.get("contrast_alpha", 1))  # old 0.8
        model_config.pop('contrast_alpha', None)
        self.load_model(model_name, device_map, model_config)

    def load_model(self, model_name, device_map: str = "cuda", model_config: dict = {}):
        print(f"======== Loading {model_name} =========")
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )

        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        quantization_config = None
        if ("cuda" in str(self.device).lower()) and is_bitsandbytes_available():
            compute_dtype = torch.float16
            user_dtype = model_config.pop("torch_dtype", None)
            if isinstance(user_dtype, torch.dtype):
                compute_dtype = user_dtype
            elif isinstance(user_dtype, str) and hasattr(torch, user_dtype):
                compute_dtype = getattr(torch, user_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=model_config.pop("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=model_config.pop("bnb_4bit_use_double_quant", True),
            )
        dtype_arg = model_config.pop(
            "torch_dtype", "auto" if self.device == "cuda" else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_arg,
            device_map=device_map,
            config=config,
            ignore_mismatched_sizes=True,
            # NOTE `sdpa` attention does not support `output_attentions=True` or `head_mask`.
            #  Please set your attention to `eager` if you want any of these features.
            attn_implementation="eager",
            # attn_implementation="flash_attention_2",
            # quantization_config=quantization_config,
            **model_config,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        # self.max_position_embeddings = config.max_position_embeddings  # 32768
        self.max_position_embeddings = 3000
        # print(f"max_length", self.max_position_embeddings)
        self.model_config = config

    def identify_evaluator_heads(
            self, max_samples=200,
            config_path="/data/mxl/PC/longbench/llmlingua/ehpc_config.jsonl"
            # config_path="ehpc_config.jsonl"
    ) -> List[Tuple[int, int]]:

        if os.path.exists(config_path):
            with jsonlines.open(config_path, 'r') as f:
                heads_config = list(f)
        else:
            heads_config = [
                {"model_name": "/data/hf/meta_llama/Llama-3.1-8B-Instruct",
                 "target_heads": [[13, 18], [13, 13], [13, 21], [13, 8], [13, 11], [13, 1], [13, 4], [13, 3]]},
                {"model_name": "/data/hf/codellama/CodeLlama-7b-hf",
                 "target_heads": [[14, 24], [14, 3], [14, 18], [14, 7], [14, 29], [14, 2], [14, 9], [14, 1]]},
                {"model_name": "/data/hf/microsoft/Phi-3.5-mini-instruct",
                 "target_heads": [[17, 7], [17, 17], [17, 30], [17, 2], [17, 6], [17, 16], [17, 25], [17, 18]]},
                {"model_name": "/data/hf/Qwen/Qwen2-0.5B-Instruct",
                 "target_heads": [[16, 6], [16, 9], [16, 4], [16, 13], [16, 1], [16, 10], [16, 5], [16, 2]]},
                {"model_name": "/data/hf/Qwen/Qwen2-0.5B-Instruct",
                 "target_heads": [[9, 10], [9, 7], [9, 13], [9, 8], [9, 12], [9, 11], [9, 9], [9, 3]]},
                {"model_name": "/data/hf/Qwen/Qwen2-0.5B-Instruct",
                 "target_heads": [[9, 10], [9, 13], [9, 7], [9, 8], [9, 12], [9, 11], [9, 9], [9, 0]]},
            ]

        for item in heads_config:
            if item['model_name'] == self.model_name:
                return item['target_heads']

        if hasattr(self, "_ehpc_heads") and self._ehpc_heads is not None:
            return self._ehpc_heads

        configs = ["4k"]  # ["4k", "8k", "16k", "32k"]
        langs = ["en"]  # ["en", "es", "de", "zh", "vi", "ar", "hi"]

        evidence_sum = None
        for config in configs:
            for lang in langs:
                ds = load_dataset("/data/hf/ameyhengle/Multilingual-Needle-in-a-Haystack",
                                  config, split=lang)
                cnt = 0
                for sample in tqdm(ds):
                    cnt += 1
                    if cnt < 11:
                        continue
                    if cnt >= max_samples:
                        break

                    prompt = sample.get("prompt", "")
                    ans = sample.get("answer_sentence", "")
                    start = sample.get("answer_start_index", 0)

                    self.model.eval()
                    with torch.no_grad():
                        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                        input_ids = inputs["input_ids"].to(self.device)

                        if input_ids.size(-1) > self.max_position_embeddings:
                            input_ids = input_ids[:, -self.max_position_embeddings:]

                        outputs = self.model(
                            input_ids=input_ids,
                            output_attentions=True,
                            return_dict=True,
                        )
                    attentions = outputs.attentions  # 24 (1,14,len,len)

                    ans_ids = self.tokenizer(ans, return_tensors="pt", add_special_tokens=False)["input_ids"][
                        0].tolist()
                    needle_range = list(range(start, start + len(ans_ids)))
                    last_idx = input_ids.size(-1) - 1

                    L = len(attentions)  # layer_num
                    H = attentions[0].size(1)  # head_num

                    if evidence_sum is None:
                        evidence_sum = torch.zeros((L, H), dtype=torch.float32, device='cpu')

                    for l in range(L):
                        attn = attentions[l][0]
                        for h in range(H):
                            score = attn[h][last_idx][needle_range].sum()
                            evidence_sum[l, h] += score.detach().cpu()

                    del outputs, attentions
        
        layer_scores, _ = torch.max(evidence_sum, dim=1)
        best_layer = int(torch.argmax(layer_scores).item())
        k = min(8, evidence_sum.size(1))
        _, top_indices = torch.topk(evidence_sum[best_layer], k)
        heads = [(best_layer, int(h.item())) for h in top_indices]
        self._ehpc_heads = heads

        entry = {
            "model_name": self.model_name,
            "target_heads": heads,
            "samples": max_samples,
        }
        with open(config_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"saved to config file: {config_path}")

        return heads

    def _compute_ehpc_attention(
            self,
            attentions: List[Tensor],
            target_heads: List[Tuple[int, int]],
            last_n=1,  # last token
            agg_method='mean'  # mean max
    ) -> Tensor:
        attn = None
        if target_heads:  # 如果指定 heads
            for l, h in target_heads:
                mat = attentions[l][0][h]
                vec = mat[-1:]  # 只使用 last token
                attn = vec if attn is None else attn + vec
        else:
            for l in range(len(attentions)):  # layer
                for h in range(attentions[l].size(1)):  # heads
                    mat = attentions[l][0][h]
                    if last_n == 1:
                        vec = mat[-last_n:]  # (last_n, chunk_size)
                    else:
                        if agg_method == 'mean':
                            vec = torch.mean(mat[-last_n:], dim=0)
                        elif agg_method == 'max':
                            vec = torch.max(mat[-last_n:], dim=0)[1]  # indices, values
                    attn = vec if attn is None else attn + vec

        return attn.detach().cpu()  # (chunk_size)

    def _get_rollout_score(self, chunk_text: str, suffix: str, chunk_size: int) -> Tensor:
        """
        Calculate the rollout score for a given chunk and suffix.
        Optimized for memory usage by clearing intermediate tensors.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(chunk_text + suffix, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(self.device)
            current_suffix_len = input_ids.size(-1) - chunk_size
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True,
            )
        attentions = outputs.attentions
        # Shape: [H, Suffix, Chunk] -> [Chunk]
        score = torch.mean(torch.mean(attentions[-1][0, :, -current_suffix_len:, :chunk_size], dim=0),
                           dim=0).detach().cpu()

        L_N = len(attentions)
        # Top-down accumulation: Iterate from the second to last layer down to the first layer
        for li in range(L_N - 2, -1, -1):  # rollout: 0-n-2, init: n-1
            # Shape: [H, Chunk, Chunk] -> [Chunk, Chunk]
            current_layer_attention = torch.mean(attentions[li][0, :, :chunk_size, :chunk_size], dim=0).detach().cpu()
            score = score @ current_layer_attention

        del outputs
        del attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return score

    def get_layerwise_rollout_scores(self, chunk_text: str, suffix: str, chunk_size: int) -> List[Tensor]:
        """
        Calculate rollout scores for a given chunk and suffix, initializing from each layer.
        Returns a list of scores, where index i corresponds to rollout starting from layer i.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(chunk_text + suffix, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(self.device)
            current_suffix_len = input_ids.size(-1) - chunk_size
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True,
            )
        attentions = outputs.attentions
        L_N = len(attentions)

        # Precompute Attention(Chunk->Chunk) for all layers
        # Shape: [Chunk, Chunk]
        intra_attentions = []
        for i in range(L_N):
            intra_attentions.append(
                torch.mean(attentions[i][0, :, :chunk_size, :chunk_size], dim=0).detach().cpu()
            )

        # Precompute Init(Suffix->Chunk) for all layers
        # Shape: [Chunk]
        init_scores = []
        for i in range(L_N):
            init_scores.append(
                torch.mean(torch.mean(attentions[i][0, :, -current_suffix_len:, :chunk_size], dim=0),
                           dim=0).detach().cpu()
            )

        layer_rollouts = []

        # Chain represents product of AC_{j-1} @ ... @ AC_0
        # Initialize as Identity matrix of size [Chunk, Chunk]
        chain = torch.eye(chunk_size, device='cpu').to(init_scores[0].dtype)

        for i in range(L_N):
            # Calculate Rollout_i
            # score = Init_i @ chain
            score = init_scores[i] @ chain
            layer_rollouts.append(score)

            # Update chain for next layer
            # chain = AC_i @ chain
            chain = intra_attentions[i] @ chain

        # Clean up
        del outputs
        del attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return layer_rollouts, init_scores

    def get_attn(
            self, context: str = "",
            input_ids: Tensor = None,
            return_attn: bool = True,
            target_heads: Optional[List[Tuple[int, int]]] = None,
            last_n=1,
            agg_method='mean',  # mean of token
    ) -> Tuple:
        """
        Compute perplexity (PPL) for each token. Optionally return attention scores.
        
        Args:
            context: Input text.
            input_ids: Optional pre-encoded input IDs.
            return_attn: Whether to return attention sum.
        Returns:
            Tuple of (ppl, input_ids, attention_mask, [attn_sum])
        """
        with torch.inference_mode():
            if input_ids is None:
                inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"].to(self.device)
            else:
                input_ids = input_ids.to(self.device)
            self.model.eval()
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=return_attn,
                return_dict=True
            )
            attn_vec = self._compute_ehpc_attention(outputs.attentions, target_heads,
                                                    last_n=last_n, agg_method=agg_method)
            return input_ids, attn_vec

    def _get_token_length(self, text: str, add_special_tokens: bool = False) -> int:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"]
        return int(ids.size(-1))

    def _chunk_context(self, origin_text: str, max_token_len: int = 1024,
                       chunk_end_tokens: List[str] = [".", "\n"]) -> List[str]:
        origin_tokens = self.tokenizer.tokenize(origin_text)
        n = len(origin_tokens)
        if n <= max_token_len:
            return [origin_text]
        origin_list = []
        st = 0

        while st < n:

            if st + max_token_len > n - 1:
                chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st:n])
                origin_list.append(chunk)
                break
            else:
                ed = st + max_token_len
                for j in range(0, ed - st):
                    if origin_tokens[ed - j] in chunk_end_tokens:
                        ed = ed - j
                        break
                chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st: ed + 1])
                origin_list.append(chunk)
                st = ed + 1
        return origin_list

    def _preserve_punctuation_mask(self, input_ids: Tensor, device: str) -> Tensor:
        """
        Return a boolean mask indicating which tokens are punctuation/special and should be preserved.
        """
        ids = input_ids[0].cpu().numpy()
        decoded_tokens = [self.tokenizer.decode([id_]) for id_ in ids]
        preserve = []
        punct_pattern = re.compile(r'^\s*[^\w\s]+\s*$')
        for token in decoded_tokens:
            is_punct = bool(punct_pattern.match(token))
            is_special = token in ["<s>", "</s>", "[CLS]", "[SEP]", "<pad>"]
            preserve.append(is_punct or is_special)
        return torch.tensor(preserve, dtype=torch.bool, device=device)

    def direct_compress(
            self,
            attn: Tensor,  # score (attn ppl or combine)
            input_ids: Tensor,
            compress_ratio: float,
            preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor]:

        score = attn.detach().cpu().view(-1)
        total_tokens = score.numel()
        k = int(total_tokens * compress_ratio)
        k = max(1, k)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, "cpu").float()
            score = score + 1e5 * punct_mask

        _, indices = torch.topk(score, k=k, largest=True)
        sorted_indices = torch.sort(indices)[0].to(input_ids.device)

        selected_input_ids = input_ids[:, sorted_indices]

        return selected_input_ids, sorted_indices

    def get_decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def compress(
            self,
            context: str,
            question: str,
            compress_ratio: float = 0.5,
            target_token: int = None,
            method: str = "attn_ppl",  # "ehpc_attn"  'p-contrast', 'p-contrast-qa':
            preserve_punct: bool = False,
            return_info: bool = True,
            rollout_m=1,
            contrast_alpha=1,
            chunk_size=1024,
    ) -> Union[str, Dict[str, any]]:
        """
        Compression interface supporting multiple strategies.
        
        Args:
            context: Input text.
            compress_ratio: Compression ratio (0 ~ 1).
            method: "ppl", "attn_ppl", "dynamic_ppl", "dynamic_attn_ppl", and "dynamic_attn_ppl_wosucce"
            fusion: "additive" or "multiplicative".
            alpha: Weight for attention in additive fusion.
            dyn_time: Number of dynamic iterations. If None, auto-calculate.
            preserve_punct: Whether to preserve punctuation and special tokens.
            return_info: If True, return dict with details; else return string.
        """

        start_time = time.time()
        seq_len = self._get_token_length(context, add_special_tokens=False)

        # NOTE add dynamic budget controller
        if target_token is not None:
            compress_ratio = target_token / seq_len
        if compress_ratio < 0 or compress_ratio >= 1:
            print(f"target {target_token} origin_len {seq_len}")
            compress_ratio = 0.99
        assert 0 <= compress_ratio < 1, "compress_ratio must be in [0, 1)"

        def _compress_one_chunk(
                chunk_text: str, question=None, compress_ratio=0.99
        ) -> Tuple[Tensor, Tensor, Tensor]:

            self.method = method
            if method == "ehpc":
                heads = self.identify_evaluator_heads()
                chunk_input_ids, score = self.get_attn(chunk_text, return_attn=True, target_heads=heads, last_n=1)
                # print(chunk_input_ids.size(), score.size())  # (1, len) (1, len)

            elif method == 'kvzip':
                repeat_prompt = f" Repeat the previous context: {chunk_text} "

                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)
                chunk_len = int(chunk_input_ids.size(1))

                input_ids_pos, attn_pos = self.get_attn(chunk_text + repeat_prompt, return_attn=True,
                                                        last_n=chunk_len, agg_method='max')
                score = torch.squeeze(attn_pos)[:chunk_len]

            elif method == 'p-contrast':
                pos_suffix = " pay more attention to key information."
                neg_suffix = " pay attention to redundant information."

                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)

                input_ids_pos, attn_pos = self.get_attn(chunk_text + pos_suffix, return_attn=True)
                input_ids_neg, attn_neg = self.get_attn(chunk_text + neg_suffix, return_attn=True)

                chunk_len = int(chunk_input_ids.size(1))

                attn_pos_chunk = torch.squeeze(attn_pos)[:chunk_len]  # default last token
                attn_neg_chunk = torch.squeeze(attn_neg)[:chunk_len]

                alpha = self.contrast_alpha
                score = attn_pos_chunk - alpha * attn_neg_chunk

            elif method == 'p-contrast-qa':
                """
                mean suffix, mean head, last_layer, alpha 0.8
                """
                pos_suffix = f" Pay more attention to key information about: {question} "
                pos_suffix_len = len(self.tokenizer(pos_suffix).input_ids)
                neg_suffix = " pay attention to redundant information. "
                neg_suffix_len = len(self.tokenizer(neg_suffix).input_ids)

                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)

                input_ids_pos, attn_pos = self.get_attn(
                    chunk_text + pos_suffix, return_attn=True, last_n=pos_suffix_len,  # NOTE
                )
                input_ids_neg, attn_neg = self.get_attn(
                    chunk_text + neg_suffix, return_attn=True, last_n=neg_suffix_len,
                )
                chunk_len = int(chunk_input_ids.size(1))
                attn_pos_chunk = attn_pos[: chunk_len]
                attn_neg_chunk = attn_neg[: chunk_len]
                alpha = self.contrast_alpha
                score = attn_pos_chunk - alpha * attn_neg_chunk

            elif method == 'attn-qa':
                """
                mean suffix, mean head, last_layer
                """
                pos_suffix = f" Pay more attention to key information about: {question} "
                pos_suffix_len = len(self.tokenizer(pos_suffix).input_ids)

                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)

                input_ids_pos, attn_pos = self.get_attn(
                    chunk_text + pos_suffix, return_attn=True, last_n=pos_suffix_len,  # NOTE
                )

                chunk_len = int(chunk_input_ids.size(1))
                score = attn_pos[: chunk_len]

            elif method in ['rollout-qa', 'rollout']:

                if method == 'rollout-qa':
                    pos_suffix = f" Pay more attention to key information about: {question}"
                else:
                    pos_suffix = " Pay attention to redundant information."

                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)
                chunk_size = chunk_input_ids.size(-1)

                score = self._get_rollout_score(chunk_text, pos_suffix, chunk_size)

            elif method in ['contrast-rollout-qa']:  # equal to contrast over rollout
                #  NOTE Rollout(Context, P+) - Rollout(Context, P-)
                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)
                chunk_size = chunk_input_ids.size(-1)

                pos_suffix = f" Pay more attention to key information about: {question} "
                neg_suffix = " pay attention to redundant information. "

                score_pos = self._get_rollout_score(chunk_text, pos_suffix, chunk_size)
                score_neg = self._get_rollout_score(chunk_text, neg_suffix, chunk_size)
                score = score_pos - self.contrast_alpha * score_neg

            elif method in ['rollout-contrast-qa']:  #   contrast last layer then rollout
                # TODO 优化计算   Attn( Context||P-||P+ ) -> Rollout
                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)
                chunk_size = chunk_input_ids.size(-1)

                pos_suffix = f" Pay more attention to key information about: {question} "
                neg_suffix = " pay attention to redundant information. \n"
                pos_suffix_size = len(self.tokenizer(pos_suffix).input_ids)

                self.model.eval()
                with torch.no_grad():
                    inputs = self.tokenizer(chunk_text + neg_suffix + pos_suffix, return_tensors="pt",
                                            add_special_tokens=False)
                    input_ids = inputs["input_ids"].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )
                attentions = outputs.attentions
                m_attentions = torch.mean(torch.stack(attentions[rollout_m:], dim=0), dim=0)

                # Shape: [H, Suffix, Chunk] -> [Chunk]
                last_layer_pos_attn = torch.mean(
                    torch.mean(m_attentions[0, :, -pos_suffix_size:, :chunk_size], dim=0),
                    dim=0).detach().cpu()
                last_layer_neg_attn = torch.mean(
                    torch.mean(m_attentions[0, :, chunk_size:-pos_suffix_size, :chunk_size], dim=0),
                    dim=0).detach().cpu()

                score = last_layer_pos_attn - contrast_alpha * last_layer_neg_attn
                score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))

                if rollout_m > 0:
                    for li in range(rollout_m - 1, -1, -1):  # [0, ..., m-1] [m, ..., n-1]
                        current_layer_attention = torch.mean(attentions[li][0, :, :chunk_size, :chunk_size],
                                                             dim=0).detach().cpu()
                        # score = score @ (current_layer_attention + current_layer_attention.T)
                        score = score @ current_layer_attention

                # score_final = score.cpu().numpy()

                del outputs
                del attentions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                raise ValueError(f"Unknown method {method}")

            selected_input_ids, sorted_indices = self.direct_compress(
                score, chunk_input_ids, compress_ratio, preserve_punct
            )
            return selected_input_ids, chunk_input_ids, torch.squeeze(score)

        if seq_len > chunk_size:  # NOTE
            chunks = self._chunk_context(context, max_token_len=chunk_size, chunk_end_tokens=[".", "\n"])
        else:
            chunks = [context]

        compressed_texts, original_tokens, scores = [], [], []
        for ch in chunks:
            select_ids, raw_ids, score = _compress_one_chunk(ch, question=question, compress_ratio=compress_ratio)

            compressed_texts.append(self.get_decode(select_ids[0].tolist()))

            original_tokens.extend(raw_ids[0].tolist())
            scores.append(score.tolist())
            assert len(score.tolist()) == len(raw_ids[0].tolist())

        decoded_text = "\n\n".join(compressed_texts)
        actual_ratio = self._get_token_length(decoded_text) / seq_len

        result = {
            "method": method,
            "compressed_prompt": decoded_text,
            "actual_ratio": actual_ratio,
            "original_tokens": original_tokens,
            "scores": scores,
            "processing_time": round(time.time() - start_time, 3),
        }

        if return_info:
            return result
        else:
            return decoded_text


def analysis_main():
    INPUT_WO_CONTEXT = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation. Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation. \n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly. \nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答，现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words. \nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words. \nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words. \nThe following are given passages. \nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report. \nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences. \nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions. \n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples. \n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples. \n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n{input}",
        "passage_count": "How many non-repeating paragraphs are there in total? The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from. \nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{input} Next line of code:\n"
    }
    model_name = '/data/hf/Qwen/Qwen2.5-7B-Instruct'
    compressor = EHPCPromptCompressor(model_name=model_name,
                                      device_map="cuda" if torch.cuda.is_available() else "cpu")

    # context = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. " * 50
    # question = "What is AI?"

    task = 'narrativeqa'
    orig_data = load_dataset(f"/data/hf/THUDM/LongBench", task, split="test")

    json_obj = orig_data[0]
    context = json_obj.get('context', "")
    question = INPUT_WO_CONTEXT[task].format(**json_obj)  # 无上下文

    chunks = compressor._chunk_context(context)
    chunk = chunks[0]  # Analyze first chunk

    print("Running Contrast Ablation Analysis...")
    Contrast_Ablation(chunk, question, compressor)

    print("Running Rollout Ablation Analysis...")
    Rollout_Ablation(chunk, question, compressor)

    print("Running Pure Contrast Ablation Analysis...")
    Pure_Contrast_Ablation(chunk, question, compressor)


def Contrast_Ablation(chunk, query, compressor):
    """
    Analyze the contribution of attention calculation layers during compression.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Setup suffixes
    pos_suffix = f" Pay more attention to key information about: {query} "
    neg_suffix = " pay attention to redundant information. "

    chunk_inputs = compressor.tokenizer(chunk, return_tensors="pt", add_special_tokens=False)
    chunk_size = chunk_inputs["input_ids"].size(-1)

    # Get layerwise rollouts
    # This assumes get_layerwise_rollout_scores is implemented in compressor
    rollout_pos_list, _ = compressor.get_layerwise_rollout_scores(chunk, pos_suffix, chunk_size)
    rollout_neg_list, _ = compressor.get_layerwise_rollout_scores(chunk, neg_suffix, chunk_size)

    alpha = compressor.contrast_alpha
    contrast_list = [p - alpha * n for p, n in zip(rollout_pos_list, rollout_neg_list)]

    # Define targets
    # Calculate Mean Contrast Score across all layers
    # mean_contrast = torch.mean(torch.stack(contrast_list), dim=0)
    mean_contrast = contrast_list[-1]
    final_a = rollout_pos_list[-1]

    # Calculate similarities
    k = 100
    k = min(k, chunk_size)

    def get_topk_indices(score, k):
        return set(torch.topk(score, k).indices.tolist())

    mean_contrast_topk = get_topk_indices(mean_contrast, k)
    final_a_topk = get_topk_indices(final_a, k)

    sim_ca_scores = []
    sim_a_scores = []
    layers = range(len(contrast_list))

    for i in layers:
        # Similarity with Mean Contrast
        current_ca_topk = get_topk_indices(contrast_list[i], k)
        overlap_ca = len(current_ca_topk.intersection(mean_contrast_topk))
        sim_ca_scores.append(overlap_ca / k)

        # Similarity with Final A (using Pos score of layer i vs Final A)
        current_pos_topk = get_topk_indices(rollout_pos_list[i], k)
        overlap_a = len(current_pos_topk.intersection(final_a_topk))
        sim_a_scores.append(overlap_a / k)

    # Plotting
    plt.figure(figsize=(10, 8))
    # Scatter plot
    plt.scatter(sim_ca_scores, sim_a_scores, c=layers, cmap='viridis', s=100)
    plt.colorbar(label='Layer ID')

    # Annotate layer IDs
    for i, (x, y) in enumerate(zip(sim_ca_scores, sim_a_scores)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(2, 5), ha='center')

    plt.xlabel('Similarity with Mean Contrastive Attention (All Layers)')
    plt.ylabel('Similarity with Final Standard Attention (A)')
    plt.title('Layer-wise Contribution Analysis (Rollout)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.plot([0, 1], [0, 1], ls="--", c=".3")

    save_dir = '../exp_pics/ehpc'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'contrast-rollout_alpha{compressor.contrast_alpha}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def Rollout_Ablation(chunk, query, compressor):
    """
    Analyze the contribution of rollout depth.
    Compares Rollout starting from layer k (R_k) with:
    1. Full Rollout (starting from layer N-1)
    2. Original Attention (Last layer raw attention)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # We only care about Positive Suffix for standard Rollout analysis?
    # Or maybe we should check the standard rollout method which usually just has one suffix or "pay attention to key info".
    # Since we are comparing "Rollout" performance/similarity, let's use the positive suffix.

    pos_suffix = f" Pay more attention to key information about: {query} "

    chunk_inputs = compressor.tokenizer(chunk, return_tensors="pt", add_special_tokens=False)
    chunk_size = chunk_inputs["input_ids"].size(-1)

    # Get layerwise rollouts and init scores (raw attention)
    rollout_list, init_scores_list = compressor.get_layerwise_rollout_scores(chunk, pos_suffix, chunk_size)

    # Define Targets
    # Target 1: Full Rollout (Rollout from last layer N-1)
    # The list is indexed 0 to N-1. So rollout_list[-1] is R_{N-1}.
    target_full_rollout = rollout_list[-1]

    # Target 2: Original Attention (Last layer raw attention)
    # init_scores_list[-1] is A_{N-1} (Attention of last layer w.r.t suffix)
    target_original_attn = init_scores_list[-1]

    k = 100
    k = min(k, chunk_size)

    def get_topk_indices(score, k):
        return set(torch.topk(score, k).indices.tolist())

    full_rollout_topk = get_topk_indices(target_full_rollout, k)
    original_attn_topk = get_topk_indices(target_original_attn, k)

    sim_full_rollout_scores = []
    sim_original_attn_scores = []
    layers = range(len(rollout_list))

    for i in layers:
        # Calculate R_k (Rollout starting from layer i)
        r_k = rollout_list[i]
        r_k_topk = get_topk_indices(r_k, k)

        # Similarity with Full Rollout
        overlap_full = len(r_k_topk.intersection(full_rollout_topk))
        sim_full_rollout_scores.append(overlap_full / k)

        # Similarity with Original Attn
        overlap_orig = len(r_k_topk.intersection(original_attn_topk))
        sim_original_attn_scores.append(overlap_orig / k)

    # Plotting
    plt.figure(figsize=(10, 8))
    # Scatter plot
    plt.scatter(sim_full_rollout_scores, sim_original_attn_scores, c=layers, cmap='viridis', s=100)
    plt.colorbar(label='Start Layer ID (k)')

    # Annotate layer IDs
    for i, (x, y) in enumerate(zip(sim_full_rollout_scores, sim_original_attn_scores)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('Similarity with Full Rollout (Start from N-1)')
    plt.ylabel('Similarity with Original Attention (Last Layer Raw)')
    plt.title('Rollout Ablation: Effect of Starting Layer')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    # Add diagonal line for reference (optional)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")

    save_dir = '../exp_pics/ehpc'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'rollout_alpha{compressor.contrast_alpha}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def Pure_Contrast_Ablation(chunk, query, compressor):
    """
    Analyze the contribution of pure attention layers (without rollout).
    Compares Layer-wise Contrastive Attention with:
    1. Final Layer Contrastive Attention
    2. Final Layer Standard Attention (Pos only)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pos_suffix = f" Pay more attention to key information about: {query} "
    neg_suffix = " pay attention to redundant information. "

    chunk_inputs = compressor.tokenizer(chunk, return_tensors="pt", add_special_tokens=False)
    chunk_size = chunk_inputs["input_ids"].size(-1)

    # Get layerwise init scores (raw attention, no rollout)
    _, pos_init_list = compressor.get_layerwise_rollout_scores(chunk, pos_suffix, chunk_size)
    _, neg_init_list = compressor.get_layerwise_rollout_scores(chunk, neg_suffix, chunk_size)

    alpha = compressor.contrast_alpha

    # Calculate Contrast Score for each layer
    # Score = Pos - alpha * Neg
    contrast_list = [p - alpha * n for p, n in zip(pos_init_list, neg_init_list)]

    # Targets (Last Layer)
    final_contrast = contrast_list[-1]
    final_attn = pos_init_list[-1]

    k = 100
    k = min(k, chunk_size)

    def get_topk_indices(score, k):
        return set(torch.topk(score, k).indices.tolist())

    final_contrast_topk = get_topk_indices(final_contrast, k)
    final_attn_topk = get_topk_indices(final_attn, k)

    sim_contrast_scores = []
    sim_attn_scores = []
    layers = range(len(contrast_list))

    for i in layers:
        # Layer i Contrast Score
        current_contrast = contrast_list[i]
        current_contrast_topk = get_topk_indices(current_contrast, k)

        # Similarity with Final Contrast
        overlap_contrast = len(current_contrast_topk.intersection(final_contrast_topk))
        sim_contrast_scores.append(overlap_contrast / k)

        # Similarity with Final Attention
        # Note: We compare Layer i Contrast with Final Attention to see if it behaves like attention
        # Or should we compare Layer i Attention with Final Attention?
        # Based on previous logic "Analyze contribution of contrastive attention", we check how Layer i Contrast relates to final targets.
        overlap_attn = len(current_contrast_topk.intersection(final_attn_topk))
        sim_attn_scores.append(overlap_attn / k)

    # Plotting
    plt.figure(figsize=(10, 8))
    # Scatter plot
    plt.scatter(sim_contrast_scores, sim_attn_scores, c=layers, cmap='viridis', s=100)
    plt.colorbar(label='Layer ID')

    # Annotate layer IDs
    for i, (x, y) in enumerate(zip(sim_contrast_scores, sim_attn_scores)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('Similarity with Final Contrastive Attention (Last Layer)')
    plt.ylabel('Similarity with Final Standard Attention (Last Layer)')
    plt.title('Pure Contrast Ablation: Layer-wise Contribution (No Rollout)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.plot([0, 1], [0, 1], ls="--", c=".3")

    save_dir = '../exp_pics/ehpc'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'contrast_alpha{compressor.contrast_alpha}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == '__main__':
    
    model_list = [
        '/data/hf/Qwen/Qwen2.5-7B-Instruct',
        # '/data/hf/Qwen/Qwen2.5-32B-Instruct',
        # '/data/hf/meta-llama/Llama-3.1-8B-Instruct',
        # "/data/hf/codellama/CodeLlama-7b-hf",  # 'CodeLlama-7B',
        # "/data/hf/microsoft/Phi-3.5-mini-instruct",
    ]
    for model in model_list:
        compressor = EHPCPromptCompressor(model, device_map="cuda:0", )
        compressor.identify_evaluator_heads(max_samples=500)
