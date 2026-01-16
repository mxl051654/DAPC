"""

@InProceedings{pmlr-v235-achtibat24a,
  title = {{A}ttn{LRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {135--168},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}


基于LRP，将上下文筛选任务构造成基于相关性判别的可解释归因

对于每个chunk，构造query相关性判别任务，
基于LRP基于决策token的token梯度归因


relevance calculate

import torch
from transformers import AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

# modify the Qwen2 module to compute LRP in the backward pass
monkey_patch(modeling_qwen2, verbose=True)

# optional 4bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent overflow in gradients
)

path = '/data/hf/Qwen/Qwen2.5-7B-Instruct'
model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16, quantization_config=quantization_config)

# optional gradient checkpointing to save memory (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(path)

prompt =

# get input embeddings so that we can compute gradients w.r.t. input embeddings
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# inference and get the maximum logit at the last position (we can also explain other tokens)
output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

# NOTE Backward pass (the relevance is initialized with the value of max_logits)
# This initiates the LRP computation through the network
max_logits.backward()  # TODO 设计面向提示压缩（不知道目标回答）

# TODO 构造判别任务
# context , query , judge relevant or not, just answer Yes or Not
# Back forward with Yes token logits

# obtain relevance by computing Gradient (1, 456, 1536) * Input (1,456,1536)
relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]

# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

"""
import jsonlines
from tqdm import tqdm
import torch
from transformers.models.qwen2 import modeling_qwen2

try:
    from lxt.efficient import monkey_patch
except ImportError:
    monkey_patch = None
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

CHUNK_SIZE = 1024


class LRPPromptCompressor:
    """
    A comprehensive prompt compressor that supports compression 
    using entropy and Attention scores with additive/multiplicative fusion.
    """

    def __init__(self, model_name,
                 device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_config: dict = {}, ):

        self.model_name = model_name  # The model used for compress
        self.pos_suffix = model_config.get("pos_suffix", "\n Focus on key information.")
        self.neg_suffix = model_config.get("neg_suffix", "\n Focus on redundant information.")
        self.contrast_alpha = float(model_config.get("contrast_alpha", 0.8))
        self.load_model(model_name, device_map, model_config)

    def load_model(self, model_name, device_map: str = "cuda", model_config: dict = {}):
        print(f"======== Loading {model_name} =========")
        if monkey_patch:
            try:
                monkey_patch(modeling_qwen2, verbose=False)
            except Exception as e:
                print(f"Warning: Failed to apply LRP monkey patch: {e}")

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

        if 'qwen2' in model_name.lower():
            model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_arg,
                device_map=device_map,
                config=config,
                ignore_mismatched_sizes=True,
                # attn_implementation="eager",
                # attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
                **model_config,
            )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        # self.max_position_embeddings = config.max_position_embeddings  # 32768
        self.max_position_embeddings = 3000
        self.model_config = config

    def _get_token_length(self, text: str, add_special_tokens: bool = False) -> int:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"]
        return int(ids.size(-1))

    def _chunk_context(self, origin_text: str, max_token_len: int = CHUNK_SIZE,
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
        """Decode token IDs to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def compress(
            self,
            context: str,
            question: str,
            compress_ratio: float = 0.5,
            target_token: int = None,
            method: str = "attn_ppl",  # "ehpc_attn"  'p-contrast', 'p-contrast-qa':
            preserve_punct: bool = False,
            return_info: bool = True
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
        ) -> Tuple[Tensor, Tensor, List[int]]:

            # Identify "True" token ID
            true_ids = self.tokenizer.encode("True", add_special_tokens=False)
            target_token_id = true_ids[0] if true_ids else self.tokenizer.eos_token_id  # Fallback

            self.method = method
            if method == 'lrp-qa':

                prompt = f"{chunk_text} \n Judge if the context is necessary to answer query ：{question} Return True or False"

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
                input_ids = inputs.input_ids

                # Embeddings and Gradient Setup
                input_embeds = self.model.get_input_embeddings()(input_ids)
                input_embeds.retain_grad()
                input_embeds.requires_grad_(True)

                # Forward Pass
                outputs = self.model(inputs_embeds=input_embeds, use_cache=False)
                target_logit = outputs.logits[0, -1, target_token_id]
                self.model.zero_grad()
                target_logit.backward()

                # Calculate Relevance
                relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
                relevance = relevance / (relevance.abs().max() + 1e-6)

                # Prepare Chunk Inputs for Compression
                chunk_inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_input_ids = chunk_inputs["input_ids"].to(self.device)
                chunk_size = chunk_input_ids.size(-1)

                score = relevance[:chunk_size]
                selected_input_ids, sorted_indices = self.direct_compress(
                    score, chunk_input_ids, compress_ratio, preserve_punct
                )
                return selected_input_ids, chunk_input_ids, score

            if method == 'ablation-qa':
                pass
                # 基于token 消融对比 answer logits 的方法

            else:
                raise ValueError(f"Unknown method {method}")

        if seq_len > CHUNK_SIZE:
            chunks = self._chunk_context(context, max_token_len=CHUNK_SIZE, chunk_end_tokens=[".", "\n"])
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


if __name__ == '__main__':
    pass
