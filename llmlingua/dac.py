import json

import torch
import numpy as np
from typing import List, Optional, Dict, Union, Tuple

# from brotlicffi import Compressor
from torch import Tensor
import time
import re
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

CHUNK_SIZE = 1024


class DACPromptCompressor:
    """
    A comprehensive prompt compressor that supports compression 
    using entropy and Attention scores with additive/multiplicative fusion.
    """

    def __init__(self,
                 model_name,
                 device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_config: dict = {}, ):
        """
        Initialize the compressor with model and tokenizer.
        
        Args:
            model_name: Pretrained language model.
            device_map: Device to run compression model.
        """
        self.model_name = model_name
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
        MODEL_CLASS = (
            AutoModelForTokenClassification
            if any("ForTokenClassification" in ar for ar in config.architectures)
            else AutoModelForCausalLM
        )
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        model = MODEL_CLASS.from_pretrained(
            model_name,
            torch_dtype=model_config.pop(
                "torch_dtype", "auto" if device_map == "cuda" else torch.float32
            ),
            device_map=device_map,
            config=config,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
            **model_config,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        self.max_position_embeddings = config.max_position_embeddings
        self.model_config = config

    def _get_token_length(self, text: str, add_special_tokens: bool = False) -> int:
        return len(self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids)

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

    def normalize(self, tensor: Tensor) -> Tensor:
        """
        Min-max normalize tensor to [0, 1].
        """
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(tensor)

    def get_ppl(self, context: str = "", input_ids: Tensor = None, attention_mask: Tensor = None,
                return_attn: bool = False,
                return_ppl: bool = True,
                ) -> Tuple:
        """
        Compute perplexity (PPL) for each token. Optionally return attention scores.
        
        Args:
            context: Input text.
            input_ids: Optional pre-encoded input IDs.
            attention_mask: Optional attention mask.
            return_attn: Whether to return attention sum.
        
        Returns:
            Tuple of (ppl, input_ids, attention_mask, [attn_sum])
        """
        with torch.inference_mode():
            if input_ids is None:
                inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
            else:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                attention_mask=attention_mask,
                output_attentions=return_attn,
            )
            token_losses = None
            if return_ppl:
                logits = outputs.logits.detach().cpu()

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous().detach().cpu()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                token_losses = token_losses.view(shift_labels.size())
                del shift_logits, shift_labels

            if return_attn:
                column_sum_cpu = None
                total_heads = 0
                for layer in range(len(outputs.attentions)):
                    attn_layer = outputs.attentions[layer][0]  # (heads, seq(row), seq(column))
                    heads = attn_layer.shape[0]
                    # all token mean
                    # layer_col = attn_layer.sum(dim=0).sum(dim=0)  # (seq,)
                    # last token
                    layer_col = attn_layer.sum(dim=0)[-1]  # (seq,)
                    layer_col_cpu = layer_col.detach().cpu()
                    column_sum_cpu = (
                        layer_col_cpu if column_sum_cpu is None else (column_sum_cpu + layer_col_cpu)
                    )
                    total_heads += heads
                    del attn_layer, layer_col, layer_col_cpu
                column_sum_cpu = (column_sum_cpu / max(total_heads, 1))[1:]
                return token_losses, input_ids, column_sum_cpu
            else:
                return token_losses, input_ids

    def _fuse_attn_ppl_additive(self, ppl: Tensor, attn: Tensor, alpha: float = 0.8) -> Tensor:
        """
        Fuse PPL and Attention using additive rule: score = alpha * attn + (1-alpha) * ppl
        
        Args:
            ppl: Perplexity scores.
            attn: Attention scores.
            alpha: Weight for attention.
        Returns:
            Fused score.
        """
        ppl_norm = self.normalize(ppl)
        attn_norm = self.normalize(attn)

        score = alpha * attn_norm + (1 - alpha) * ppl_norm
        return score

    def _fuse_attn_ppl_multiplicative(self, ppl: Tensor, attn: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Fuse PPL and Attention using multiplicative rule: score = attn * (1/ppl)
        
        Args:
            ppl: Perplexity scores.
            attn: Attention scores.
        Returns:
            Fused score.
        """
        score = new_ppl = torch.mul(ppl, attn)
        return score

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
            ppl: Tensor = None,
            input_ids: Tensor = None,
            compress_ratio: float = None,
            attn_sum: Tensor = None,
            fusion: str = "additive",
            alpha: float = 0.8,
            preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress using fused PPL and Attention scores.
        """

        if attn_sum is None:
            score = ppl.detach().cpu().view(-1)
        elif ppl is None:
            score = self.normalize(attn_sum.detach().cpu())
        else:
            ppl_cpu = ppl.detach().cpu()
            attn_cpu = attn_sum.detach().cpu()
            if fusion == "additive":
                score = self._fuse_attn_ppl_additive(ppl_cpu, attn_cpu, alpha)
            elif fusion == "multiplicative":
                score = self._fuse_attn_ppl_multiplicative(ppl_cpu, attn_cpu)
            else:
                raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        score = score.view(-1)
        total_tokens = score.numel()
        k = int(total_tokens * compress_ratio)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, "cpu").float()
            score = score + 1e5 * punct_mask  # 给标点加一个极大分数

        _, indices = torch.topk(score, k=k, largest=True)
        sorted_indices = torch.sort(indices)[0].to(input_ids.device)

        selected_ids = input_ids[:, sorted_indices]
        return selected_ids, score

    def direct_compress_attn_wosucce(
            self,
            ppl: Tensor,
            input_ids: Tensor,
            attention_mask: Tensor,
            compress_ratio: float,
            attn_sum: Tensor,
            fusion: str = "additive",
            alpha: float = 0.8,
            preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress using fused PPL and Attention scores.
        """
        if compress_ratio <= 0:
            return input_ids, attention_mask, torch.arange(input_ids.size(1))

        ppl_cpu = ppl.detach().cpu()
        attn_cpu = attn_sum.detach().cpu()
        if fusion == "additive":
            score = self._fuse_attn_ppl_additive(ppl_cpu, attn_cpu, alpha)
        elif fusion == "multiplicative":
            score = self._fuse_attn_ppl_multiplicative(ppl_cpu, attn_cpu)
        else:
            raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        score = score.view(-1)
        total_tokens = score.numel()
        k = int(total_tokens * (1 - compress_ratio))
        k = max(1, k)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, "cpu").float()
            score = score - 1e5 * punct_mask

        _, indices = torch.topk(score, k=k, largest=True)
        sorted_indices = torch.sort(indices)[0].to(self.device)

        all_values = torch.arange(ppl_cpu.numel())
        del_indices = all_values[~torch.isin(all_values, sorted_indices.cpu())]
        differences = del_indices[1:] - del_indices[:-1]
        mask = torch.ones_like(del_indices, dtype=torch.bool)
        mask[1:] = differences == 1
        mask[0] = False

        for i in range(1, len(mask)):
            if mask[i - 1]:
                mask[i] = False
        filtered_indices = del_indices[mask]
        all_indices, _ = torch.sort(torch.cat((indices.cpu(), filtered_indices)))
        all_indices = all_indices.to(input_ids.device)

        selected_input_ids = input_ids[:, all_indices]
        selected_attention_mask = attention_mask[:, all_indices]
        new_attn_sum = attn_cpu[all_indices.cpu()]

        return selected_input_ids, selected_attention_mask, sorted_indices, new_attn_sum

    def get_decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def compress(
            self,
            context: str,
            compress_ratio: float = 0.5,
            target_token=None,
            method: str = "attn_ppl",
            fusion: str = "additive",
            alpha: float = 0.8,
            dyn_time: Optional[int] = None,
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
        seq_len = len(self.tokenizer(context).input_ids)

        if target_token is not None:
            compress_ratio = target_token / seq_len
        if compress_ratio < 0 or compress_ratio >= 1:
            print(f"target {target_token} origin_len {seq_len}")
            compress_ratio = 0.99
        assert 0 <= compress_ratio < 1, "compress_ratio must be in [0, 1)"

        if dyn_time is None:
            dyn_time = min(max(1, seq_len // 100), 15)

        def _compress_one_chunk(chunk_text: str) -> Tuple[Tensor, Tensor, List[int]]:
            if method == "ppl":
                ppl, input_ids = self.get_ppl(chunk_text)
                selected_ids, score = self.direct_compress(
                    ppl=ppl, input_ids=input_ids[:, 1:], compress_ratio=compress_ratio,
                    preserve_punct=preserve_punct
                )

            elif method == "attn":
                _, input_ids, attn_sum = self.get_ppl(chunk_text, return_attn=True, return_ppl=False)
                selected_ids, score = self.direct_compress(
                    ppl=None, input_ids=input_ids[:, 1:], compress_ratio=compress_ratio, attn_sum=attn_sum,
                    preserve_punct=preserve_punct
                )

            elif method == "attn_ppl":
                ppl, input_ids, attn_sum = self.get_ppl(chunk_text, return_attn=True)
                selected_ids, score = self.direct_compress(
                    ppl=ppl, input_ids=input_ids[:, 1:], compress_ratio=compress_ratio, attn_sum=attn_sum,
                    fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
                )

            # elif method == "dynamic_ppl":
            #     ppl, input_ids, attention_mask = self.get_ppl(chunk_text)
            #     local_dyn = dyn_time if dyn_time is not None else min(max(1, input_ids.size(1) // 100), 15)
            #     # TODO ?
            #     real_ratio = (1 - (1 - compress_ratio) ** (1.0 / local_dyn))
            #     kept = None
            #     for _ in range(local_dyn):
            #         selected_input_ids, selected_attention_mask, kept_indices = self.direct_compress(
            #             ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, preserve_punct
            #         )
            #         selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
            #         selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
            #         ppl, input_ids, attention_mask = self.get_ppl("", input_ids=selected_input_ids,
            #                                                       attention_mask=selected_attention_mask)
            #         kept = kept_indices.tolist()
            #     return selected_input_ids, selected_attention_mask, kept if kept is not None else []
            # elif method == "dynamic_attn_ppl":
            #     ppl, input_ids, attention_mask, attn_sum = self.get_ppl(chunk_text, return_attn=True)
            #     local_dyn = dyn_time if dyn_time is not None else min(max(1, input_ids.size(1) // 100), 15)
            #     real_ratio = (1 - (1 - compress_ratio) ** (1.0 / local_dyn))
            #     kept = None
            #     for _ in range(local_dyn):
            #         selected_input_ids, selected_attention_mask, kept_indices, attn_sum = self.direct_compress_attn(
            #             ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, attn_sum,
            #             fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
            #         )
            #         selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
            #         selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
            #         ppl, input_ids, attention_mask, attn_sum = self.get_ppl("", input_ids=selected_input_ids,
            #                                                                 attention_mask=selected_attention_mask,
            #                                                                 return_attn=True)
            #         kept = kept_indices.tolist()
            #     return selected_input_ids, selected_attention_mask, kept if kept is not None else []
            # elif method == "dynamic_attn_ppl_wosucce":
            #     ppl, input_ids, attention_mask, attn_sum = self.get_ppl(chunk_text, return_attn=True)
            #     local_dyn = dyn_time if dyn_time is not None else min(max(1, input_ids.size(1) // 100), 15)
            #     real_ratio = (1 - (1 - compress_ratio) ** (1.0 / local_dyn))
            #     kept = None
            #     for _ in range(local_dyn):
            #         selected_input_ids, selected_attention_mask, kept_indices, attn_sum = self.direct_compress_attn_wosucce(
            #             ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, attn_sum,
            #             fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
            #         )
            #         selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
            #         selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
            #         ppl, input_ids, attention_mask, attn_sum = self.get_ppl("", input_ids=selected_input_ids,
            #                                                                 attention_mask=selected_attention_mask,
            #                                                                 return_attn=True)
            #         kept = kept_indices.tolist()
            #     return selected_input_ids, selected_attention_mask, kept if kept is not None else []
            else:
                raise ValueError("Unknown method")

            selected_ids = torch.cat((input_ids[:, :1], selected_ids), dim=1)
            score = torch.concat((torch.tensor([1]), score), dim=0)
            return selected_ids, input_ids, score

        if seq_len > CHUNK_SIZE:
            chunks = self._chunk_context(context, max_token_len=CHUNK_SIZE, chunk_end_tokens=[".", "\n"])
        else:
            chunks = [context]

        compressed_texts, original_tokens, scores = [], [], []
        for ch in chunks:
            select_ids, raw_ids, score = _compress_one_chunk(ch)

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
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # Initialize the compressor (supports Hugging Face models)
    model_name = "/data/hf/Qwen/Qwen2-0.5B-Instruct"
    compressor = DACPromptCompressor(model_name)

    # Long input context (e.g., retrieved documents, conversation history)
    context = """
    Artificial intelligence is a branch of computer science aimed at creating systems capable of performing tasks that typically require human intelligence...
    """

    # Perform compression
    result = compressor.compress(
        context=context,
        compress_ratio=0.9,  # Keep only 10% of tokens (10x compression)
        target_token=None,
        method="dynamic_attn_ppl",  # Compression method
        fusion="additive",  # Fusion strategy
        alpha=0.8,  # Attention weight in additive fusion
        dyn_time=10,  # Number of dynamic iterations
        preserve_punct=False,  # Preserve punctuation and special tokens or not
        return_info=True  # Return detailed info
    )

    # Output results
    print("Compressed text:", result["compressed_text"])
    print("Original tokens:", result["original_tokens"])
    print("Compressed tokens:", result["compressed_tokens"])
    print("Actual compression ratio:", result["actual_ratio"])
