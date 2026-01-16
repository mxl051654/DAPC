import os
import json
import re
import torch
import numpy as np
import tiktoken

from typing import List
from munch import Munch
from sentence_splitter import split_text_into_sentences

import evaluate
class Metric:
    def __init__(self, metric_name):
        self.metric_name = metric_name

    def compute(self, predictions, references):
        """
        计算指定的指标，如准确率。
        :param predictions: 模型的预测结果
        :param references: 真正的标签
        :return: 计算结果（如准确率）
        """
        if self.metric_name == "accuracy":
            return self._compute_accuracy(predictions, references)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def _compute_accuracy(self, predictions, references):
        """
        计算准确率
        :param predictions: 模型预测的标签
        :param references: 真实标签
        :return: 准确率（correct predictions / total samples）
        """
        correct = sum(p == r for p, r in zip(predictions, references))
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}

metric = Metric("accuracy")
# metric = evaluate.load("accuracy")


# from cpc_model.llama import LlamaBiForMNTPandSentEmbeddingsV2
# from cpc_model.mistral import MistralBiForMNTPandSentEmbeddingsV2
from .cpc_model.common import ModelType, build_model
from .cpc_model.model import load_model_and_tokenizer


def preprocess_logits_for_metrics(outputs, labels):
    if isinstance(outputs, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        logits, loss_contrastive, last_hidden_state = outputs
    return logits.argmax(dim=-1), loss_contrastive, last_hidden_state


def compute_metrics(eval_preds):
    outputs, labels = eval_preds
    preds, loss_contrastive, last_hidden_state = outputs

    preds = preds[:, :-1]
    labels = labels[:, 1:]
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    result = metric.compute(predictions=preds, references=labels)

    result['loss_contrastive'] = loss_contrastive.mean()

    return result


@torch.no_grad()
def get_ppl_one_step(model, tokenizer, prefix, suffix):
    inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    all_tokens = prefix_tokens + suffix_tokens

    input_ = {
        'input_ids': torch.LongTensor(all_tokens)[None].cuda(),
        'attention_mask': torch.FloatTensor([1 for _ in all_tokens])[None].cuda(),
    }

    outputs_ = model(**input_)

    logits = torch.log_softmax(outputs_.logits[0][:-1], dim=-1)

    logits_chunk = logits[len(prefix_tokens) - 1:]

    scores = []
    assert len(suffix_tokens) == logits_chunk.shape[0]
    for i, t in enumerate(suffix_tokens):
        scores.append((inv_vocab[t], logits_chunk[i, t].item()))
    return {
        'logprobs': logits_chunk,
        'suffix_logprobs': scores,
    }


def kl_divergence(logp, logq):
    return torch.mean(torch.sum(torch.exp(logp) * (logp - logq), dim=1), dim=0)


def split_text_into_sentences_keep_slashn(text, language):
    parts = text.split('\n')

    new_parts = []
    i = 0
    prev = -1
    N = len(parts)

    while i < N:
        while i < N and parts[i].strip() == '':
            i += 1

        if prev != -1:
            new_parts.append([parts[prev]] + (['\n'] * (i - prev)))
        prev = i
        i = prev + 1

    if i == N:
        new_parts.append([parts[prev]] + (['\n'] * (N - 1 - prev)))

    new_parts = [
        [npp for npp in np if npp]
        for np in new_parts
    ]

    sents_splitted = []
    for new_part in new_parts:
        snippet, *lineterms = new_part
        if snippet == '\n':
            sents_splitted.extend(['\n'])
            continue
        sents = split_text_into_sentences(snippet, language=language)
        sents[-1] = sents[-1] + ''.join(lineterms)
        sents_splitted.extend(sents)

    return sents_splitted


class SentenceEmbeddingType:
    AVG = 1


class SpecTokenType:
    END_OF_SENT = "<end_of_sent>"
    END_OF_QUESTION = "<end_of_question>"


def tokenize_and_clip_segments(
        tokenizer,
        segments: List[str],
        segments_labels: List[int],
        max_seq_len: int,
        end_of_sentence_token: str = None):
    encodings = {
        'text_input_ids': [],
        'text_segment_ids': [],
        'segments': [],
        'segments_labels': [],
        'sentence_input_ids': [],
    }
    for i, (seg, seg_label) in enumerate(zip(segments, segments_labels)):
        inputs = tokenizer.encode(seg, add_special_tokens=False)
        if len(encodings['text_input_ids']) + len(inputs) > max_seq_len:
            break
        encodings['text_input_ids'].extend(inputs)
        encodings['text_segment_ids'].extend([i] * len(inputs))
        encodings['segments'].append(seg)
        encodings['segments_labels'].append(seg_label)
        encodings['sentence_input_ids'].append(inputs)
    return Munch(encodings)


import torch
from torch import nn, Tensor


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def mismatched_sizes_all_gather(tensor: Tensor, group=None, async_op=False, mismatched_axis=0):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor([tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda")
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, mismatched_sizes, group=group, async_op=async_op)
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros((*tensor.shape[:mismatched_axis], max_size, *tensor.shape[mismatched_axis + 1:]),
                         device=tensor.device, dtype=tensor.dtype)
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype) for _ in range(world_size)]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert not tensor_list[rank].narrow(mismatched_axis, sizes[rank],
                                            padded.shape[mismatched_axis] - sizes[rank]).count_nonzero().is_nonzero(), \
            "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled


class SamplePreprocessor:
    def __init__(
            self,
            tokenizer,
            max_context_len: int,
            use_question_as_suffix: bool = False,
            sentence_embedding_type: int = SentenceEmbeddingType.AVG):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.use_question_as_suffix = use_question_as_suffix
        self.sentence_embedding_type = sentence_embedding_type

    def chunkify(self, segments, segments_labels, suffix_question_segments):
        init_length = len(self.tokenizer.encode(' '.join(segments), add_special_tokens=False))
        question_length = len(self.tokenizer.encode(' '.join(suffix_question_segments), add_special_tokens=False))

        N = len(segments)

        num_chunks = 1
        good = False
        while not good:
            buckets = []
            buckets_labels = []
            n_per_chunk = N // num_chunks if N % num_chunks == 0 else (N // num_chunks) + 1
            for i in range(0, N, n_per_chunk):
                buckets.append(segments[i:i + n_per_chunk])
                buckets_labels.append(segments_labels[i:i + n_per_chunk])

            good = True
            for b in buckets:
                chunk = ' '.join(b)
                if len(self.tokenizer.encode(chunk, add_special_tokens=False)) > self.max_context_len:
                    good = False
                    break
            if not good:
                num_chunks += 1

        # Rearrange
        lens = [
            [
                len(self.tokenizer.encode(s, add_special_tokens=False))
                for s in b
            ]
            for b in buckets
        ]

        question_labels = [1 for _ in range(len(suffix_question_segments))]

        new_buckets = []
        new_buckets_labels = []
        cur_bucket = []
        cur_bucket_labels = []
        cur_len = -1
        for b, bsl, bslens in zip(buckets, buckets_labels, lens):
            for s, sl, slen in zip(b, bsl, bslens):
                cur_bucket.append(s)
                cur_bucket_labels.append(sl)

                cur_len += slen + 1
                if cur_len >= init_length / num_chunks:
                    new_buckets.append(cur_bucket)
                    new_buckets_labels.append(cur_bucket_labels)
                    cur_bucket = []
                    cur_bucket_labels = []
                    cur_len = -1
        if cur_bucket:
            new_buckets.append(cur_bucket)
            new_buckets_labels.append(cur_bucket_labels)
            cur_bucket = []
            cur_bucket_labels = []
            cur_len = -1

        buckets = new_buckets
        buckets_labels = new_buckets_labels

        return buckets, buckets_labels

    def _ensure_sents_not_too_long(self, sents):
        shortened_sents = []
        for s in sents:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            if len(tokens) > self.max_context_len:
                chunk_size = self.max_context_len // 20
                n_chunks = int(np.ceil(len(tokens) / chunk_size))
                token_chunks = [tokens[i * chunk_size:i * (chunk_size + 1)] for i in range(n_chunks)]
                smaller_sents = [self.tokenizer.decode(chunk) for chunk in token_chunks]
                shortened_sents.extend(smaller_sents)
            else:
                shortened_sents.append(s)
        return shortened_sents

    def __call__(self, context, question, question_for_suffix=None):
        sents = split_text_into_sentences_keep_slashn(context, language='en')
        sents = [t for t in sents if t.strip()]
        sents = self._ensure_sents_not_too_long(sents)
        sents_question_as_suffix = []
        if self.use_question_as_suffix:
            sents_question_as_suffix = split_text_into_sentences_keep_slashn(question_for_suffix, language='en')
            sents_question_as_suffix = [t for t in sents_question_as_suffix if t.strip()]

        sents_labels = [0 for _ in sents] + [1 for _ in sents_question_as_suffix]

        chunks, chunks_labels = self.chunkify(
            sents + sents_question_as_suffix,
            sents_labels,
            sents_question_as_suffix,
        )

        end_of_sent_token_id = None
        end_of_question_token_id = None

        encodings = [
            tokenize_and_clip_segments(
                tokenizer=self.tokenizer,
                segments=chunk,
                segments_labels=chunk_labels,
                max_seq_len=self.max_context_len,
                sentence_embedding_type=self.sentence_embedding_type,
                end_of_sentence_token=end_of_sent_token_id
            )
            for chunk, chunk_labels in zip(chunks, chunks_labels)
        ]

        encodings_question = self.tokenizer.batch_encode_plus([question], add_special_tokens=False, padding='longest')

        return {
            'context': encodings,
            'question': encodings_question,
        }


# 压缩样本
# 压缩样本
@torch.no_grad()  # 禁用梯度计算，以减少内存使用和加速计算
def compress_sample(model, tokenizer, openai_tokenizer, sample, compression_target_tokens, boost_match_regex):
    # 定义正则匹配函数，用于匹配特定的文本模式
    def re_checker(text):
        if boost_match_regex is not None:
            # 如果提供了boost_match_regex，则用它来匹配文本
            return re.match(boost_match_regex, text) is not None
        return False  # 如果没有提供正则表达式，则默认不进行匹配

    # 从输入的样本中获取编码（encodings），该数据包含了上下文和问题的编码信息
    encodings = sample['encodings']

    global_index = 0  # 全局索引，用于标识每个段落
    all_similarities = []  # 用于存储每个段落与问题的相似度信息

    # 初始化空列表，用来存储所有句子和句子标签
    all_sents = []
    all_sents_labels = []
    # 遍历上下文中的所有片段，将它们合并到all_sents和all_sents_labels中
    for enc in encodings['context']:
        all_sents.extend(enc['segments'])  # 收集所有的句子
        all_sents_labels.extend(enc['segments_labels'])  # 收集每个句子的标签

    # 获取问题部分的编码信息
    encodings_question = encodings['question']
    # 将问题的input_ids和attention_mask转为Tensor，并将其移到GPU上
    inputs_question = {
        'input_ids': torch.LongTensor(encodings_question['input_ids']).cuda(),
        'attention_mask': torch.LongTensor(encodings_question['attention_mask']).cuda(),
    }

    # 通过模型计算问题的嵌入向量
    outputs_question = model(**inputs_question, is_train=False)
    # 采用mean pooling策略从模型输出中获取问题的嵌入
    q_emb = mean_pooling(outputs_question, inputs_question['attention_mask'])

    # 遍历上下文的每一个编码部分，进行处理
    for encodings_context in encodings['context']:
        # 将上下文中的每个片段的文本编码信息转为Tensor
        inputs_context = {
            'input_ids': torch.LongTensor(encodings_context['text_input_ids'])[None].cuda(),
            'attention_mask': torch.LongTensor([1 for _ in encodings_context['text_input_ids']])[None].cuda(),
            'text_segment_ids': torch.LongTensor(encodings_context['text_segment_ids'])[None].cuda()
        }
        # 只保留input_ids和attention_mask用于计算
        inputs_context_filtered = {
            k: inputs_context[k] for k in ['input_ids', 'attention_mask']
        }
        # 获取上下文部分的模型输出
        outputs_context = model(**inputs_context_filtered, is_train=False)

        max_seg_idx = encodings_context['text_segment_ids'][-1]  # 获取上下文中最大的段落索引
        embeddings_context_sentwise = []  # 存储上下文中每个句子的嵌入

        # 遍历每个句子，计算句子级的嵌入
        for sent_id in range(max_seg_idx + 1):
            mask = (inputs_context['text_segment_ids'] == sent_id)  # 创建句子掩码
            emb = mean_pooling(outputs_context, mask)  # 使用mean pooling计算该句子的嵌入
            embeddings_context_sentwise.append(emb)

        # 将所有句子的嵌入合并成一个Tensor
        embeddings_context_sentwise = torch.cat(embeddings_context_sentwise, 0)

        # 计算每个句子嵌入与问题嵌入的余弦相似度
        sim = cos_sim(embeddings_context_sentwise, q_emb)[:, 0].cpu().numpy()

        assert len(sim) == len(encodings_context['segments'])  # 确保相似度的长度与段落数相同

        # 将每个句子的相似度、句子文本及标签保存到all_similarities中
        for seg, seg_is_fake, s in zip(encodings_context['segments'], encodings_context['segments_labels'], sim):
            all_similarities.append((seg, s, global_index, re_checker(seg), seg_is_fake))
            global_index += 1

    taken = -1  # 初始化已选择的tokens数量
    indices = set()  # 用于存储选择的句子的索引

    # 按照相似度（以及是否匹配正则表达式）排序，选择最相关的句子
    for seg, s, idx, re_score, seg_is_fake in sorted(all_similarities, reverse=True, key=lambda x: (x[3], x[1])):
        if seg_is_fake:  # 跳过“虚假”句子
            continue
        ln = len(openai_tokenizer.encode(seg))  # 计算句子的token长度
        taken += ln + 1  # 累加已选择的tokens数量
        indices.add(idx)  # 将该句子的索引添加到选择的句子集合中
        if taken > compression_target_tokens:  # 如果已选择的tokens数量超过目标数量，则停止选择
            break

    # 根据选择的句子索引从all_similarities中提取被选择的句子
    compressed_segments = [seg for seg, s, idx, re_score, seg_is_fake in all_similarities if idx in indices]
    return compressed_segments  # 返回压缩后的句子



class CPCPromptCompressor:
    """
    一个动态评估句子与问题相关性的编码器
    """
    def __init__(self,
                 model_type,
                 use_question_as_suffix=False,
                 use_openai_tokenizer_to_measure_length=False):
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
        # configs = {
        #     ModelType.MISTRAL: os.path.join(configs_dir, 'cpc-1.0-mistral.json'),
        #     ModelType.LLAMA: os.path.join(configs_dir, 'cpc-1.0-llama.json'),
        # }
        configs = {
            ModelType.MISTRAL: {
                'config_path': os.path.join(configs_dir, 'cpc-1.0-mistral.json'),
                'lora_name_or_path': 'deadcode99/cpc-1.0-mistral-7b-ds-v5-iter66-lora-bidirectional-attn',
                'tokenizer_name_or_path': 'deadcode99/cpc-1.0-mistral-7b-tokenizer',
            },
            ModelType.LLAMA: {
                'config_path': os.path.join(configs_dir, 'cpc-1.0-llama.json'),
                'lora_name_or_path': 'deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn',
                'tokenizer_name_or_path': 'deadcode99/cpc-1.0-llama-1b-tokenizer',
            },
        }

        assert model_type in configs, f"Unsupported model type: {model_type}. Supported configuration are: {configs.keys()}"

        cfg = configs[model_type]

        self.model, self.tokenizer = load_model_and_tokenizer(**cfg)

        self.model.eval()

        with open(cfg['config_path']) as fin:
            train_conf = json.load(fin)

        self.processor = SamplePreprocessor(
            tokenizer=self.tokenizer,
            max_context_len=train_conf['max_seq_length'],
            use_question_as_suffix=use_question_as_suffix,
            sentence_embedding_type=SentenceEmbeddingType.AVG,
        )

        self.openai_tokenizer = None
        if use_openai_tokenizer_to_measure_length:
            self.openai_tokenizer = tiktoken.encoding_for_model('gpt-4')

    def _get_tokenizer_for_length_measure(self):
        if self.openai_tokenizer is not None:
            return self.openai_tokenizer
        return self.tokenizer

    def compress(self,
                 context,
                 question,
                 compression_target_tokens,
                 boost_sents_regexp=None):

        encodings = self.processor(
            context=context,
            question=question,
            question_for_suffix=question,
        )
        sentences = compress_sample(
            model=self.model,
            tokenizer=self.tokenizer,
            openai_tokenizer=self._get_tokenizer_for_length_measure(),
            sample={
                'encodings': encodings,
            },
            compression_target_tokens=compression_target_tokens,
            boost_match_regex=boost_sents_regexp,
        )

        return ' '.join(sentences)


if __name__ == '__main__':
    pass
