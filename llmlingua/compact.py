import os
import re
import json

import argparse
from tqdm import tqdm
import logging
import datetime
from pytz import timezone, utc

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_logger(name=None):
    if not name:
        name = 'main'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers
    if not logger.hasHandlers():
        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a custom formatter
        def customTime(*args):
            utc_dt = datetime.datetime.now()
            my_tz = timezone("Asia/Seoul")
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter.converter = customTime
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(ch)

    return logger


def get_dataset(data, ans_key='answers', ctxs_key='ctxs', demos='', n_docs=100):
    entries = []
    for ins in data:
        question = ins['question']
        docs = ins[ctxs_key]
        document_list = []
        for i in range(n_docs):
            if ctxs_key == 'context':
                title = docs[i][0]
                text = docs[i][1]
            else:
                title = docs[i]['title']
                text = docs[i]['text']

            document_list.append(docs[i])
            # document_list.append(f"{title} {text}")

        entry = {'documents_list': document_list,
                 'question': question,
                 'answer': ", ".join(ins[ans_key]),
                 'answers': ins[ans_key],
                 'demos': demos
                 }

        if '_id' in ins:
            entry['_id'] = ins['_id']
        else:
            if 'id' in ins:
                entry['id'] = ins['id']

        if 'supporting_facts' in ins:
            entry['supporting_facts'] = ins['supporting_facts']

        entries += [entry]

    # return [dsp.Example(**entry) for entry in entries]
    return entries


def create_prompt(query, iter_idx, document_input, prev_summary, prev_eval, tokenizer,
                  eos_token="<|endoftext|>", add_generation_prompt=False):
    if iter_idx == 0:
        instruction = "1. Generate a summary of source documents to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {query}\n\nSource documents: {document_input}\n\nSummary:"
    else:
        instruction = "1. Generate a summary of the previous summary and the source documents to answer the question based on the evaluation of the previous summary. The evaluation indicates the missing information needed to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {query}\n\nPrevious summary: {prev_summary}\n\nEvaluation of previous summary: {prev_eval}\n\nSource documents: {document_input}\n\nSummary:"

    messages = [
        {"role": "user", "content": prompt},
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    return chat_format


def parse_output_without_sentence(text):
    # 创建一个空字典，存储提取的内容
    sections = {}

    # 定义三个正则表达式模式，分别用于匹配含有或不含“Summary:”前缀的摘要部分，以及“Evaluation:”部分
    summary_pattern_with_prefix = r'(Summary:)(.*?)(?=Evaluation:|$)'  # 含有“Summary:”的部分
    summary_pattern_without_prefix = r'(^.*?)(?=Evaluation:|$)'  # 不含“Summary:”的部分
    evaluation_pattern = r'(Evaluation:)(.*?)(?=Summary:|$)'  # 匹配“Evaluation:”部分

    # 使用re.search进行正则表达式匹配，re.DOTALL使得点号可以匹配换行符
    summary_match_with_prefix = re.search(summary_pattern_with_prefix, text, re.DOTALL)
    summary_match_without_prefix = re.search(summary_pattern_without_prefix, text, re.DOTALL)
    evaluation_match = re.search(evaluation_pattern, text, re.DOTALL)

    # 提取并清理匹配到的内容
    if summary_match_with_prefix:
        # 如果找到带有“Summary:”前缀的摘要部分，提取并去除前后空白
        sections['summary'] = summary_match_with_prefix.group(2).strip()
    elif summary_match_without_prefix:
        # 如果找到不带“Summary:”前缀的摘要部分，提取并去除前后空白
        sections['summary'] = summary_match_without_prefix.group(1).strip()
    else:
        # 如果都没有找到，则将“summary”设为空字符串
        sections['summary'] = ""

    if evaluation_match:
        # 如果找到“Evaluation:”部分，提取并去除前后空白
        sections['eval'] = evaluation_match.group(2).strip()
    if sections['summary'] == "":
        sections['eval'] = evaluation_match.group(1).strip()

    # 清理多余的换行符，如果有连续的两个换行符，将它们替换为空
    sections['summary'] = sections['summary'].replace("\n\n", "")
    sections['eval'] = sections['eval'].replace("\n\n", "")

    # 返回包含“summary”和“eval”部分的字典
    return sections


class CompActCompressor:
    """
    # NOTE 相当于针对上下文逐seg判断是否必要
    将检索到的文档分割为多个片段（如每段5个文档），顺序处理。每次迭代时，
    模型联合分析已压缩上下文（C_{t-1}）与新片段（S_t），生成更凝练的上下文（C_t）
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dir = '/data/hf/cwyoon99/CompAct-7b'
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def compress(self, query, docs):

        seg_size = 5
        segs = []
        for i in range(int(len(docs)/seg_size)):
            segs.append("\n".join(docs[i*seg_size:(i+1)*seg_size]))

        prev_summary = []
        prev_eval = []
        for i, seg in enumerate(segs):
            # NOTE p
            prev_summary_temp = prev_summary[-1] if i != 0 else ""
            prev_eval_temp = prev_eval[-1].replace('[INCOMPLETE]', '').strip() if i != 0 else ""

            input_prompt = create_prompt(query, i, seg, prev_summary_temp, prev_eval_temp,
                                         self.tokenizer, eos_token="", add_generation_prompt=True)

            with torch.no_grad():
                inputs = self.tokenizer(input_prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                outputs = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=500,
                    temperature=0, top_p=1.0, pad_token_id=self.tokenizer.eos_token_id
                )
            output = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()

            # print('output', output)
            parsed_sections = parse_output_without_sentence(output)
            prev_summary.append(parsed_sections['summary'])
            prev_eval.append(parsed_sections['eval'])
            # compressing extensive documents into compact context (under 200 tokens)
            print(f"summary of segment {i}: {prev_summary[-1]}\ntermination of segment {i}: {prev_eval[-1]}\n")

            if "[COMPLETE]" in output:
                break

        return prev_summary[-1]

    def test(self):
        # { question,answer,answers,iterations list of documents_list}
        example = json.load(open('/data/mxl/PC/CompAct/data/example.json'))
        print(f"question: {example['question']}\nanswer: {example['answer']}")

        docs = []
        for i, iteration in enumerate(example['iterations']):
            docs.extend(iteration['documents_list'])

        summary = self.compress(example['query'], docs)
        print(summary)


if __name__ == '__main__':
    compressor = CompActCompressor()
    compressor.test()
