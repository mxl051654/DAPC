import re


def get_method_para(args, infer=False):
    if args.method in ['original', 'zero-shot']:
        method_str = args.method
    else:
        method_str = args.method
        if 'rollout-contrast' in args.method:
            method_str += f'_m{args.rollout_m}'

        method_str += f"_cs{args.chunk_size}"
        if f"contrast" in args.method:
            method_str += f"_ca{args.contrast_alpha}"

        if infer:
            if args.compress_rate is not None:
                method_str += f"_r{args.compress_rate}"
            else:
                method_str += f"_t{args.target_len}"
            method_str += f"_{args.budget_policy}"  # chunk global
    return method_str


def my_load_dataset(bench, dataset):
    from datasets import load_dataset, Features, Value, Sequence
    data_dir = '/data/hf'
    assert bench in ['longbench', 'longbench-e', 'infinitebench']
    dataset_list = DATASET_LIST[bench]
    assert dataset in dataset_list

    data_name = f"{dataset}_e" if bench == 'longbench-e' else dataset
    if 'longbench' in bench:
        data = load_dataset(f'{data_dir}/THUDM/LongBench', data_name, split='test')
    elif 'infinite' in bench:
        ft = Features({
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string"))
        })
        data = load_dataset(f"{data_dir}/xinrongzhang2022/InfiniteBench", features=ft)[data_name]
    else:
        print(f"no bench {bench}")
    return data


def prepare_query(item, bench, dataset, infer=False, method=None):
    if infer:  # 需要 context
        template = DEFAULT_INPUT[bench][dataset]
        if method == 'zero-shot':
            template = template.replace("{context}", "context")

        if bench in ['longbench', 'longbench-e']:
            prompt = template.format(**item)  # 无上下文
        elif bench == 'infinitebench':
            if dataset == "code_run":
                find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", item['input'])
                func_call = find_result[0]
                func = func_call.split("(")[0]
                return template.format(
                    func=func,
                    func_call=func_call,
                    context=item["context"],
                )
            elif dataset in ["code_debug", "code_debug_qa"]:
                code = item["context"]
                if dataset == "code_debug":
                    return template.format(
                        context=code,
                        OPTION_A=item["options"][0],
                        OPTION_B=item["options"][1],
                        OPTION_C=item["options"][2],
                        OPTION_D=item["options"][3],
                    )
                return template.format(context=code)
            elif dataset == "longdialogue_qa_eng":
                script = item["context"]
                prompt = template.format(context=script)
                return prompt
            elif dataset in [
                "longbook_choice_eng",
                "longbook_qa_eng",
                "longbook_sum_eng",
                "longbook_qa_chn",
            ]:
                book = item["context"]
                if dataset == "longbook_choice_eng":
                    return template.format(
                        question=item["input"],
                        context=book,
                        OPTION_A=item["options"][0],
                        OPTION_B=item["options"][1],
                        OPTION_C=item["options"][2],
                        OPTION_D=item["options"][3],
                    )
                elif dataset == "longbook_qa_eng":
                    return template.format(
                        question=item["input"],
                        context=book,
                    )
                elif dataset == "longbook_sum_eng":
                    return template.format(context=book)
                elif dataset == "longbook_qa_chn":
                    return template.format(
                        question=item["input"],
                        context=book,
                    )
                else:
                    raise ValueError
            elif dataset == "math_calc":
                return template.format(context=item["context"])
            elif dataset == "math_find":
                prompt = item['input']
                context = item['context']
                find_result = re.findall(r"The .+ of", prompt)
                assert find_result, f"Cannot find the target number in {prompt}"
                target_number = find_result[0].lower()[:-3]
                prefix = f"What is {target_number} in the following list?"
                return template.format(
                    prefix=prefix,
                    context=context,
                    input=prompt,
                )

            # Default behavior if content key exists
            if "content" in item:
                content = item["content"]
                del item["content"]
                item["context"] = content

            format_dict = {
                "context": item["context"],
                "input": item["input"],
            }
            prompt = template.format(**format_dict)

    else:
        if bench in ['longbench', 'longbench-e']:
            prompt = INPUT_WO_CONTEXT[bench][dataset].format(**item)  # 无上下文
        elif bench == 'infinitebench':
            template = INPUT_WO_CONTEXT[bench][dataset]
            if dataset == "code_run":
                find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", item['input'])
                func_call = find_result[0]
                func = func_call.split("(")[0]
                return template.format(
                    func=func,
                    func_call=func_call,
                    context=item["context"],
                )
            elif dataset in ["code_debug", "code_debug_qa"]:
                code = item["context"]
                if dataset == "code_debug":
                    return template.format(
                        context=code,
                        OPTION_A=item["options"][0],
                        OPTION_B=item["options"][1],
                        OPTION_C=item["options"][2],
                        OPTION_D=item["options"][3],
                    )
                return template.format(context=code)
            elif dataset == "longdialogue_qa_eng":
                script = item["context"]
                prompt = template.format(context=script)
                return prompt
            elif dataset in [
                "longbook_choice_eng",
                "longbook_qa_eng",
                "longbook_sum_eng",
                "longbook_qa_chn",
            ]:
                book = item["context"]
                if dataset == "longbook_choice_eng":
                    return template.format(
                        question=item["input"],
                        context=book,
                        OPTION_A=item["options"][0],
                        OPTION_B=item["options"][1],
                        OPTION_C=item["options"][2],
                        OPTION_D=item["options"][3],
                    )
                elif dataset == "longbook_qa_eng":
                    return template.format(
                        question=item["input"],
                        context=book,
                    )
                elif dataset == "longbook_sum_eng":
                    return template.format(context=book)
                elif dataset == "longbook_qa_chn":
                    return template.format(
                        question=item["input"],
                        context=book,
                    )
                else:
                    raise ValueError
            elif dataset == "math_calc":
                return template.format(context=item["context"])
            elif dataset == "math_find":
                prompt = item['input']
                context = item['context']
                find_result = re.findall(r"The .+ of", prompt)
                assert find_result, f"Cannot find the target number in {prompt}"
                target_number = find_result[0].lower()[:-3]
                prefix = f"What is {target_number} in the following list?"
                return template.format(
                    prefix=prefix,
                    context=context,
                    input=prompt,
                )

            # Default behavior if content key exists
            if "content" in item:
                content = item["content"]
                del item["content"]
                item["context"] = content

            format_dict = {
                "context": item["context"],
                "input": item["input"],
            }
            prompt = template.format(**format_dict)
    return prompt


DATASET_LIST = {
    'longbench-e': [
        "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa",
        "gov_report", "multi_news",
        "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",
        "lcc", "repobench-p"
    ],
    'longbench': [
        'narrativeqa', 'qasper', 'multifieldqa_en',
        'hotpotqa', '2wikimqa', 'musique',
        'gov_report', 'qmsum', 'multi_news',
        'trec', 'triviaqa', 'samsum',
        'passage_count', 'passage_retrieval_en',
        'repobench-p', 'lcc',
    ],
    'infinitebench': [
        "passkey", "number_string", "kv_retrieval",
        "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_eng", "longbook_qa_chn",
        "longdialogue_qa_eng",
        "math_find", "math_calc",
        "code_run", "code_debug",
    ]
}

DATASET_SIZE = {
    'longbench': {
        'narrativeqa': 200,  # 基于故事或剧本回答问题，包括对角色、情节、主题等重要元素的理解
        'qasper': 200,  # 基于 NLP 研究论文回答问题，由 NLP 从业者提出并回答  NOTE parsed paper paper
        'multifieldqa_en': 150,  # 基于来自相对多样化领域的长篇文章回答英文问题
        'hotpotqa': 200,  # 基于多个给定文档回答相关问题   # NOTE 存在相关内容
        '2wikimqa': 200,  # 基于多个给定文档回答相关问题
        'musique': 200,  # 基于多个给定文档回答相关问题
        'gov_report': 200,  # 需要总结政府工作报告的摘要任务  NOTE 无输入
        'qmsum': 200,  # 根据用户查询对会议记录进行摘要的任务  NOTE 指定摘要主题
        'multi_news': 200,  # 多文档摘要任务，要求对多篇新闻进行摘要  NOTE 无输入
        'trec': 200,  # 分类任务，需要对问题进行分类，总共包含 50 个类别   NOTE Question:xxx Type:
        'triviaqa': 200,  # 单文档问答任务，提供几个少样本示例  NOTE 无关的 { Passage:xxx,  Answer: xxx} demos
        'samsum': 200,  # 对话摘要任务，提供几个少样本示例   NOTE  无关的 {Dialogue:xxx , Summary: xxx}
        'passage_count': 200,  # 确定给定重复文章中的不同段落总数   NOTE 无输入   Passage 1 : N
        'passage_retrieval_en': 200,  # 给定 30 段英文维基百科段落，确定哪个段落与给定的摘要对应  NOTE 返回段落ID
        'repobench-p': 500,  # 给定 GitHub 仓库中多个文件内的代码（包括跨文件依赖），预测下一行代码  NOTE 无输入
        'lcc': 500,  # 给定一段长代码，预测下一行代码  NOTE 无输入  注意回答只需要一句
    },
    'longbench-e': {
        'qasper': 224,
        'multifieldqa_en': 150,
        'hotpotqa': 300,
        '2wikimqa': 300,
        'gov_report': 300,
        'multi_news': 294,
        'trec': 300,
        'triviaqa': 300,
        'samsum': 300,
        'passage_count': 300,
        'passage_retrieval_en': 300,
        'lcc': 300,
        'repobench-p': 300
    },
    # NOTE  https://github.com/OpenBMB/InfiniteBench
    'infinitebench': {
        "passkey": 590,
        "number_string": 590,
        "kv_retrieval": 500,
        "longbook_sum_eng": 103,
        "longbook_choice_eng": 229,
        "longbook_qa_eng": 351,
        "longbook_qa_chn": 189,  # 175,
        "longdialogue_qa_eng": 200,
        "math_find": 350,
        "math_calc": 50,
        "code_run": 400,
        "code_debug": 394,
    }
}

DATALENGTH = {

    'longbench': {

        "narrativeqa": 2.9836,
        "qasper": 0.5025,
        "multifieldqa_en": 0.7118,
        "hotpotqa": 1.3371,
        "2wikimqa": 0.7474,
        "musique": 1.6269,
        "gov_report": 1.0589,
        "qmsum": 1.3885,
        "multi_news": 0.2653,
        "trec": 0.6814,
        "triviaqa": 1.1441,
        "samsum": 0.9029,

    }

}

GENERATE_SIZE = {
    # NOTE https://github.com/microsoft/MInference/tree/main/scbench

    'longbench': {
        'narrativeqa': 128,  # 基于故事或剧本回答问题，包括对角色、情节、主题等重要元素的理解
        'qasper': 128,  # 基于 NLP 研究论文回答问题，由 NLP 从业者提出并回答  NOTE parsed paper paper
        'multifieldqa_en': 128,  # 基于来自相对多样化领域的长篇文章回答英文问题

        'hotpotqa': 128,  # 基于多个给定文档回答相关问题   # NOTE 存在相关内容
        '2wikimqa': 128,  # 基于多个给定文档回答相关问题
        'musique': 128,  # 基于多个给定文档回答相关问题

        'gov_report': 256,  # 需要总结政府工作报告的摘要任务  NOTE 无输入
        'qmsum': 256,  # 根据用户查询对会议记录进行摘要的任务  NOTE 指定摘要主题
        'multi_news': 256,  # 多文档摘要任务，要求对多篇新闻进行摘要  NOTE 无输入

        'trec': 64,  # 分类任务，需要对问题进行分类，总共包含 50 个类别   NOTE Question:xxx Type:
        'triviaqa': 64,  # 单文档问答任务，提供几个少样本示例  NOTE 无关的 { Passage:xxx,  Answer: xxx} demos
        'samsum': 64,  # 对话摘要任务，提供几个少样本示例   NOTE  无关的 {Dialogue:xxx , Summary: xxx}

        'passage_count': 16,  # 确定给定重复文章中的不同段落总数   NOTE 无输入   Passage 1 : N
        'passage_retrieval_en': 16,  # 给定 30 段英文维基百科段落，确定哪个段落与给定的摘要对应  NOTE 返回段落ID

        'repobench-p': 64,  # 给定 GitHub 仓库中多个文件内的代码（包括跨文件依赖），预测下一行代码  NOTE 无输入
        'lcc': 64,  # 给定一段长代码，预测下一行代码  NOTE 无输入  注意回答只需要一句
    },

    'longbench-e': {
        'qasper': 128,
        'multifieldqa_en': 128,
        'hotpotqa': 128,
        '2wikimqa': 128,
        'gov_report': 256,
        'multi_news': 256,
        'trec': 64,
        'triviaqa': 64,
        'samsum': 64,
        'passage_count': 16,
        'passage_retrieval_en': 16,
        'lcc': 64,
        'repobench-p': 64,
    },

    # NOTE  https://github.com/OpenBMB/InfiniteBench
    'infinitebench': {
        "passkey": 6,  # 在嘈杂的长上下文中检索隐藏的密钥
        "number_string": 12,  # 在嘈杂的长上下文中定位重复隐藏号码。
        "kv_retrieval": 50,  # 从字典和键中查找对应的值。
        "longbook_sum_eng": 1200,  # 用核心实体替换创建的假书的摘要。
        "longbook_choice_eng": 40,  # 假书衍生的多项选择题。
        "longbook_qa_eng": 40,  # 基于假书的自由形式的问题回答。
        "longbook_qa_chn": 40,  # 关于一套新收藏图书的问答
        "longdialogue_qa_eng": 40,  # 部分匿名脚本中通话者的识别
        "math_find": 3,  # 在长列表中查找特殊整数。
        "math_calc": 30000,  # 涉及超长算术方程的计算。
        "code_run": 5,  # 模拟多个简单的合成函数的执行。
        "code_debug": 5,  # 查找代码库中哪个函数包含崩溃错误（以多项选择形式）。
    }
}

TARGET_ORDER = {
    'longbench-e': {
        'Single-Document QA': ['qasper', 'multifieldqa_en'],
        'Multi-Document QA': ['hotpotqa', '2wikimqa'],
        'Summarization': ['gov_report', 'multi_news'],
        'Few-shot Learning': ['trec', 'triviaqa', 'samsum'],
        'Synthetic': ['passage_count', 'passage_retrieval_en'],
        'Code': ['repobench-p', 'lcc'],
    },
    'longbench': {
        'Single-Document QA': ['narrativeqa', 'qasper', 'multifieldqa_en'],
        'Multi-Document QA': ['hotpotqa', '2wikimqa', 'musique'],
        'Summarization': ['gov_report', 'qmsum', 'multi_news'],
        'Few-shot Learning': ['trec', 'triviaqa', 'samsum'],
        'Synthetic': ['passage_count', 'passage_retrieval_en'],
        'Code': ['repobench-p', 'lcc'],
    }
}

IDNAME = {
    'longbench': '_id',
    'longbench-e': '_id',
    'infinitebench': 'id'
}

# NOTE # dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
# DEFAULT_INPUT = {
#     "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
#     "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
#     "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
#     "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
#     "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
#     "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
#     "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
#     "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
#     "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
#     "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
#     "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
#     "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
#     "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
#     "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
#     "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
#     "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
#     "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
# }

# NOTE optimized prompts  Context 和 query 之间需要大量分隔
DEFAULT_INPUT = {
    'longbench': {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. "
                       "Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation."
                       "\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. "
                  "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
                  "Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
                  "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\"."
                  " Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\n"
                           "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}"
                           "\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                   "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}"
                      "\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}"
                 "\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}"
                      "\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}"
                 "\n\n请基于上述会议记录写一段会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}"
                "\n\nNow please determine the type of the question below.\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}"
                    "\n\nNow Answer the question based on the given passage. Only give me the answer and do not output any other words. {input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}"
                "\n\n现在请判断给定新闻的类别，{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. "
                         "In other words, how many non-repeating paragraphs are there in total?\n\n{context}"
                         "\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}"
                                "\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc."
                                "\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}"
                                "\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. \n{context}"
               "\n\nNow answer the next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}"
                       "\n\nNow complete the code given below:\n{input}Next line of code:\n"
    },

    'longbench-e': {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. "
                       "Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation."
                       "\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. "
                  "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
                  "Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
                  "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\"."
                  " Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\n"
                           "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}"
                           "\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                   "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}"
                      "\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}"
                 "\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}"
                      "\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}"
                 "\n\n请基于上述会议记录写一段会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}"
                "\n\nNow please determine the type of the question below.\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}"
                    "\n\nNow Answer the question based on the given passage. Only give me the answer and do not output any other words. {input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}"
                "\n\n现在请判断给定新闻的类别，{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. "
                         "In other words, how many non-repeating paragraphs are there in total?\n\n{context}"
                         "\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}"
                                "\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc."
                                "\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}"
                                "\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. \n{context}"
               "\n\nNow answer the next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}"
                       "\n\nNow complete the code given below:\n{input}Next line of code:\n"
    },
    'infinitebench': {
        "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
        "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
        "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",
        "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
        "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
        "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
        "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
        "math_find": "{prefix}\n\n{context}\n\n{input}",
        "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",
        "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",
        "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
        "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",
    },
}

# NOTE 去除上下文插槽 （for query aware compression）
INPUT_WO_CONTEXT = {
    'longbench': {
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
    },
    'longbench-e': {
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
    },
    'infinitebench': {
        "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{input}\n\nThe pass key is",
        "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{input}\n\nThe sequence of digits is",
        "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{input}",
        "longbook_sum_eng": "Summarize the book below.\n\nSummary:",  # noqa
        "longbook_choice_eng": "Read the book and answer the question.\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
        "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\nQuestion: {question}\nAnswer:",
        "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n问题：{question}\n答案：",  # noqa
        "math_find": "\n\n{input}",
        "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nValues:",
        "code_run": "There is a function called {func} in the following Python code.\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",
        "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
        "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. \n\nThe name that has been replaced with $$MASK$$ is likely",
    },
}


def count_dataset_samples():
    target = [
        'narrativeqa', 'qasper', 'multifieldqa_en',
        'hotpotqa', '2wikimqa', 'musique',
        'gov_report', 'qmsum', 'multi_news',
        'trec', 'triviaqa', 'samsum',
    ]

    from transformers import AutoTokenizer
    from matplotlib import pyplot as plt
    import os
    model_name_or_path = '/data/hf/Qwen/Qwen2.5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    plt.rcParams.update({'font.size': 25})

    for category in [
        'longbench',
        # 'longbench-e',
        # 'infinitebench'
    ]:
        dataset_list = DATASET_LIST[category]
        all_lengths = []
        counts = []
        labels = []
        for dataset in dataset_list:

            if dataset not in target:
                continue

            data = my_load_dataset(category, dataset)
            sample_length = [len(tokenizer.tokenize(x['context'])) / (1e4) for x in data]
            all_lengths.append(sample_length)
            print(dataset, sum(sample_length) / len(sample_length))
            counts.append(len(data))
            labels.append(dataset)

        positions = list(range(1, len(labels) + 1))
        fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 1.2), 8))
        bp = ax1.boxplot(all_lengths, positions=positions, widths=0.6, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor='#87CEEB', alpha=0.7)
        # ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Context Length (1e4)')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        # ax1.set_title(f'{category} Dataset Length')

        plt.tight_layout()

        out_dir = os.path.join('exp_pics', 'const')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{category}_length_count.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    count_dataset_samples()
