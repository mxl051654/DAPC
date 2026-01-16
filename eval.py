import os
import json
import argparse
import shutil
import re
import numpy as np
import pandas as pd
from datasets import load_dataset

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from consts import DATASET_SIZE

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,  # fuzz.ratio 文本相似度
    "repobench-p": code_sim_score,
}

target_order = {
    'Single-Document QA': ['narrativeqa', 'qasper', 'multifieldqa_en'],
    'Multi-Document QA': ['hotpotqa', '2wikimqa', 'musique'],
    'Summarization': ['gov_report', 'qmsum', 'multi_news'],
    'Few-shot Learning': ['trec', 'triviaqa', 'samsum'],
    # 'Synthetic': ['passage_count', 'passage_retrieval_en'],
    # 'Code': ['repobench-p', 'lcc'],
}


class BaseEvaluator:
    def __init__(self, filter_dict,
                 compress_model='/data/hf/Qwen/Qwen2.5-7B-Instruct',
                 reeval=False, dataset=None):
        self.filter_dict = filter_dict
        self.reeval = reeval
        self.root_dir = f'{compress_model.split("/")[-1]}_pred'
        self.compress_model = compress_model
        self.log_dir = 'excel_log'
        self.target_bench = filter_dict.get('bench', None)
        self.all_datasets = [v for vs in target_order.values() for v in vs]
        self.dataset = dataset

    def _should_skip_dir(self, out_dir, bench, model_name, method_str, method_paras):
        # 如果指定了 benchmark，且当前目录的 benchmark 不匹配，则跳过
        if self.target_bench and bench != self.target_bench:
            return True

        # 使用 filter_dict 进行过滤
        cont = False
        # filter_dict keys: bench, model_name, method_str, method_paras, out_dir
        # check bench
        if self.filter_dict.get('bench') and self.filter_dict['bench'] != bench:
            return True
        # check other fields
        checks = [
            ('model_name', model_name),
            ('method_str', method_str),
            ('method_paras', method_paras),
            ('out_dir', out_dir)
        ]
        for key, val in checks:
            required = self.filter_dict.get(key)
            if required and required not in val:
                return True
        return False

    def scorer(self, dataset, predictions, answers, lengths, all_classes):
        """
        Abstract method for scoring a dataset.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def format_score(self, score):
        """
        Format score for display.
        """
        if score is None:
            return "-"
        return f"{score:.2f}"

    def mean_or_none(self, vals):
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            return None
        return round(float(np.mean(vals)), 2)

    def mean_of_dicts(self, dict_list):
        if not dict_list:
            return None

        # Initialize sums and counts for each key
        sums = {}
        counts = {}

        all_keys = set()
        for d in dict_list:
            if d is None: continue
            all_keys.update(d.keys())

        for key in all_keys:
            vals = [d[key] for d in dict_list if d is not None and key in d]
            if vals:
                sums[key] = np.mean(vals)

        # Round values
        return {k: round(v, 2) for k, v in sums.items()}

    def print_summary(self, scores, bench_name):
        # 格式化输出
        def fmt(x, w=10):
            return f"{self.format_score(x):{'>'}{w}}"

        for k, vs in target_order.items():
            print(f"\n================ {k} ================")
            header = f"{'Dataset':<22}{'Score':>10}"
            print(header)
            cat_vals = []
            for v in vs:
                if v not in scores:
                    continue
                val = scores[v]
                cat_vals.append(val)
                print(f"{v:<22}{fmt(val)}")

            # 对于 dict 类型的分数（如 longbench-e），计算均值
            if cat_vals:
                if isinstance(cat_vals[0], (int, float)):
                    cat_mean = self.mean_or_none(cat_vals)
                    print(f"{'Category Avg':<22}{fmt(cat_mean)}")
                elif isinstance(cat_vals[0], dict):
                    cat_mean = self.mean_of_dicts(cat_vals)
                    print(f"{'Category Avg':<22}{fmt(cat_mean)}")
                else:
                    print(f"{'Category Avg':<22}{'-'}")
            else:
                print(f"{'Category Avg':<22}{'-'}")

        print("\n================ Overall ================")
        overall_vals = []
        for v in self.all_datasets:
            if v in scores:
                overall_vals.append(scores[v])

        if overall_vals:
            if isinstance(overall_vals[0], (int, float)):
                overall_mean = self.mean_or_none(overall_vals)
                print(f"{'All Tasks Avg':<22}{fmt(overall_mean)}")
            elif isinstance(overall_vals[0], dict):
                overall_mean = self.mean_of_dicts(overall_vals)
                print(f"{'All Tasks Avg':<22}{fmt(overall_mean)}")
            else:
                print(f"{'All Tasks Avg':<22}{'-'}")
        else:
            print(f"{'All Tasks Avg':<22}{'-'}")

    def save_detailed_report(self, rows, bench_name):
        outdf = pd.DataFrame(rows, columns=["model", "method", "paras"] + self.all_datasets)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        outdf.to_excel(
            f"{self.log_dir}/{self.compress_model.split('/')[-1]}{bench_name}_results_detailed.xlsx",
            index=False)
        return outdf

    def save_summary_report(self, rows, bench_name):
        summary_rows = []
        for r in rows:
            sr = {"model": r["model"], "method": r["method"], "paras": r["paras"]}
            for k, vs in target_order.items():
                vals = [r.get(v, None) for v in vs]

                float_vals = [x for x in vals if x is not None and isinstance(x, (int, float))]
                dict_vals = [x for x in vals if x is not None and isinstance(x, dict)]

                if float_vals:
                    sr[k] = round(float(np.mean(float_vals)), 2)
                elif dict_vals:
                    sr[k] = str(self.mean_of_dicts(dict_vals))
                else:
                    sr[k] = None

            overall_vals = [r.get(v, None) for v in self.all_datasets]
            float_vals = [x for x in overall_vals if x is not None and isinstance(x, (int, float))]
            dict_vals = [x for x in overall_vals if x is not None and isinstance(x, dict)]

            if float_vals:
                sr["全部数据"] = round(float(np.mean(float_vals)), 2)
            elif dict_vals:
                sr["全部数据"] = str(self.mean_of_dicts(dict_vals))
            else:
                sr["全部数据"] = None

            summary_rows.append(sr)

        summary_df = pd.DataFrame(summary_rows,
                                  columns=["model", "method", "paras"] + list(target_order.keys()) + ["全部数据"])
        summary_df.to_excel(f"{self.log_dir}/{self.compress_model.split('/')[-1]}"
                            f"_{bench_name}_results_summary.xlsx", index=False)
        return summary_df

    def run(self):
        if not os.path.exists(self.root_dir):
            print(f"Directory {self.root_dir} does not exist.")
            return

        exp_dirs = [x for x in os.listdir(self.root_dir) if not x.endswith('xlsx')]
        rows = []
        bench_processed = None

        for out_dir in exp_dirs:  # 遍历全部目录
            parts = out_dir.split('_')
            if len(parts) < 3:
                continue

            bench = parts[0]
            model_name = parts[1]
            method_str = parts[2]
            method_paras = '_'.join(parts[3:])
            if self._should_skip_dir(out_dir, bench, model_name, method_str, method_paras):
                continue

            bench_processed = bench  # Record which bench we are processing
            print(f"\n=== processing {out_dir} ===")
            save_path = f"{self.root_dir}/{out_dir}/result.json"

            scores = dict()
            if os.path.exists(save_path):  # 加载现有记录
                with open(save_path, "r", encoding="utf-8") as f:
                    scores = json.load(f)

            result_dir = f"{self.root_dir}/{out_dir}/result"
            if not os.path.exists(result_dir):
                print(f"Result directory {result_dir} not found.")
                continue

            all_files = os.listdir(result_dir)
            print(f"=== Evaluating on: {all_files} ===")
            for filename in all_files:  # 遍历目录下所有数据集
                if not filename.endswith("jsonl"):
                    continue

                if self.dataset is not None and self.dataset not in filename:
                    continue

                dataset = filename.split('.')[0]
                original_size = DATASET_SIZE[bench].get(dataset, 0)

                if not self.reeval:
                    if dataset in scores and f"{dataset}_info" in scores:
                        if scores[f"{dataset}_info"]["eval_count"] == original_size:
                            print(f"Skipping {dataset}, already evaluated.")
                            continue

                predictions, answers, lengths = [], [], []
                all_classes = 'null'
                to_eval_path = f"{result_dir}/{filename}"

                with open(to_eval_path, "r", encoding="utf-8") as f:
                    count, total = 0, 0
                    for line in f:
                        try:
                            total += 1
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                            count += 1
                        except Exception as e:
                            pass

                if count < 0.9 * total:
                    print(f"Warning: file {to_eval_path} seems incomplete or corrupted. Removing.")
                    try:
                        os.remove(to_eval_path)
                    except:
                        pass
                    continue

                score = self.scorer(dataset, predictions, answers, lengths, all_classes)
                scores[dataset] = score
                scores[f"{dataset}_info"] = {"eval_count": count, "total_count": original_size}

                # Simple print of current score
                display_score = self.format_score(score)
                print(f"{dataset}: {display_score} per: {count / original_size if original_size else 0:.2f}")

            # Save updated scores
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)

            self.print_summary(scores, bench)

            # Prepare row for excel
            row = {"model": model_name, "method": method_str, "paras": method_paras}
            for v in self.all_datasets:
                row[v] = scores.get(v, None)
            rows.append(row)

        if rows:
            bench_name = self.target_bench if self.target_bench else (bench_processed if bench_processed else "unknown")
            df_detail = self.save_detailed_report(rows, bench_name)
            df_summ = self.save_summary_report(rows, bench_name)
        return df_detail, df_summ


class LongBenchEvaluator(BaseEvaluator):
    def scorer(self, dataset, predictions, answers, lengths, all_classes):
        if len(predictions) > 0:
            total_score = 0.
            for (prediction, ground_truths) in zip(predictions, answers):
                score = 0.
                if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                    prediction = prediction.lstrip('\n').split('\n')[0]
                for ground_truth in ground_truths:
                    score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
                total_score += score
            return round(100 * total_score / len(predictions), 2)
        else:
            return 0


class LongBenchEEvaluator(BaseEvaluator):
    def scorer(self, dataset, predictions, answers, lengths, all_classes):
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
        for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)

        for key in scores.keys():
            if scores[key]:
                scores[key] = round(100 * np.mean(scores[key]), 2)
            else:
                scores[key] = 0.0  # Or None
        return scores

    def format_score(self, score):
        if isinstance(score, dict):
            # Calculate mean if possible for display
            vals = [v for v in score.values() if v is not None]
            avg = np.mean(vals) if vals else 0
            # Return a compact string
            return f"{avg:.2f} ({score.get('0-4k', 0):.1f}/{score.get('4-8k', 0):.1f}/{score.get('8k+', 0):.1f})"
        return str(score)

    def save_detailed_report(self, rows, bench_name):
        expanded_rows = []
        new_columns = ["model", "method", "paras"]
        for d in self.all_datasets:
            # new_columns.extend([f"{d}", f"{d}_0-4k", f"{d}_4-8k", f"{d}_8k+", f"{d}_avg"])
            new_columns.extend([f"{d}_avg"])

        for r in rows:
            new_r = {"model": r["model"], "method": r["method"], "paras": r["paras"]}
            for dataset in self.all_datasets:
                val = r.get(dataset)
                if isinstance(val, dict):
                    # Extract values
                    v0_4k = val.get('0-4k')
                    v4_8k = val.get('4-8k')
                    v8k_plus = val.get('8k+')

                    # Calculate average
                    sub_vals = [x for x in [v0_4k, v4_8k, v8k_plus] if x is not None]
                    avg = round(np.mean(sub_vals), 2) if sub_vals else None

                    # new_r[f"{dataset}"] = str(val) # Keep original as string
                    # new_r[f"{dataset}_0-4k"] = v0_4k
                    # new_r[f"{dataset}_4-8k"] = v4_8k
                    # new_r[f"{dataset}_8k+"] = v8k_plus
                    new_r[f"{dataset}_avg"] = avg
                else:
                    # Fallback if not a dict (e.g. None or unexpected)
                    # new_r[f"{dataset}"] = val
                    # new_r[f"{dataset}_0-4k"] = None
                    # new_r[f"{dataset}_4-8k"] = None
                    # new_r[f"{dataset}_8k+"] = None
                    new_r[f"{dataset}_avg"] = None

            expanded_rows.append(new_r)

        outdf = pd.DataFrame(expanded_rows, columns=new_columns)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        outdf.to_excel(f"{self.log_dir}/{bench_name}_results_detailed.xlsx", index=False)

if __name__ == '__main__':
    # NOTE main
    filter_dict = {
        # 'bench': 'longbench-e',
        'bench': 'longbench',
        'model_name': '8B',
        'method_str': None,
        'method_paras': None,
        'out_dir': None,
    }
    # target = 'trec'
    target = None
    bench_type = filter_dict.get('bench', 'longbench')

    for cm in [
        # '/data/hf/Qwen/Qwen2.5-0.5B-Instruct',
        # '/data/hf/Qwen/Qwen2.5-1.5B-Instruct',
        # '/data/hf/Qwen/Qwen2.5-3B-Instruct',
        '/data/hf/Qwen/Qwen2.5-7B-Instruct',
    ]:
        print(f"Process compress model {cm}")
        if bench_type == 'longbench-e':
            evaluator = LongBenchEEvaluator(filter_dict, compress_model=cm, reeval=False, dataset=target)
        else:
            evaluator = LongBenchEvaluator(filter_dict, compress_model=cm, reeval=False, dataset=target)
        evaluator.run()
