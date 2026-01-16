import os
import json
import argparse
import shutil
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
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


def view_different_ratio(budget_policy, reeval=False):
    filter_dict = {
        'bench': 'longbench',
        'model_name': '7B',
        'method_str': 'original',
        'method_paras': None,
        'out_dir': None,  # global, chunk
    }
    evaluator = LongBenchEvaluator(filter_dict, reeval=reeval)
    df_det_origin, df_sum_origin = evaluator.run()

    # filter_dict = {
    #     'bench': 'longbench',
    #     'model_name': '7B',
    #     'method_str': None,
    #     'method_paras': 'r',
    #     'out_dir': budget_policy,  # global, chunk
    # }
    # evaluator = LongBenchEvaluator(filter_dict, reeval=reeval)
    # df_det, df_sum = evaluator.run()

    # Qwen2.5-7B-Instruct_dif_ratio.xlsx
    df_det = pd.read_excel('/data/mxl/PC/longbench/excel_log/Qwen2.5-7B-Instruct_dif_ratio.xlsx')

    DNs = {
        'narrativeqa': 'NarrativeQA',
        'qasper': 'Qasper',
        'multifieldqa_en': 'MultiFieldQA-en',
        'hotpotqa': 'HotpotQA',
        '2wikimqa': '2WikiMultihopQA',
        'musique': 'MuSiQue',
    }
    target_methods = [
        'p-contrast-qa',
        'ehpc',
        'kvzip',
        'dac',
        'lrp-qa',
        'attn',
        'ppl',
    ]
    new_names = [
        'Contrast-QA',
        'EHPC',
        'KVzip-PC',
        'DAC',
        'LRP-QA',
        'Attn',
        'PPL',
    ]
    hex_colors = [
        '#3D6B98',
        '#598EB2',
        '#7FB5CD',
        '#A4D9E1',
        '#B8E3E8',
        '#CCEFF2',
        '#E0FAFC',
    ]
    FONT_SIZES = {
        'title': 20,
        'label': 20,
        'tick': 20,
        'legend': 20,
    }
    plt.rcParams['font.family'] = 'Times New Roman'

    rename_map = dict(zip(target_methods, new_names))
    method_order = [rename_map[m] for m in target_methods]
    palette = dict(zip(method_order, hex_colors))

    # df_sum = df_sum[df_sum['method'].isin(target_methods)]
    df_det = df_det[df_det['method'].isin(target_methods)]

    # df_sum['method'] = df_sum['method'].replace(rename_map)
    df_det['method'] = df_det['method'].replace(rename_map)

    pic_dir = f'exp_pics/dif_ratio/{budget_policy}'
    os.makedirs(pic_dir, exist_ok=True)

    def extract_ratio(s):
        if not isinstance(s, str):
            return None
        m = re.search(r'r(\d+(\.\d+)?)', s)
        return float(m.group(1)) if m else None

    # NOTE 1. Summary Plot
    # df_sum['ratio'] = df_sum['paras'].apply(extract_ratio)
    # df_sum = df_sum.dropna(subset=['ratio']).sort_values(by='ratio')
    #
    # categories = list(target_order.keys())
    # sns.set_theme(style="whitegrid")
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # axes = axes.flatten()
    #
    # handles, labels = [], []
    # for i, cat in enumerate(categories):
    #     if i < len(axes):
    #         if cat in df_sum.columns:
    #             sns.lineplot(data=df_sum, x='ratio', y=cat, hue='method', style='method',
    #                          markers=True, dashes=False, ax=axes[i], markersize=12, linewidth=2)
    #
    #             if not df_sum_origin.empty and cat in df_sum_origin.columns:
    #                 origin_val = df_sum_origin[cat].mean()
    #                 if pd.notna(origin_val):
    #                     axes[i].axhline(y=origin_val, color='gray', linestyle='--', label='Original', linewidth=2)
    #
    #             axes[i].set_title(cat, fontsize=FONT_SIZES['title'], fontweight='bold')
    #             axes[i].set_xlabel("Compression Ratio", fontsize=FONT_SIZES['label'])
    #             axes[i].set_ylabel("Score", fontsize=FONT_SIZES['label'])
    #             axes[i].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    #             axes[i].grid(True, linestyle='--', alpha=0.7)
    #
    #             for spine in axes[i].spines.values():
    #                 spine.set_edgecolor('black')
    #                 spine.set_alpha(1.0)
    #
    #             if not handles:
    #                 h, l = axes[i].get_legend_handles_labels()
    #                 handles.extend(h)
    #                 labels.extend(l)
    #
    #             axes[i].get_legend().remove()
    #         else:
    #             axes[i].set_visible(False)
    #
    # for j in range(len(categories), len(axes)):
    #     axes[j].set_visible(False)
    # """
    # | 参数     | 数值   | 含义                   |
    # | ------ | ---- | -------------------- |
    # | left   | 0.05 | 左侧留出 5% 空间           |
    # | bottom | 0    | 底部不额外留空间             |
    # | right  | 1    | 右侧到最右边               |
    # | top    | 0.85 | 顶部只用到 85%，上方留 15% 空白 |
    # """
    # plt.tight_layout(rect=[0, 0, 1, 0.85])
    #
    # if handles:
    #     n_items = len(labels)
    #     ncol = n_items
    #     if n_items > 6:
    #         ncol = (n_items + 1) // 2
    #
    #     legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
    #                         ncol=ncol, fontsize=FONT_SIZES['legend'], frameon=False, )
    #     legend.get_frame().set_edgecolor('black')
    #     legend.get_frame().set_linewidth(1.0)
    #     # for text in legend.get_texts():
    #     #     text.set_fontweight('bold')
    #
    # save_path = os.path.join(pic_dir, 'summary_metrics.pdf')
    # plt.savefig(save_path, dpi=600, format='pdf')
    # plt.close()
    # print(f"Saved summary plot to {save_path}")

    # NOTE 2. Detailed Plot
    df_det['ratio'] = df_det['paras'].apply(extract_ratio)
    df_det = df_det.dropna(subset=['ratio']).sort_values(by='ratio')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    categories = [
        'Single-Document QA',
        'Multi-Document QA',
        # 'Summarization',
        # 'Few-shot Learning'
    ]

    handles, labels = [], []
    for row_idx, cat in enumerate(categories):  # 大类
        datasets = target_order.get(cat, [])

        for col_idx in range(len(datasets)):
            ax = axes[row_idx, col_idx]
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # 增加 x / y 轴主刻度数量

            dataset = datasets[col_idx]
            if dataset in df_det.columns:
                sns.lineplot(
                    data=df_det,
                    x='ratio',
                    y=dataset,
                    hue='method',
                    style='method',
                    hue_order=method_order,
                    style_order=method_order,
                    markers=True,
                    dashes=False,
                    ax=ax,
                    markersize=12,
                    linewidth=2,
                    palette=palette,
                )

                if not df_det_origin.empty and dataset in df_det_origin.columns:
                    origin_val = df_det_origin[dataset].mean()
                    if pd.notna(origin_val):
                        ax.axhline(y=origin_val, color='gray', linestyle='--', label='Original', linewidth=2)

                ax.set_title(DNs[dataset], fontsize=FONT_SIZES['title'], )  # fontweight='bold'
                ax.set_xlabel("Compress Ratio", fontsize=FONT_SIZES['label'])
                ax.set_ylabel("Score (%)", fontsize=FONT_SIZES['label'])
                ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
                ax.grid(True, linestyle='--', alpha=0.9)

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_alpha(1.0)

                if not handles:
                    h, l = ax.get_legend_handles_labels()
                    handles.extend(h)
                    labels.extend(l)

                ax.get_legend().remove()
            else:
                ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Make space for left labels and top legen
    n_items = len(labels)
    ncol = n_items
    if n_items > 6:
        ncol = (n_items + 1) // 2
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                        ncol=ncol, fontsize=FONT_SIZES['legend'],
                        frameon=True)  # ：legend 有背景框（边框 + 背景）
    # for text in legend.get_texts():
    #     text.set_fontweight('bold')
    frame = legend.get_frame()
    frame.set_facecolor('none')  # 去掉背景
    frame.set_edgecolor('black')  # 只留边框（高级）

    save_path = os.path.join(pic_dir, 'dif_ratio_detailed.pdf')
    plt.savefig(save_path, dpi=600, format='pdf')
    plt.close()
    print(f"Saved unified detailed plot to {save_path}")


def view_pred(filter_dict, dataset):
    root_dir = 'pred'
    exp_dirs = [x for x in os.listdir(root_dir) if not x.endswith('xlsx')]

    for out_dir in exp_dirs:
        bench = out_dir.split('_')[0]
        model_name = out_dir.split('_')[1]
        method_str = out_dir.split('_')[2]
        method_paras = '_'.join(out_dir.split('_')[3:])

        cont = False
        for v, cv in zip(filter_dict.values(), [bench, model_name, method_str, method_paras, out_dir]):
            if v and not v in cv:
                cont = True
        if cont:
            continue

        all_files = os.listdir(f"{root_dir}/{out_dir}/result")

        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue

            if not filename.startswith(dataset):
                continue

            predictions = []
            with open(f"{root_dir}/{out_dir}/result/{filename}", "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        predictions.append(data)
                    except Exception as e:
                        pass

            # Note: orig_data load_dataset might fail if path is invalid, but keeping as is
            # orig_data = load_dataset(f"/data/hf/THUDM/LongBench", filter_dict['dataset'], split="test")
            print(f"Loaded {len(predictions)} predictions for {filename}")


def view_different_compressor():
    data = []
    data_0_5B = [
        ["p-contrast-qa", 30.7, 37.22, 22.64, 61.36, 37.98],
        ["dac", 27.23, 25.98, 22.47, 55.44, 32.78],
        ["kvzip", 22.66, 25.32, 21.55, 57.36, 31.72],
        ["ppl", 23.64, 24.78, 22.21, 53.36, 31],
        ["ehpc", 21.16, 24.51, 22.07, 50.61, 29.59],
        ["attn", 27.25, 26.89, 23.38, 60.35, 34.47]
    ]
    for row in data_0_5B:
        data.append(["0.5B"] + row)
    data_1_5B = [
        ["p-contrast-qa", 33.2, 39.25, 22.61, 63.66, 39.68],
        ["dac", 28.07, 25.85, 22.35, 56.44, 29.02],
        ["kvzip", 23.54, 24.66, 21.81, 57.2, 31.8],
        ["ppl", 22.45, 23.25, 21.95, 53.08, 30.18],
        ["ehpc", 25.39, 28.88, 22.73, 51.04, 32.01],
        ["attn", 27.94, 28.18, 23.17, 58.04, 34.33]
    ]
    for row in data_1_5B:
        data.append(["1.5B"] + row)
    data_3B = [
        ["p-contrast-qa", 31.37, 39.52, 22.52, 63.26, 39.17],
        ["dac", 25.14, 25.24, 21.45, 57.34, 24.26],
        ["kvzip", 22.31, 24.29, 21.76, 59.91, 32.07],
        ["ppl", 22.52, 23.18, 21.91, 53.02, 30.16],
        ["ehpc", 24.04, 27.69, 22.61, 53.35, 31.92],
        ["attn", 27.42, 27.53, 23.28, 58.22, 34.11]
    ]
    for row in data_3B:
        data.append(["3B"] + row)
    data_7B = [
        ["p-contrast-qa", 33.57, 43.42, 23.89, 62.58, 37.12],
        ["dac", 26.93, 26.57, 24.05, 58.99, 34.13],
        ["kvzip", 23.35, 24.45, 21.44, 57.87, 31.78],
        ["ppl", 23.06, 22.51, 22.43, 52.14, 30.03],
        ["ehpc", 21.09, 24.56, 22.83, 47.03, 28.88],
        ["attn", 27.39, 30.13, 24.44, 54.54, 34.12]
    ]
    for row in data_7B:
        data.append(["7B"] + row)

    df = pd.DataFrame(data, columns=[
        "Model Size", "method",
        "Single-Document QA", "Multi-Document QA", "Summarization", "Few-shot Learning", "All"])

    size_map = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3, "7B": 7}
    df["Size Num"] = df["Model Size"].map(size_map)
    df = df.sort_values("Size Num")

    original = {"Single-Document QA": 40.98, "Multi-Document QA": 44.51, "Summarization": 25.07,
                "Few-shot Learning": 66.88}
    zero_shot = {"Single-Document QA": 11.18, "Multi-Document QA": 21.91, "Summarization": 11.29,
                 "Few-shot Learning": 43.71}

    metrics = ["Single-Document QA", "Multi-Document QA"]

    name_map = {
        'p-contrast-qa': 'Contrast-QA',
        'dac': 'DAC',
        'kvzip': 'KVzip-PC',
        'ppl': 'PPL',
        'ehpc': 'EHPC',
        'attn': 'Attn'
    }
    df['method'] = df['method'].replace(name_map)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    hex_colors = [
        '#3D6B98',
        '#598EB2',
        '#7FB5CD',
        '#A4D9E1',
        '#B8E3E8',
        '#CCEFF2',
        '#E0FAFC',
    ]
    palette = dict(zip(name_map.values(), hex_colors))

    FONT_SIZES = {
        'title': 20,
        'label': 20,
        'tick': 20,
        'legend': 20,
    }

    handles, labels = [], []
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(
            data=df, x="Model Size", y=metric, hue="method",
            style="method", markers=True,
            dashes=False, ax=ax, markersize=10, linewidth=2.5,
            palette=palette,
        )

        ax.axhline(y=original[metric], color='gray', linestyle='--', label='Original', linewidth=2)
        ax.axhline(y=zero_shot[metric], color='black', linestyle=':', label='Zero-shot', linewidth=2)

        ax.set_title(metric, fontsize=FONT_SIZES['title'], )  # fontweight='bold'
        ax.set_xlabel("Compressor Model Size", fontsize=FONT_SIZES['label'])
        ax.set_ylabel("Score (%)", fontsize=FONT_SIZES['label'])
        ax.tick_params(labelsize=FONT_SIZES['tick'])

        if i == 1:
            ax.set_ylim(15, 46)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_alpha(1.0)

        if not handles:
            h, l = ax.get_legend_handles_labels()
            handles = h
            labels = l
        ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if handles:
        unique_labels = []
        unique_handles = []
        seen = set()
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)

        n_items = len(unique_labels)
        ncol = (n_items + 1) // 2

        legend = fig.legend(unique_handles, unique_labels, loc='upper center',
                            bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                            fontsize=FONT_SIZES['legend'], frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('none')  # 去掉背景
        frame.set_edgecolor('black')  # 只留边框（高级）

    save_dir = "exp_pics/dif_compressor"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "compressor_comparison.pdf")
    plt.savefig(save_path, dpi=600, format='pdf')
    print(f"Saved plot to {save_path}")


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

    # NOTE 检查预测的文本
    # filter_dict = {
    #     'bench': 'longbench',
    #     'model_name': None,
    #     'method_str': 'contrast',
    #     'method_paras': 't2',
    #     'out_dir': None,
    # }
    # view_pred(filter_dict, dataset='trec')

    # NOTE 绘制 不同压缩率
    # for bp in [
    #     # 'global',
    #     'chunk'
    # ]:
    #     view_different_ratio(bp,
    #                          # reeval=True
    #                          )

    # NOTE 绘制 不同 压缩模型
    # view_different_compressor()
