# Prompt Compression on LongBench

长上下文带来的推理和记忆成本，是当前大模型在真实应用中面临的关键瓶颈之一。本仓库基于 LLMLingua 系列方法，对 LongBench / LongBench-E / InfiniteBench 等基准数据集上的提示词进行压缩，并提供了完整的压缩与评测脚本，方便复现与扩展。

核心脚本包括：
- `longbench/compress.py`：对基准数据集进行提示压缩，生成压缩后的 JSONL 文件
- `longbench/eval.py`：读取模型预测结果，对 LongBench 系列任务进行统一评测与可视化
- `longbench/llmlingua/`：基于 LLMLingua 的多种压缩器实现与封装

如果你在研究或工程中关注长上下文推理效率、本地部署和压缩方案对性能的影响，本项目可以作为一个开箱即用、易于改造的基线系统。

## 特性亮点

- 覆盖多种提示压缩策略：包含 LLMLingua、LLMLingua-2、LongLLMLingua，以及 DAC、EHPC、KVzip-PC、LRP-QA、Attn、PPL 等多种方法
- 同时支持 query-agnostic 与 query-aware 场景：既可以对纯上下文进行压缩，也可以结合任务和问题进行有任务感知的压缩
- 针对 LongBench / LongBench-E / InfiniteBench 的 prompt 模板与长度配置已调好，直接运行即可复现基准结果
- 提供详细的评测脚本和可视化工具，自动汇总各任务、各大类以及整体平均分，并导出为 Excel 表格
- 支持对不同压缩方法/压缩率/压缩模型规模进行横向比较，辅助研究分析

## 环境依赖

建议环境：
- Python 3.9+
- 一块支持 CUDA 的 GPU（用于加载 Qwen2.5 等大模型）

主要 Python 依赖（完整列表见 `requirements.txt`）：
- torch
- transformers (推荐版本 4.38.2 左右)
- datasets
- numpy, pandas
- matplotlib, seaborn
- tqdm
- fuzzywuzzy 等用于文本相似度计算的工具
- 本仓库中自带的 `longbench/llmlingua` 作为本地 LLMLingua 实现

安装示例：

```bash
pip install -r requirements.txt
```

## 数据准备

`longbench/consts.py` 中默认的数据根目录为：

```python
data_dir = "/data/hf"
```

请确保：
- LongBench 数据集放在类似 `data_dir/THUDM/LongBench` 的目录结构下
- InfiniteBench 等其他数据集路径与 `consts.py` 中的默认配置保持一致

如果你的数据目录不同，可以：
- 直接修改 `consts.py` 中的 `data_dir`，或者
- 在自行调用数据加载逻辑时保持相同的数据组织方式

## 压缩脚本：`longbench/compress.py`

该脚本负责对指定的基准数据集进行提示压缩，并将结果保存为 JSONL 文件。核心入口函数为 `run_compress`。

常用参数：
- `--method`：压缩方法名称，例如 `kvzip`、`ehpc`、`dac`、`p-contrast-qa`、`rollout-qa` 等
- `--model`：压缩器所使用的底层模型路径，例如 `/data/hf/Qwen/Qwen2.5-7B-Instruct`
- `--chunk_size`：长文本切块大小
- `--contrast_alpha`：对比学习类方法的融合系数
- `--rollout_m`：rollout 步数
- `--target_token`：压缩后目标 token 数
- `--compress_rate`：压缩率（与 `target_token` 二选一）
- `--bench`：基准名称，支持 `longbench`、`longbench-e`
- `--dataset`：具体数据集名称，例如 `narrativeqa`、`qasper` 等
- `--data_dir`：数据根目录
- `--save_dir`：压缩结果保存目录

脚本运行时会自动将结果保存在：

```text
{model_name}_compressed/{bench}/{dataset}_{method_str}.jsonl
```

其中 `method_str` 由 `consts.get_method_para` 生成，包含方法名、chunk_size、contrast_alpha 等关键信息。

示例：使用 KVzip-PC 方法压缩 LongBench 上的 `narrativeqa`：

```bash
python longbench/compress.py \
  --method kvzip \
  --model /data/hf/Qwen/Qwen2.5-7B-Instruct \
  --bench longbench \
  --dataset narrativeqa \
  --target_token 2000 \
  --chunk_size 1024 \
  --contrast_alpha 1.0 \
  --rollout_m 1
```

脚本会自动跳过已压缩且包含评分信息的数据，并对缺失或损坏的文件进行清理。

### 合成压缩分数：`synthetic_compress`

`compress.py` 中额外提供了 `synthetic_compress` 函数，用于将不同方法的 token 重要性分数按权重线性融合，得到“合成”压缩器。例如：
- 将 `p-contrast-qa` 与 `lrp-qa` 的分数按照设定的 `alpha` 加权求和
- 重新分块后写出新的 JSONL 结果，方法名中会包含权重组合信息

这对于研究“多种压缩策略组合效果”非常有用。

## 评测脚本：`longbench/eval.py`

评测脚本负责读取推理结果 JSONL 文件，对不同任务进行打分，并按任务类别和整体平均分进行汇总，可选择导出为详细和摘要版 Excel 表。

评测流程概览：
1. 从 `{compress_model_name}_pred/` 目录中遍历实验文件夹，例如：
   - `longbench_Qwen2.5-7B-Instruct_kvzip_cs1024_ca1.0_t2000_chunk`
2. 在对应子目录下读取各数据集的预测结果 JSONL 文件
3. 按数据集名称调用相应的 metrics 函数（见 `metrics.py`），计算 F1、ROUGE、分类准确率、检索召回率、代码相似度等
4. 按任务类别（Single-Document QA, Multi-Document QA, Summarization 等）和整体任务求平均
5. 将详细结果和汇总结果分别保存到 `excel_log/` 目录下

你可以直接运行：

```bash
python longbench/eval.py
```

在脚本末尾的 `if __name__ == "__main__":` 部分，可以修改：
- `filter_dict`：筛选特定基准、模型规模、方法和超参数
- `bench_type`：选择 `longbench` 或 `longbench-e`
- `target`：只评测某一个数据集，默认为 `None` 表示全量评测

脚本还提供了若干可视化函数：
- `view_different_ratio`：对比不同压缩率下的性能曲线
- `view_different_compressor`：对比不同压缩模型规模和压缩方法的表现
- `view_pred`：快速查看某个实验配置下的原始预测文本

图像会保存到 `exp_pics/` 目录下，支持论文绘图级别的精度和美观度设置。

## 压缩器实现：`longbench/llmlingua/`

`longbench/llmlingua` 目录包含了针对不同压缩策略的统一接口封装，入口为：

- `llmlingua/__init__.py`：导出 `PromptCompressor`、`DACPromptCompressor`、`EHPCPromptCompressor`、`LRPPromptCompressor` 等
- `llmlingua/llmlingua.py`：LLMLingua 基础实现
- `llmlingua/dac.py`、`llmlingua/ehpc.py`、`llmlingua/lrp.py` 等：对应的具体压缩器实现

在 `longbench/compress.py` 中，通过 `CompressorFactory` 将方法名映射到具体压缩器：
- `llmlingua`、`llmlingua-2`、`longllmlingua` 对应 `PromptCompressor`
- `dac`、`attn`、`ppl` 对应 `DACPromptCompressor`
- `ehpc`、`kvzip`、`p-contrast-qa`、`rollout-qa` 等对应 `EHPCPromptCompressor`
- `lrp-qa` 对应 `LRPPromptCompressor`

如果你希望加入新的压缩方法，只需要：
1. 在 `llmlingua/` 中实现新的压缩器类
2. 在 `__init__.py` 中导出该类
3. 在 `CompressorFactory` 中添加方法名到该类的映射逻辑

即可无缝接入现有的数据处理与评测流程。

## 项目结构示意

```text
.
├── longbench/
│   ├── compress.py           # 提示压缩主脚本
│   ├── eval.py               # 评测与可视化脚本
│   ├── consts.py             # 数据集列表、prompt 模板、长度配置等
│   ├── metrics.py            # 不同任务的评测指标实现
│   └── llmlingua/            # 各类压缩方法的统一封装与实现
├── requirements.txt
└── README.md
```

## 使用建议与扩展方向

- 作为论文或项目的压缩基线，对比不同压缩算法在 LongBench 上的效果
- 在本地或服务器上快速复现 LLMLingua 系列方法，并结合 Qwen2.5 等开源模型进行实验
- 根据自己的任务设计新的 prompt 模板和压缩策略，只需复用 `CompressorFactory` 和评测脚本即可
- 将本仓库作为长上下文系统优化的起点，在生产环境中验证压缩对吞吐和成本的收益

如果本仓库对你的研究或项目有帮助，欢迎在 GitHub 上点亮 Star，并在论文或报告中注明引用。期待你的反馈与改进建议。

