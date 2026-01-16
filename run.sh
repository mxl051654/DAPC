#!/bin/bash
#sudo renice -5 -p $$

models=(
#    '/data/hf/Qwen/Qwen2.5-0.5B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-1.5B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-3B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-7B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-32B-Instruct'
#    '/data/hf/meta-llama/Llama-3.2-1B-Instruct'
#    '/data/hf/meta-llama/Llama-3.2-3B-Instruct'
#    '/data/hf/meta-llama/Llama-3.1-8B-Instruct'
#    '/data/hf/meta-llama/Llama-4-Scout-17B-16E'
#    '/data/hf/meta-llama/Llama-3.3-70B-Instruct'
#    'CodeLlama-7B'
#    'microsoft/Phi-3.5-mini-instruct'
#    'microsoft/Phi-3.5-mini-3.8B-Instruct'
#    'deepseek-chat'

    '/data/hf/meta-llama/Llama-3.1-8B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-7B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-32B-Instruct'
)

methods=(
#     'original'
     'zero-shot'

#    'llmlingua-2'
#    'llmlingua'
#    'longllmlingua'
#
#    'attn'  # last token attn only
#    'ppl'  # ppl only
#    'dac'  # attn + ppl
#    'attn-qa'
#    'ehpc'  # select heads
#    'kvzip'

#    'p-contrast'
#    'p-contrast-qa'
#    'rollout'
#    'rollout-qa'
#    'contrast-rollout-qa'
#    'rollout-contrast-qa'
#    'lrp-qa'

#    'add-lrp-qa-0.5-p-contrast-qa-0.1'
#    'add-lrp-qa-0.5-p-contrast-qa-0.2'
#    'add-lrp-qa-0.5-p-contrast-qa-0.3'
#    'add-lrp-qa-0.5-p-contrast-qa-0.4'
#    'add-lrp-qa-0.5-p-contrast-qa-0.5'
#    'add-lrp-qa-0.5-p-contrast-qa-0.6'
#    'add-lrp-qa-0.5-p-contrast-qa-0.7'
#    'add-lrp-qa-0.5-p-contrast-qa-0.8'
#    'add-lrp-qa-0.5-p-contrast-qa-0.9'
#    'add-lrp-qa-0.5-p-contrast-qa-1.0'
)

targets=(
  2000
#  3000
)
budget_policies=(
  'chunk'
#  'global'
)
datasets_longbench_e=(
    'qasper' 'multifieldqa_en'
    'hotpotqa' '2wikimqa'
#    'gov_report' 'multi_news'
#    'trec' 'triviaqa' 'samsum'
#    'passage_count' 'passage_retrieval_en'
#    'lcc' 'repobench-p'
)
datasets_longbench=(
     'narrativeqa' 'qasper' 'multifieldqa_en' # Single-Document QA':
     'hotpotqa' '2wikimqa' 'musique'  # 'Multi-Document QA'
     'gov_report' 'qmsum' 'multi_news' # 'Summarization'
     'trec' 'triviaqa' 'samsum'  #  'Few-shot Learning'
#     'passage_count' 'passage_retrieval_en'  # 'Synthetic'
#     'repobench-p' 'lcc'  #  'Code'
)

benches=(
  'longbench'
#  'longbench-e'
#  'infinitebench'
)
PORT=8002

for bench in "${benches[@]}"; do
    if [ "$bench" = "longbench-e" ]; then
      datasets=("${datasets_longbench_e[@]}")
    else
      datasets=("${datasets_longbench[@]}")
    fi

    for bg in "${budget_policies[@]}"; do
        for model in "${models[@]}"; do
            MODEL_NAME=$(basename "${model}")

            for method in "${methods[@]}"; do
                for dataset in "${datasets[@]}"; do
                    echo "Model: ${MODEL_NAME} Method: ${method} Budget: ${bg} Dataset: ${dataset}"

                    if [ "${method}" = "original" ]; then
                        python pred_vllm.py \
                            --bench "${bench}"  \
                            --model "${model}" \
                            --port "${PORT}" \
                            --dataset "${dataset}"

                    else
                        for tgt in "${targets[@]}"; do
                        echo "Target length ${tgt}"

                            python pred_vllm.py \
                                --bench "${bench}" \
                                --model "${model}" \
                                --port "${PORT}" \
                                --method "${method}" \
                                --target_len "${tgt}" \
                                --budget_policy "${bg}" \
                                --dataset "${dataset}"

                        done
                    fi
                done
            done
        done
    done
done


