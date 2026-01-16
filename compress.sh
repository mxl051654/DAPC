
datasets_longbench=(
     'narrativeqa' 'qasper' 'multifieldqa_en' # Single-Document QA':
     'hotpotqa' '2wikimqa' 'musique'  # 'Multi-Document QA'
     'gov_report' 'qmsum' 'multi_news' # 'Summarization'
     'trec' 'triviaqa' 'samsum'  #  'Few-shot Learning'
#     'passage_count' 'passage_retrieval_en'  # 'Synthetic'
#     'repobench-p' 'lcc'  #  'Code'
)
datasets_longbench_e=(
    'qasper' 'multifieldqa_en'
    'hotpotqa' '2wikimqa'
#    'gov_report' 'multi_news'
#    'trec' 'triviaqa' 'samsum'
#    'passage_count' 'passage_retrieval_en'
#    'lcc' 'repobench-p'
)
datasets_infinitebench=(
#    "passkey" "number_string" "kv_retrieval"
#    "longbook_sum_eng"
    "longbook_choice_eng" "longbook_qa_eng"
    "longbook_qa_chn"
#    "longdialogue_qa_eng"
#    "math_find" "math_calc"
#    "code_run" "code_debug"
)

methods=(
#    'original'

#    'llmlingua-2'
#    'llmlingua'
#    'longllmlingua'

#    'attn'  
#     'attn-qa' 
#    'ppl'  
#    'dac'  
#    'ehpc'  
    'kvzip'

#    'p-contrast'
#    'p-contrast-qa'
#    'rollout'
#    'rollout-qa'
#    'contrast-rollout-qa'
#    'rollout-contrast-qa'
    'lrp-qa'
)
targets=(
  2000
#  3000
)
benches=(
#  'longbench'
  'longbench-e'
#  'infinitebench'
)

models=(
    '/data/hf/Qwen/Qwen2.5-7B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-3B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-1.5B-Instruct'
#    '/data/hf/Qwen/Qwen2.5-0.5B-Instruct'
#    '/data/hf/meta-llama/Llama-3.1-8B-Instruct'
)

export CUDA_VISIBLE_DEVICES=3

for model in "${models[@]}"; do

    for bench in "${benches[@]}"; do
        # NOTE set datasets
        if [ "$bench" = "longbench-e" ]; then
          datasets=("${datasets_longbench_e[@]}")
        elif [ "$bench" = "infinitebench" ]; then
          datasets=("${datasets_infinitebench[@]}")
        else
          datasets=("${datasets_longbench[@]}")
        fi

        for method in "${methods[@]}"; do
            for target in "${targets[@]}"; do
                for dataset in "${datasets[@]}"; do

                    echo "method ${method} target ${target} dataset ${dataset}"
                    python compress.py \
                        --method "${method}" \
                        --model "${model}" \
                        --target_token "${target}" \
                        --bench "${bench}" \
                        --dataset "${dataset}"
                done
            done
        done
    done
done
