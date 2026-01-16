import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from ehpc import EHPCPromptCompressor


class EHPCAnalyzer(EHPCPromptCompressor):
    def analyze_layers(self, output_dir, max_samples=200):
        configs = ["4k"]
        langs = ["en"]

        evidence_sum = None

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                return

        print(f"Starting analysis for model: {self.model_name}")
        print(f"Results will be saved to: {output_dir}")

        for config in configs:
            for lang in langs:
                print(f"Loading dataset: config={config}, split={lang}")
                try:
                    # Try loading from local path as in original code
                    ds = load_dataset("/data/hf/ameyhengle/Multilingual-Needle-in-a-Haystack", config, split=lang)
                except Exception:
                    print("Could not load from local path, trying HuggingFace Hub...")
                    try:
                        ds = load_dataset("ameyhengle/Multilingual-Needle-in-a-Haystack", config, split=lang)
                    except Exception as e:
                        print(f"Failed to load dataset: {e}")
                        continue

                cnt = 0
                for sample in tqdm(ds, desc="Processing samples"):
                    cnt += 1
                    if cnt < 11:
                        continue
                    if cnt >= max_samples:
                        break

                    prompt = sample.get("prompt", "")
                    ans = sample.get("answer_sentence", "")
                    start = sample.get("answer_start_index", 0)

                    self.model.eval()
                    with torch.no_grad():
                        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                        input_ids = inputs["input_ids"].to(self.device)

                        if input_ids.size(-1) > self.max_position_embeddings:
                            input_ids = input_ids[:, -self.max_position_embeddings:]

                        outputs = self.model(
                            input_ids=input_ids,
                            output_attentions=True,
                            return_dict=True,
                        )
                    attentions = outputs.attentions  # 24 (1,14,len,len)

                    ans_ids = self.tokenizer(ans, return_tensors="pt", add_special_tokens=False)["input_ids"][
                        0].tolist()
                    needle_range = list(range(start, start + len(ans_ids)))
                    last_idx = input_ids.size(-1) - 1

                    L = len(attentions)  # layer_num
                    H = attentions[0].size(1)  # head_num

                    if evidence_sum is None:
                        evidence_sum = torch.zeros((L, H), dtype=torch.float32, device='cpu')

                    for l in range(L):
                        attn = attentions[l][0]
                        for h in range(H):
                            # Boundary check for needle_range
                            valid_needle_range = [idx for idx in needle_range if idx < attn.size(-1)]
                            if not valid_needle_range:
                                continue

                            score = attn[h][last_idx][valid_needle_range].sum()
                            evidence_sum[l, h] += score.detach().cpu()

                    del outputs, attentions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if evidence_sum is None:
            print("No evidence collected. Please check dataset or model.")
            return

        # Calculate layer scores (max over heads) for line plot
        layer_scores, _ = torch.max(evidence_sum, dim=1)
        layer_scores = layer_scores.numpy()
        
        model_basename = os.path.basename(self.model_name).replace('/', '_') or "model"

        # 1. Plot Layer Importance Line Chart
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(layer_scores)), layer_scores, marker='o', linestyle='-', color='b')
        plt.title(f"Layer Importance Scores - {os.path.basename(self.model_name)}")
        plt.xlabel("Layer Index")
        plt.ylabel("Importance Score (Max Head Score)")
        plt.grid(True)
        
        layer_save_path = os.path.join(output_dir, f"{model_basename}_layer_importance.png")
        plt.savefig(layer_save_path)
        print(f"Layer importance plot saved to {layer_save_path}")
        plt.close()

        # 2. Process Head Importance for Heatmap (Sorted per layer)
        # evidence_sum shape: (Layer, Head)
        # Sort descending for each layer
        sorted_evidence_sum, _ = torch.sort(evidence_sum, descending=True, dim=1)
        sorted_evidence_np = sorted_evidence_sum.numpy()

        # Plotting Heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(sorted_evidence_np, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Importance Score')
        
        plt.title(f"Layer-Head Importance Heatmap (Sorted) - {os.path.basename(self.model_name)}")
        plt.xlabel("Head Rank (Sorted by Importance)")
        plt.ylabel("Layer Index")
        
        heatmap_save_path = os.path.join(output_dir, f"{model_basename}_head_importance_heatmap.png")
        plt.savefig(heatmap_save_path)
        print(f"Head importance heatmap saved to {heatmap_save_path}")
        plt.close()
        
        # # Save raw data
        # np.save(os.path.join(output_dir, f"{model_basename}_layer_scores.npy"), layer_scores)
        # np.save(os.path.join(output_dir, f"{model_basename}_sorted_head_scores.npy"), sorted_evidence_np)
        # print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze EHPC Layer Importance")
    parser.add_argument("--model_name", type=str, default= '/data/hf/Qwen/Qwen2.5-7B-Instruct',
                        help="Path or name of the model")
    parser.add_argument("--output_dir", type=str, default="/data/mxl/PC/longbench/exp_pics/ehpc",
                        help="Directory to save results")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to process")

    args = parser.parse_args()

    # Handle Windows path for default or provided Linux-style path
    output_dir = args.output_dir
    if os.name == 'nt':
        # If it looks like a Linux absolute path (starts with /), try to map it to current drive
        if output_dir.startswith('/'):
            current_drive = os.getcwd()[0]
            output_dir = f"{current_drive}:{output_dir}"
        output_dir = output_dir.replace('/', '\\')

    analyzer = EHPCAnalyzer(model_name=args.model_name)
    analyzer.analyze_layers(output_dir, max_samples=args.max_samples)
