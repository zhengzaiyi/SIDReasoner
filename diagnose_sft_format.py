"""Diagnostic script: test both SFT reasoning activation models on test set.

Checks output format correctness for:
  - Normal model:       expects 1 </think> block  + 3 SID tokens after it
  - Step-aligned model: expects 3 </think> blocks + 3 SID tokens after last block
"""

import os, re, json, sys, argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SID_TOKEN_RE = re.compile(r"<[^>]+>")


def load_model_and_tokenizer(model_path, device="cuda:3"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    sid_index_path = "./data/Amazon/index/Office_Products.index.json"
    if os.path.exists(sid_index_path):
        with open(sid_index_path, "r") as f:
            indices = json.load(f)
        new_tokens = set()
        for sids in indices.values():
            for tok in sids:
                new_tokens.add(tok)
        existing_vocab = set(tokenizer.get_vocab().keys())
        tokens_to_add = [t for t in sorted(new_tokens) if t not in existing_vocab]
        if tokens_to_add:
            tokenizer.add_tokens(tokens_to_add)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


def build_test_prompts(test_csv, max_samples=200):
    df = pd.read_csv(test_csv)
    if max_samples > 0:
        df = df.head(max_samples)
    prompts_and_targets = []
    for _, row in df.iterrows():
        history_sids = eval(row["history_item_sid"])
        history_str = ", ".join(history_sids)
        target_sid = row["item_sid"]

        instruction = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n"
            "Can you recommend the next item for the user based on their interaction history?\n"
        )
        user_content = (
            f"The user has sequentially interacted with items {history_str}. "
            "Can you recommend the next item for him? Let's think step by step "
            "before making recommendation. Directly output the item SID after thinking."
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ]
        prompts_and_targets.append((messages, target_sid))
    return prompts_and_targets


def generate_response(model, tokenizer, messages, max_new_tokens=1024):
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=False)


def analyze_format_normal(response: str) -> dict:
    """Normal model: expect exactly 1 </think> block then >=3 SID tokens."""
    think_end_count = response.count("</think>")
    think_start_count = response.count("<think>")

    match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if match:
        tail = match.group(1).strip()
        sid_tokens = SID_TOKEN_RE.findall(tail)[:3]
    else:
        tail = ""
        sid_tokens = []

    has_think_block = think_start_count >= 1 and think_end_count >= 1
    has_3_sids = len(sid_tokens) == 3
    format_ok = has_think_block and has_3_sids

    return {
        "format_ok": format_ok,
        "think_start_count": think_start_count,
        "think_end_count": think_end_count,
        "sid_tokens_after_think": sid_tokens,
        "tail": tail[:200],
    }


def analyze_format_step_aligned(response: str) -> dict:
    """Step-aligned model: expect exactly 3 </think> blocks then >=3 SID tokens."""
    think_end_count = response.count("</think>")
    think_start_count = response.count("<think>")

    parts = response.split("</think>")
    if len(parts) > 1:
        tail = parts[-1].strip()
        sid_tokens = SID_TOKEN_RE.findall(tail)[:3]
    else:
        tail = ""
        sid_tokens = []

    has_3_blocks = think_end_count == 3 and think_start_count == 3
    has_3_sids = len(sid_tokens) == 3
    format_ok = has_3_blocks and has_3_sids

    return {
        "format_ok": format_ok,
        "think_start_count": think_start_count,
        "think_end_count": think_end_count,
        "sid_tokens_after_think": sid_tokens,
        "tail": tail[:200],
    }


def run_diagnosis(model_path, model_name, test_csv, analyzer_fn,
                  max_samples=200, device="cuda:3"):
    print(f"\n{'='*80}")
    print(f"  Diagnosing: {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Analyzer  : {analyzer_fn.__name__}")
    print(f"{'='*80}\n")

    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    prompts_and_targets = build_test_prompts(test_csv, max_samples=max_samples)

    total = len(prompts_and_targets)
    correct_format = 0
    correct_answer = 0
    incorrect_examples = []
    think_end_count_dist = {}

    for idx, (messages, target_sid) in enumerate(tqdm(prompts_and_targets, desc=model_name)):
        response = generate_response(model, tokenizer, messages, max_new_tokens=1024)
        analysis = analyzer_fn(response)

        cnt = analysis["think_end_count"]
        think_end_count_dist[cnt] = think_end_count_dist.get(cnt, 0) + 1

        if analysis["format_ok"]:
            correct_format += 1
            gt_sids = SID_TOKEN_RE.findall(target_sid)[:3]
            pred_sids = analysis["sid_tokens_after_think"]
            if pred_sids == gt_sids:
                correct_answer += 1
        else:
            if len(incorrect_examples) < 15:
                incorrect_examples.append({
                    "idx": idx,
                    "target_sid": target_sid,
                    "response_head": response[:500],
                    "analysis": analysis,
                })

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results for: {model_name}")
    print(f"{'='*60}")
    print(f"  Total samples       : {total}")
    print(f"  Format correct      : {correct_format}/{total} ({100*correct_format/total:.1f}%)")
    print(f"  Answer correct      : {correct_answer}/{total} ({100*correct_answer/total:.1f}%)")
    print(f"  </think> count dist : {dict(sorted(think_end_count_dist.items()))}")

    if incorrect_examples:
        print(f"\n  --- Incorrect format examples (showing up to 15) ---")
        for ex in incorrect_examples:
            print(f"\n  [Sample {ex['idx']}] target={ex['target_sid']}")
            print(f"    think_start_count={ex['analysis']['think_start_count']}, "
                  f"think_end_count={ex['analysis']['think_end_count']}, "
                  f"sid_tokens={ex['analysis']['sid_tokens_after_think']}")
            resp = ex["response_head"].replace("\n", "\\n")
            print(f"    Response: {resp}")

    del model
    torch.cuda.empty_cache()
    return {
        "total": total,
        "correct_format": correct_format,
        "correct_answer": correct_answer,
        "think_end_dist": think_end_count_dist,
        "incorrect_examples": incorrect_examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",
                        default="./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    normal_model = "./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint"
    step_aligned_model = "./output_dir/Office_Products_stage2_step_aligned_reasoning_activation_Qwen3-1.7B/final_checkpoint"

    print("\n" + "="*80)
    print("  SFT REASONING ACTIVATION FORMAT DIAGNOSTIC")
    print("="*80)

    r1 = run_diagnosis(
        normal_model, "Normal (single-think)",
        args.test_csv, analyze_format_normal,
        max_samples=args.max_samples, device=args.device,
    )

    r2 = run_diagnosis(
        step_aligned_model, "Step-aligned (multi-think)",
        args.test_csv, analyze_format_step_aligned,
        max_samples=args.max_samples, device=args.device,
    )

    print("\n" + "="*80)
    print("  COMPARISON SUMMARY")
    print("="*80)
    print(f"  Normal      : format={100*r1['correct_format']/r1['total']:.1f}%  "
          f"answer={100*r1['correct_answer']/r1['total']:.1f}%  "
          f"</think> dist={dict(sorted(r1['think_end_dist'].items()))}")
    print(f"  Step-aligned: format={100*r2['correct_format']/r2['total']:.1f}%  "
          f"answer={100*r2['correct_answer']/r2['total']:.1f}%  "
          f"</think> dist={dict(sorted(r2['think_end_dist'].items()))}")
