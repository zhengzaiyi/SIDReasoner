"""Quick verification: print all "correct" predictions to check for false positives."""

import os, re, json, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SID_TOKEN_RE = re.compile(r"<[abc]_\d+>")  # strict SID pattern only


def load_model_and_tokenizer(model_path, device="cuda:0"):
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


def extract_final_sids(response: str) -> list:
    """Extract SID tokens after the LAST </think> block."""
    parts = response.split("</think>")
    if len(parts) < 2:
        return []
    tail = parts[-1].strip()
    return SID_TOKEN_RE.findall(tail)[:3]


if __name__ == "__main__":
    test_csv = "./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
    max_samples = 200
    device = "cuda:0"

    normal_model_path = "./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint"

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(normal_model_path, device=device)
    prompts_and_targets = build_test_prompts(test_csv, max_samples=max_samples)

    total = len(prompts_and_targets)
    correct = 0
    match_details = []
    partial_match_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for idx, (messages, target_sid) in enumerate(tqdm(prompts_and_targets, desc="Verifying")):
        response = generate_response(model, tokenizer, messages)
        pred_sids = extract_final_sids(response)
        gt_sids = SID_TOKEN_RE.findall(target_sid)[:3]

        n_match = 0
        for i in range(min(len(pred_sids), len(gt_sids))):
            if pred_sids[i] == gt_sids[i]:
                n_match += 1
            else:
                break
        partial_match_counts[n_match] = partial_match_counts.get(n_match, 0) + 1

        is_match = pred_sids == gt_sids and len(pred_sids) == 3

        if is_match:
            correct += 1
            match_details.append({
                "idx": idx,
                "gt": gt_sids,
                "pred": pred_sids,
                "response_tail": response.split("</think>")[-1].strip()[:200],
            })

        if idx < 5:
            print(f"\n[Sample {idx}]")
            print(f"  GT  : {gt_sids}")
            print(f"  Pred: {pred_sids}")
            print(f"  Match: {is_match} (prefix_match={n_match})")
            resp_tail = response.split("</think>")[-1].strip()[:150]
            print(f"  Tail: {resp_tail}")

    print(f"\n{'='*60}")
    print(f"Total: {total}")
    print(f"Exact match (all 3): {correct}/{total} = {100*correct/total:.1f}%")
    print(f"Partial match distribution: {partial_match_counts}")
    print(f"  0 match: {partial_match_counts.get(0,0)} ({100*partial_match_counts.get(0,0)/total:.1f}%)")
    print(f"  1 match: {partial_match_counts.get(1,0)} ({100*partial_match_counts.get(1,0)/total:.1f}%)")
    print(f"  2 match: {partial_match_counts.get(2,0)} ({100*partial_match_counts.get(2,0)/total:.1f}%)")
    print(f"  3 match: {partial_match_counts.get(3,0)} ({100*partial_match_counts.get(3,0)/total:.1f}%)")

    if match_details:
        print(f"\n--- All exact matches ---")
        for m in match_details:
            print(f"  [{m['idx']}] GT={m['gt']}  Pred={m['pred']}  Tail={m['response_tail'][:100]}")
