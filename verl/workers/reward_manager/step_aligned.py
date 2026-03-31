from __future__ import annotations

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score.step_alignment import RecommendationStepAligner, compute_step_aligned_reward
from verl.workers.reward_manager import register


@register("step_aligned")
class StepAlignedRewardManager:
    """Assign per-SID rewards to the corresponding think-step boundaries."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs):
        item_info_path = reward_kwargs.get("item_info_path")
        if not item_info_path:
            raise ValueError("step_aligned reward manager requires reward_kwargs.item_info_path")

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.compute_score = compute_score
        self.match_reward = float(reward_kwargs.get("match_reward", 1.0))
        self.format_reward = float(reward_kwargs.get("format_reward", 0.0))
        self.require_exact_think_blocks = bool(reward_kwargs.get("require_exact_think_blocks", True))
        self.aligner = RecommendationStepAligner(tokenizer=tokenizer, item_info_path=item_info_path)

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        total_scores = []
        already_printed = {}
        for i in range(len(data)):
            valid_len = int(valid_response_lengths[i].item())
            if valid_len <= 0:
                empty_result = {
                    "score": 0.0,
                    "step_rewards": [],
                    "think_end_positions": [],
                    "predicted_sid": [],
                    "ground_truth_sid": [],
                    "sid_length": 0,
                    "block_count": 0,
                    "block_count_match": False,
                    "sid_valid": False,
                    "format_bonus": 0.0,
                }
                total_scores.append(0.0)
                for key, value in empty_result.items():
                    reward_extra_info[key].append(value)
                continue

            valid_response_ids = data.batch["responses"][i][:valid_len].tolist()
            ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
            result = compute_step_aligned_reward(
                response_token_ids=valid_response_ids,
                ground_truth=ground_truth,
                aligner=self.aligner,
                match_reward=self.match_reward,
                format_reward=self.format_reward,
                require_exact_think_blocks=self.require_exact_think_blocks,
            )

            think_end_positions = list(result["think_end_positions"])
            step_rewards = list(result["step_rewards"])
            if think_end_positions:
                max_assignable = min(len(step_rewards), len(think_end_positions))
                for step_idx in range(max_assignable):
                    reward_tensor[i, think_end_positions[step_idx]] = float(step_rewards[step_idx])
                if result["format_bonus"] and max_assignable > 0:
                    reward_tensor[i, think_end_positions[max_assignable - 1]] += float(result["format_bonus"])

            total_scores.append(float(result["score"]))
            for key, value in result.items():
                reward_extra_info[key].append(value)

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                valid_prompt_len = int(attention_mask[i][:prompt_len].sum().item())
                prompt_str = self.tokenizer.decode(
                    data.batch["prompts"][i][-valid_prompt_len:],
                    skip_special_tokens=False,
                )
                response_str = self.tokenizer.decode(data.batch["responses"][i][:valid_len], skip_special_tokens=False)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", result["score"])
                print("[step_rewards]", result["step_rewards"])
                print("[predicted_sid]", result["predicted_sid"])
                print("[think_end_positions]", result["think_end_positions"])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(total_scores, dtype=torch.float32, device=prompt_ids.device)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
