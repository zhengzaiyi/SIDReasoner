from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_step_aligned_advantage
from verl.utils.reward_score.step_alignment import RecommendationStepAligner, compute_step_aligned_reward
from verl.workers.reward_manager.step_aligned import StepAlignedRewardManager


class DummyTokenizer:
    def __init__(self):
        tokens = [
            "<pad>",
            "<think>",
            "</think>",
            "reason_a",
            "reason_b",
            "reason_c",
            "<a_0>",
            "<a_1>",
            "<b_0>",
            "<b_1>",
            "<b_2>",
            "<c_0>",
            "<c_1>",
            "<c_3>",
            "prompt",
        ]
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.token_to_id[text]]

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if skip_special_tokens and token == "<pad>":
                continue
            tokens.append(token)
        return "".join(tokens)


def build_item_info(tmp_path: Path) -> str:
    item_info_path = tmp_path / "items.txt"
    item_info_path.write_text(
        "\n".join(
            [
                "<a_0><b_0><c_0>\titem-0",
                "<a_0><b_1><c_1>\titem-1",
                "<a_1><b_2><c_3>\titem-2",
            ]
        ),
        encoding="utf-8",
    )
    return str(item_info_path)


def build_response(tokenizer: DummyTokenizer, *tokens: str) -> list[int]:
    return [tokenizer.token_to_id[token] for token in tokens]


class StepAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.tokenizer = DummyTokenizer()
        self.item_info_path = build_item_info(self.tmp_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_step_aligned_reward_exact_match(self):
        aligner = RecommendationStepAligner(tokenizer=self.tokenizer, item_info_path=self.item_info_path)
        response_token_ids = build_response(
            self.tokenizer,
            "<think>",
            "reason_a",
            "</think>",
            "<think>",
            "reason_b",
            "</think>",
            "<think>",
            "reason_c",
            "</think>",
            "<a_0>",
            "<b_0>",
            "<c_0>",
        )

        result = compute_step_aligned_reward(
            response_token_ids=response_token_ids,
            ground_truth="<a_0><b_0><c_0>",
            aligner=aligner,
            match_reward=1.0,
        )

        self.assertEqual(result["score"], 3.0)
        self.assertEqual(result["step_rewards"], [1.0, 1.0, 1.0])
        self.assertEqual(result["think_end_positions"], [2, 5, 8])
        self.assertEqual(result["predicted_sid"], ["<a_0>", "<b_0>", "<c_0>"])
        self.assertTrue(result["block_count_match"])

    def test_compute_step_aligned_reward_requires_exact_block_count(self):
        aligner = RecommendationStepAligner(tokenizer=self.tokenizer, item_info_path=self.item_info_path)
        response_token_ids = build_response(
            self.tokenizer,
            "<think>",
            "reason_a",
            "</think>",
            "<think>",
            "reason_b",
            "</think>",
            "<a_0>",
            "<b_0>",
            "<c_0>",
        )

        result = compute_step_aligned_reward(
            response_token_ids=response_token_ids,
            ground_truth="<a_0><b_0><c_0>",
            aligner=aligner,
            match_reward=1.0,
            require_exact_think_blocks=True,
        )

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["step_rewards"], [0.0, 0.0, 0.0])
        self.assertFalse(result["block_count_match"])

    def test_step_aligned_reward_manager_places_rewards_on_think_boundaries(self):
        manager = StepAlignedRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,
            reward_fn_key="data_source",
            item_info_path=self.item_info_path,
            match_reward=1.0,
        )

        response_token_ids = build_response(
            self.tokenizer,
            "<think>",
            "reason_a",
            "</think>",
            "<think>",
            "reason_b",
            "</think>",
            "<think>",
            "reason_c",
            "</think>",
            "<a_0>",
            "<b_0>",
            "<c_0>",
        )
        response_tensor = torch.tensor([response_token_ids + [self.tokenizer.pad_token_id, self.tokenizer.pad_token_id]])
        prompt_tensor = torch.tensor([[self.tokenizer.token_to_id["prompt"]]])
        attention_mask = torch.tensor([[1] + [1] * len(response_token_ids) + [0, 0]])
        data = DataProto.from_dict(
            tensors={
                "prompts": prompt_tensor,
                "responses": response_tensor,
                "attention_mask": attention_mask,
            },
            non_tensors={
                "data_source": np.array(["rec"], dtype=object),
                "reward_model": np.array([{"ground_truth": "<a_0><b_0><c_0>"}], dtype=object),
            },
        )

        result = manager(data, return_dict=True)
        reward_tensor = result["reward_tensor"][0].tolist()

        self.assertEqual(reward_tensor[2], 1.0)
        self.assertEqual(reward_tensor[5], 1.0)
        self.assertEqual(reward_tensor[8], 1.0)
        self.assertEqual(sum(reward_tensor), 3.0)

    def test_compute_grpo_step_aligned_advantage_maps_step_scores_to_think_spans(self):
        response_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )
        advantages, returns = compute_grpo_step_aligned_advantage(
            token_level_rewards=torch.zeros_like(response_mask),
            response_mask=response_mask,
            index=np.array([0, 0]),
            step_rewards=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=object),
            think_end_positions=np.array([[2, 5, 8], [2, 5, 8]], dtype=object),
        )

        expected_positive = float(1.0 / torch.sqrt(torch.tensor(2.0)))
        self.assertTrue(torch.allclose(advantages[0, 0:3], torch.full((3,), expected_positive)))
        self.assertTrue(torch.allclose(advantages[0, 3:6], torch.full((3,), expected_positive)))
        self.assertTrue(torch.allclose(advantages[0, 9:12], torch.zeros(3)))
        self.assertTrue(torch.allclose(advantages[1, 0:3], -torch.full((3,), expected_positive)))
        self.assertTrue(torch.allclose(returns[0, 0:3], torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
