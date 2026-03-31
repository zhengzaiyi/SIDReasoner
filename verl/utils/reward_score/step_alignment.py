from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SID_TOKEN_PATTERN = re.compile(r"<[^>]+>")


def extract_sid_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return SID_TOKEN_PATTERN.findall(text)


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {token!r} to map to one token, got {token_ids}.")
    return int(token_ids[0])


def _resolve_item_info_path(item_info_path: str) -> str:
    path = Path(item_info_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Item info path not found: {path}")
    return str(path)


class RecommendationStepAligner:
    """Shared SID/think alignment utilities for rollout and reward."""

    def __init__(self, tokenizer, item_info_path: str):
        self.tokenizer = tokenizer
        self.item_info_path = _resolve_item_info_path(item_info_path)

        self.think_start_token_id = _single_token_id(tokenizer, "<think>")
        self.think_end_token_id = _single_token_id(tokenizer, "</think>")

        self.valid_sid_sequences: list[tuple[str, ...]] = []
        self.sid_length = 0
        self.prefix_to_allowed_tokens: dict[tuple[str, ...], list[str]] = {}
        self.prefix_to_allowed_token_ids: dict[tuple[str, ...], list[int]] = {}
        self.position_tokens: dict[int, list[str]] = {}
        self.position_token_ids: dict[int, list[int]] = {}
        self.sid_token_to_id: dict[str, int] = {}
        self.sid_token_id_to_token: dict[int, str] = {}
        self._load_item_info()

    def _load_item_info(self) -> None:
        prefix_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
        position_sets: dict[int, set[str]] = defaultdict(set)
        all_tokens: set[str] = set()

        with open(self.item_info_path, "r", encoding="utf-8") as f:
            for line in f:
                semantic_id = line.split("\t", 1)[0].strip()
                sid_tokens = tuple(extract_sid_tokens(semantic_id))
                if not sid_tokens:
                    continue

                if self.sid_length == 0:
                    self.sid_length = len(sid_tokens)
                elif len(sid_tokens) != self.sid_length:
                    raise ValueError(
                        "Mixed SID lengths are not supported for step alignment: "
                        f"expected {self.sid_length}, got {len(sid_tokens)} in {semantic_id!r}."
                    )

                self.valid_sid_sequences.append(sid_tokens)
                all_tokens.update(sid_tokens)
                for position, token in enumerate(sid_tokens):
                    prefix_sets[sid_tokens[:position]].add(token)
                    position_sets[position].add(token)

        if self.sid_length == 0:
            raise ValueError(f"No valid SID sequences found in {self.item_info_path}")

        for token in sorted(all_tokens):
            token_id = _single_token_id(self.tokenizer, token)
            self.sid_token_to_id[token] = token_id
            self.sid_token_id_to_token[token_id] = token

        for prefix, tokens in prefix_sets.items():
            ordered_tokens = sorted(tokens)
            self.prefix_to_allowed_tokens[prefix] = ordered_tokens
            self.prefix_to_allowed_token_ids[prefix] = [self.sid_token_to_id[token] for token in ordered_tokens]

        for position, tokens in position_sets.items():
            ordered_tokens = sorted(tokens)
            self.position_tokens[position] = ordered_tokens
            self.position_token_ids[position] = [self.sid_token_to_id[token] for token in ordered_tokens]

    def sid_length_from_ground_truth(self, ground_truth: str | None) -> int:
        ground_truth_sid = extract_sid_tokens(ground_truth)
        return len(ground_truth_sid) if ground_truth_sid else self.sid_length

    def get_ground_truth_sid_tokens(self, ground_truth: str | None) -> list[str]:
        sid_length = self.sid_length_from_ground_truth(ground_truth)
        return extract_sid_tokens(ground_truth)[:sid_length]

    def find_think_end_positions(self, response_token_ids: Sequence[int]) -> list[int]:
        return [idx for idx, token_id in enumerate(response_token_ids) if int(token_id) == self.think_end_token_id]

    def extract_predicted_sid_tokens(
        self,
        response_token_ids: Sequence[int],
        expected_sid_length: int | None = None,
    ) -> list[str] | None:
        sid_length = expected_sid_length or self.sid_length
        think_end_positions = self.find_think_end_positions(response_token_ids)
        tail_start = think_end_positions[-1] + 1 if think_end_positions else 0
        tail_sid_tokens = [
            self.sid_token_id_to_token[int(token_id)]
            for token_id in response_token_ids[tail_start:]
            if int(token_id) in self.sid_token_id_to_token
        ]
        if len(tail_sid_tokens) < sid_length:
            return None
        return tail_sid_tokens[:sid_length]

    def allowed_token_ids_for_prefix(self, sid_prefix: Sequence[str], position: int) -> list[int]:
        prefix = tuple(sid_prefix[:position])
        allowed = self.prefix_to_allowed_token_ids.get(prefix)
        if allowed:
            return allowed
        fallback = self.position_token_ids.get(position)
        if fallback:
            return fallback
        return []

    def sid_tokens_from_token_ids(self, token_ids: Iterable[int]) -> list[str]:
        return [self.sid_token_id_to_token[int(token_id)] for token_id in token_ids if int(token_id) in self.sid_token_id_to_token]

    def is_valid_sid_sequence(self, sid_tokens: Sequence[str]) -> bool:
        if len(sid_tokens) < self.sid_length:
            return False
        for position in range(self.sid_length):
            prefix = tuple(sid_tokens[:position])
            next_token = sid_tokens[position]
            if next_token not in self.prefix_to_allowed_tokens.get(prefix, []):
                return False
        return True


def compute_step_aligned_reward(
    response_token_ids: Sequence[int],
    ground_truth: str | None,
    aligner: RecommendationStepAligner,
    match_reward: float = 1.0,
    format_reward: float = 0.0,
    require_exact_think_blocks: bool = True,
) -> dict:
    sid_length = aligner.sid_length_from_ground_truth(ground_truth)
    ground_truth_sid = aligner.get_ground_truth_sid_tokens(ground_truth)
    think_end_positions = aligner.find_think_end_positions(response_token_ids)
    predicted_sid = aligner.extract_predicted_sid_tokens(response_token_ids, expected_sid_length=sid_length)

    exact_block_match = len(think_end_positions) == sid_length
    has_required_blocks = len(think_end_positions) >= sid_length
    sid_valid = predicted_sid is not None and aligner.is_valid_sid_sequence(predicted_sid)

    step_rewards = [0.0] * sid_length
    if predicted_sid is not None and ground_truth_sid and (exact_block_match or (has_required_blocks and not require_exact_think_blocks)):
        for index in range(min(sid_length, len(predicted_sid), len(ground_truth_sid))):
            if predicted_sid[index] == ground_truth_sid[index]:
                step_rewards[index] = float(match_reward)

    format_bonus = float(format_reward) if (sid_valid and exact_block_match) else 0.0

    return {
        "score": float(sum(step_rewards) + format_bonus),
        "step_rewards": step_rewards,
        "format_bonus": format_bonus,
        "predicted_sid": predicted_sid or [],
        "ground_truth_sid": ground_truth_sid,
        "sid_length": sid_length,
        "think_end_positions": think_end_positions,
        "block_count": len(think_end_positions),
        "block_count_match": exact_block_match,
        "sid_valid": sid_valid,
    }
