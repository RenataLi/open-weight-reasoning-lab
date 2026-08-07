from __future__ import annotations

from typing import Any


def trl_completion_text(tokenizer: Any, completion: str) -> str:
    """Mirror TRL's non-conversational DPO preprocessing for completion EOS."""

    eos_token = tokenizer.eos_token
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("Preference tokenization requires a non-empty EOS token")
    return completion if completion.endswith(eos_token) else completion + eos_token


def trl_preference_token_ids(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
) -> tuple[list[int], list[int], list[int]]:
    """Return prompt, completion, and full ids exactly as TRL DPO does for strings."""

    completion_with_eos = trl_completion_text(tokenizer, completion)
    prompt_ids = list(tokenizer(prompt)["input_ids"])
    full_ids = list(tokenizer(prompt + completion_with_eos)["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer boundary mismatch for preference prompt/completion")
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("Preference completion has no tokens after TRL preprocessing")
    return prompt_ids, completion_ids, full_ids


def trl_preference_lengths(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
) -> dict[str, int]:
    prompt_ids, completion_ids, full_ids = trl_preference_token_ids(
        tokenizer,
        prompt=prompt,
        completion=completion,
    )
    return {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "sequence_tokens": len(full_ids),
    }
