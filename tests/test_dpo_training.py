from __future__ import annotations

import torch
from datasets import Dataset
from trl.data_utils import _tokenize as trl_tokenize
from trl import DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

from reasoning_lab.evaluation.preferences import _encode_batch, _masked_sequence_logps
from reasoning_lab.preference_tokenization import trl_preference_token_ids
from reasoning_lab.training.dpo import _assert_no_truncation, _preference_dataset


def test_installed_trl_supports_preregistered_anchored_loss() -> None:
    config = DPOConfig(
        output_dir="unused",
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, 0.25],
        disable_dropout=True,
        report_to="none",
    )
    assert config.loss_type == ["sigmoid", "sft"]
    assert config.loss_weights == [1.0, 0.25]
    assert config.disable_dropout is True


class _CharacterTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text=None, **kwargs):
        del kwargs
        return {"input_ids": [1, *(ord(char) for char in str(text))]}


def test_preference_dataset_rejects_duplicate_and_identical(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"id":"a","prompt":"P","chosen":" C","rejected":" R"}\n'
        '{"id":"b","prompt":"Q","chosen":" X","rejected":" Y"}\n',
        encoding="utf-8",
    )
    dataset = _preference_dataset(path)
    assert len(dataset) == 2
    assert dataset.column_names == ["id", "prompt", "chosen", "rejected"]

    path.write_text(
        '{"id":"a","prompt":"P","chosen":" C","rejected":" C"}\n',
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="identical"):
        _preference_dataset(path)


def test_masked_sequence_logps_score_only_completion_tokens() -> None:
    # Token probabilities are deliberately easy to inspect. Positions 0 and 1
    # predict labels at positions 1 and 2; only the latter is a completion.
    logits = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [9.0, 9.0, 9.0]]]
    )
    input_ids = torch.tensor([[0, 0, 1]])
    completion_mask = torch.tensor([[0, 0, 1]])
    sums, counts = _masked_sequence_logps(logits, input_ids, completion_mask)
    expected = torch.log_softmax(logits[:, 1, :].float(), dim=-1)[0, 1]
    assert torch.allclose(sums, expected.unsqueeze(0))
    assert counts.tolist() == [1]


def test_preference_encoder_matches_trl_non_conversational_eos_preprocessing() -> None:
    tokenizer = _CharacterTokenizer()
    prompt = "Question: "
    completion = "answer"
    prompt_ids, completion_ids, full_ids = trl_preference_token_ids(
        tokenizer,
        prompt=prompt,
        completion=completion,
    )
    trl_prompt = trl_tokenize(tokenizer, prompt)["input_ids"]
    trl_full = trl_tokenize(
        tokenizer, prompt + completion + tokenizer.eos_token
    )["input_ids"]
    assert prompt_ids == trl_prompt
    assert full_ids == trl_full
    assert completion_ids == trl_full[len(trl_prompt) :]

    encoded, lengths = _encode_batch(
        tokenizer,
        [{"id": "p", "prompt": prompt, "chosen": completion, "rejected": "other"}],
        max_length=128,
    )
    assert lengths[0] == len(completion_ids)
    assert int(encoded["completion_mask"][0].sum()) == len(completion_ids)

    _, rejected_ids, _ = trl_preference_token_ids(
        tokenizer,
        prompt=prompt,
        completion="other",
    )
    trl_batch = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        max_length=128,
    )
    trl_encoded = trl_batch(
        [
            {
                "prompt_ids": prompt_ids,
                "chosen_ids": completion_ids,
                "rejected_ids": rejected_ids,
            }
        ]
    )
    for key in ("input_ids", "attention_mask", "completion_mask"):
        assert torch.equal(encoded[key], trl_encoded[key])
    vocab_size = max(int(encoded["input_ids"].max()) + 2, 256)
    logits = torch.randn(2, encoded["input_ids"].shape[1], vocab_size)
    ours = _masked_sequence_logps(
        logits,
        encoded["input_ids"],
        encoded["completion_mask"],
    )
    trl_equivalent = _masked_sequence_logps(
        logits,
        trl_encoded["input_ids"],
        trl_encoded["completion_mask"],
    )
    assert torch.equal(ours[0], trl_equivalent[0])
    assert torch.equal(ours[1], trl_equivalent[1])


def test_truncation_audit_counts_the_eos_that_trl_appends() -> None:
    tokenizer = _CharacterTokenizer()
    dataset = Dataset.from_list(
        [{"id": "p", "prompt": "P", "chosen": "C", "rejected": "R"}]
    )
    length_without_eos = len(tokenizer("PC")["input_ids"])
    import pytest

    with pytest.raises(ValueError, match="exceeding"):
        _assert_no_truncation(
            dataset,
            tokenizer,
            max_length=length_without_eos,
        )
