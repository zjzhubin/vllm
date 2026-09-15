# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"
DRAFT_MODEL_TARGET_DIR = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL_DIR = "amd/PARD-Llama-3.2-1B"

NUM_SPECULATIVE_TOKENS = 3


def _make_config(
    target_dir: str,
    draft_dir: str,
    method: str,
    **overrides,
) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=target_dir,
        runner="generate",
        max_model_len=100,
        trust_remote_code=(method == "dflash"),
    )
    kwargs = dict(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_dir,
        method=method,
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
    )
    kwargs.update(overrides)
    return SpeculativeConfig(**kwargs)


def test_dflash_derives_probabilistic_when_unspecified():
    cfg = _make_config(
        DFLASH_TARGET_DIR, DFLASH_DRAFT_DIR, method="dflash"
    )
    assert cfg.draft_sample_method == "probabilistic"


def test_dflash_respects_explicit_greedy():
    cfg = _make_config(
        DFLASH_TARGET_DIR,
        DFLASH_DRAFT_DIR,
        method="dflash",
        draft_sample_method="greedy",
    )
    assert cfg.draft_sample_method == "greedy"


def test_dflash_respects_explicit_probabilistic():
    cfg = _make_config(
        DFLASH_TARGET_DIR,
        DFLASH_DRAFT_DIR,
        method="dflash",
        draft_sample_method="probabilistic",
    )
    assert cfg.draft_sample_method == "probabilistic"


def test_non_dflash_method_derives_greedy():
    cfg = _make_config(
        DRAFT_MODEL_TARGET_DIR, DRAFT_MODEL_DIR, method="draft_model"
    )
    assert cfg.draft_sample_method == "greedy"
