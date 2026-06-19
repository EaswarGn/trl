#!/usr/bin/env python3
"""
Minimal training example for AsyncAtroposGRPOTrainer.

This script demonstrates how to set up and launch asynchronous GRPO training
using Atropos as the rollout source.  Before running this script, you must
have the following services running:

1. **vLLM server** (vanilla vLLM with dev mode and NCCL weight transfer):
   ``CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve \\
       deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
       --max-model-len 4096 \\
       --logprobs-mode processed_logprobs \\
       --weight-transfer-config '{"backend":"nccl"}'``

2. **Atropos API server** (buffers scored trajectories):
   ``run-api``  (default port 8000)

3. **Atropos environment** (generates and scores trajectories):
   ``python atropos/environments/gsm8k_server.py serve \\
       --openai.model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
       --openai.base_url http://localhost:8001/v1 \\
       --env.group_size 8 \\
       --slurm false``

Usage
-----
Single GPU::

    python trl/trl/experimental/async_atropos_grpo/example.py

Multi-GPU (with accelerate)::

    accelerate launch --num_processes 4 \\
        trl/trl/experimental/async_atropos_grpo/example.py
"""

from datasets import load_dataset
from trl.experimental.async_atropos_grpo import (
    AsyncAtroposGRPOTrainer,
    AsyncAtroposGRPOConfig,
)


def main():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    config = AsyncAtroposGRPOConfig(
        output_dir="./async_atropos_grpo_output",
        # --- Atropos API settings ---
        atropos_api_url="http://localhost:8000",
        atropos_group_size=8,
        atropos_trainer_id="trl_async_atropos",
        atropos_batch_timeout=300.0,
        atropos_poll_interval=1.0,
        atropos_max_retries=3,
        atropos_max_inflight_batches=2,
        # --- vLLM server (vanilla vLLM with VLLM_SERVER_DEV_MODE=1) ---
        vllm_server_base_url="http://localhost:9001",
        vllm_server_timeout=240.0,
        weight_sync_steps=1,
        # --- Training hyperparameters ---
        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=2048,
        temperature=1.0,
        epsilon=0.2,
        epsilon_high=0.28,
        max_steps=100,         # total optimizer steps
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=True,
        # --- Async rollout pipeline ---
        max_inflight_tasks=-1,  # auto-compute
        max_staleness=4,
        queue_maxsize=1024,
        heartbeat_stale_after_s=300.0,
        # --- Logging ---
        log_completions=True,
        num_completions_to_print=3,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # The Atropos environment (e.g. gsm8k_server.py) samples prompts from
    # its own dataset.  The trainer still needs a dataset for the dataloader
    # contract, but it is ignored during training (rollouts come from Atropos).
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train", streaming=True)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = AsyncAtroposGRPOTrainer(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        args=config,
        train_dataset=dataset,
        # No reward_funcs needed — scores come from Atropos environment.
        # The trainer will use a pass-through that returns all zeros.
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()