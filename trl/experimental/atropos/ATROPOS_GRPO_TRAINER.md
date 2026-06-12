# AtroposGRPOTrainer

A subclass of [TRL](https://github.com/huggingface/trl)'s `GRPOTrainer` that replaces the in-process rollout generation layer with [Atropos](https://github.com/NousResearch/atropos) as the trajectory source. The trainer handles the full GRPO training loop — weight updates, KL penalties, advantage normalisation, and logging — while Atropos environments handle prompt sampling, generation, and scoring externally.

---

## Table of Contents

1. [Architecture Overview](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#architecture-overview)
2. [How It Differs from GRPOTrainer](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#how-it-differs-from-grpotrainer)
3. [Prerequisites](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#prerequisites)
4. [Installation](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#installation)
5. [The Four-Process Stack](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#the-four-process-stack)
6. [Configuration Reference](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#configuration-reference)
   * [AtroposGRPOConfig](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#atroposgrpoconfig)
   * [Inherited GRPOConfig Fields](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#inherited-grpoconfig-fields)
7. [API Reference](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#api-reference)
   * [AtroposGRPOTrainer](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#atroposgrpotrainer-class)
   * [AtroposAPIClient](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#atrosapiclient-class)
   * [make_atropos_trainer](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#make_atropos_trainer)
8. [The Atropos Batch Contract](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#the-atropos-batch-contract)
9. [Training Loop Internals](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#training-loop-internals)
10. [Weight Synchronisation](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#weight-synchronisation)
11. [Advantage Computation](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#advantage-computation)
12. [KL Penalty and Reference Model](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#kl-penalty-and-reference-model)
13. [Evaluation](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#evaluation)
14. [PEFT / LoRA Support](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#peft--lora-support)
15. [Multi-GPU and Distributed Training](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#multi-gpu-and-distributed-training)
16. [Logging and Metrics](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#logging-and-metrics)
17. [Checkpointing and Resuming](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#checkpointing-and-resuming)
18. [Troubleshooting](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#troubleshooting)
19. [Design Decisions and Caveats](https://claude.ai/chat/9820972e-af62-4089-9abc-e865bb5177b5#design-decisions-and-caveats)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│              TRL vLLM server  (trl vllm-serve)               │
│              Host: 0.0.0.0   Port: 8001                      │
│  • Holds the live copy of the model weights                  │
│  • Serves OpenAI-compatible /v1 completions API              │
│  • Accepts weight pushes via NCCL from the trainer           │
└────────────────────┬─────────────────────────────────────────┘
                     │  OpenAI /v1/completions  (generation)
         ┌───────────▼──────────────────────────────┐
         │   Atropos environment microservice(s)     │
         │   e.g. gsm8k_server.py, code_server.py   │
         │   • Samples prompts from a dataset        │
         │   • Calls TRL vLLM to generate responses  │
         │   • Scores each response                  │
         │   • POST /scored_data  →  Atropos API     │
         └───────────┬──────────────────────────────┘
                     │  POST /scored_data
         ┌───────────▼──────────────────────────────┐
         │   Atropos API server  (run-api)           │
         │   Port: 8000                              │
         │   • Buffers scored trajectory batches     │
         │   • Exposes GET /batch to the trainer     │
         │   • Accepts POST /register from trainer   │
         └───────────┬──────────────────────────────┘
                     │  GET /batch  (scored trajectories)
         ┌───────────▼──────────────────────────────┐
         │   AtroposGRPOTrainer                      │
         │   1. _prepare_inputs()                    │
         │      a. sync_weights() → TRL vLLM         │
         │      b. wait_for_batch() → Atropos API    │
         │      c. _convert_atropos_batch() → tensors│
         │   2. compute_loss()        ◄── GRPOTrainer│
         │   3. optimizer.step()      ◄── GRPOTrainer│
         │   (repeat from 1)                         │
         └──────────────────────────────────────────┘
```

**Key principle:** There is exactly one vLLM instance in the whole system — the TRL vLLM server. The Atropos environments call it for generation via its OpenAI-compatible HTTP API. The trainer pushes updated weights to it via NCCL after each training window. This eliminates the need for any Atropos-owned vLLM server.

---

## How It Differs from GRPOTrainer

| Aspect                        | `GRPOTrainer`                                               | `AtroposGRPOTrainer`                                                      |
| ----------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Rollout generation**  | In-process, called inside `_generate_and_score_completions` | External — Atropos environments call the TRL vLLM server                   |
| **Reward computation**  | `reward_funcs`applied to completions on the trainer process | Inside the Atropos environment; arrives as `scores`in the batch           |
| **`_prepare_inputs`** | Calls `_generate_and_score_completions`                     | Calls `_atropos_client.wait_for_batch()`then `_convert_atropos_batch()` |
| **Weight sync**         | Called inside `_generate_and_score_completions`             | Called inside `_prepare_inputs`at the start of each generation window     |
| **`training_step`**   | Unmodified                                                    | Unmodified (no override)                                                    |
| **`compute_loss`**    | Unmodified                                                    | Unmodified (inherited directly)                                             |
| **`train_dataset`**   | Required                                                      | Optional — a placeholder `IterableDataset`is created automatically       |
| **Config class**        | `GRPOConfig`                                                | `AtroposGRPOConfig`(subclass of `GRPOConfig`)                           |
| **`reward_funcs`**    | Required                                                      | Optional — defaults to a no-op pass-through                                |

Everything that is not listed above — gradient accumulation, GRPO clipping, KL penalty, PEFT adapter handling, DeepSpeed/FSDP, logging, checkpointing, evaluation — is completely unchanged and delegated to `GRPOTrainer`.

---

## Prerequisites

* Python ≥ 3.10
* PyTorch ≥ 2.1
* [TRL](https://github.com/huggingface/trl) (with vLLM support): `pip install trl[vllm]`
* [vLLM](https://github.com/vllm-project/vllm) ≥ 0.4: `pip install vllm`
* [Atropos](https://github.com/NousResearch/atropos): `pip install atropos` (or install from source)
* `requests`, `accelerate`, `transformers` (pulled in by TRL)

A GPU is required for the TRL vLLM server and the trainer. The Atropos API server and environment microservices can run on CPU.

---

## Installation

```bash
# 1. Install TRL with vLLM extras
pip install "trl[vllm]"

# 2. Install vLLM (pin the version your cluster supports)
pip install vllm

# 3. Install Atropos
pip install atropos
# or from source:
# git clone https://github.com/NousResearch/atropos && cd atropos && pip install -e .

# 4. Copy atropos_grpo_trainer.py into your project
cp atropos_grpo_trainer.py your_project/
```

---

## The Four-Process Stack

A full training run requires four separate processes, typically in four terminal windows or tmux panes. Start them in the order shown.

### Process 1 — TRL vLLM Server

Serves the model for generation and accepts weight updates from the trainer.

```bash
trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45
```

> **GPU memory note:** The vLLM server and the trainer share the same GPU(s). Tune `--gpu-memory-utilization` so that both fit. A common split is `0.45` for the server and the remaining `~0.45` for training (PyTorch needs headroom too). For larger models, use tensor parallelism across multiple GPUs.

Wait until you see `Application startup complete` before starting anything else.

### Process 2 — Atropos API Server

Buffers scored trajectories from environments and serves them to the trainer.

```bash
run-api
# Listens on http://0.0.0.0:8000 by default
```

### Process 3 — Atropos Environment

One or more environment processes that generate and score trajectories. Each environment must be pointed at the TRL vLLM server.

```bash
python environments/gsm8k_server.py serve \
  --openai.model_name Qwen/Qwen2.5-1.5B-Instruct \
  --openai.base_url http://localhost:8001/v1 \
  --env.group_size 8 \
  --slurm false
```

The `--openai.base_url` must point to the TRL vLLM server, not any separate vLLM instance.

### Process 4 — Trainer

```bash
python train_atropos.py
```

---

## Configuration Reference

### AtroposGRPOConfig

`AtroposGRPOConfig` is a `@dataclass` that subclasses `GRPOConfig`. It adds six Atropos-specific fields and inherits all of TRL's existing training hyperparameters.

```python
from atropos_grpo_trainer import AtroposGRPOConfig

config = AtroposGRPOConfig(
    output_dir="./my_run",
    # --- Atropos-specific ---
    atropos_api_url="http://localhost:8000",
    atropos_group_size=8,
    atropos_trainer_id="trl_grpo",
    atropos_batch_timeout=300.0,
    atropos_poll_interval=1.0,
    atropos_max_retries=3,
    # --- TRL vLLM server ---
    use_vllm=True,
    vllm_mode="server",
    vllm_server_host="0.0.0.0",
    vllm_server_port=8001,
    # --- Training ---
    per_device_train_batch_size=4,
    num_generations=8,
    max_steps=1000,
    learning_rate=1e-6,
    beta=0.01,
)
```

#### Atropos-specific fields

| Field                     | Type      | Default                     | Description                                                                                                       |
| ------------------------- | --------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `atropos_api_url`       | `str`   | `"http://localhost:8000"` | Base URL of the Atropos `run-api`server.                                                                        |
| `atropos_group_size`    | `int`   | `8`                       | Number of completions per prompt group.**Must match the `group_size`in your Atropos environment config.** |
| `atropos_trainer_id`    | `str`   | `"trl_grpo"`              | Identifier sent to the Atropos API on `/register`. Useful when running multiple trainers against the same API.  |
| `atropos_batch_timeout` | `float` | `300.0`                   | Seconds to wait for a batch before raising `TimeoutError`. Increase this if your environment is slow to score.  |
| `atropos_poll_interval` | `float` | `1.0`                     | Seconds between `/batch`polls when no data is ready yet.                                                        |
| `atropos_max_retries`   | `int`   | `3`                       | HTTP retry count on transient network failures.                                                                   |

### Inherited GRPOConfig Fields

These are the most commonly changed TRL fields. See the [TRL GRPOConfig docs](https://huggingface.co/docs/trl/grpo_trainer) for the full list.

| Field                           | Default        | Description                                                                                |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------ |
| `use_vllm`                    | `False`      | Must be `True`. Forced to `True`with a warning if you pass `False`.                  |
| `vllm_mode`                   | `"colocate"` | Must be `"server"`. Set this explicitly.                                                 |
| `vllm_server_host`            | `"0.0.0.0"`  | Host where `trl vllm-serve`is running.                                                   |
| `vllm_server_port`            | `8000`       | Port where `trl vllm-serve`is listening.                                                 |
| `vllm_server_base_url`        | `None`       | Full base URL override (e.g.`"http://vllm-host:8001"`). Takes precedence over host+port. |
| `per_device_train_batch_size` | `8`          | Micro-batch size per GPU for the backward pass.                                            |
| `num_generations`             | `8`          | Completions per prompt (=`atropos_group_size`). Set these equal.                         |
| `steps_per_generation`        | `1`          | How many optimizer steps reuse one fetched batch before fetching the next.                 |
| `num_iterations`              | `1`          | GRPO μ — number of policy update iterations per batch.                                   |
| `max_steps`                   | `-1`         | Total optimizer steps. Set this or `num_train_epochs`.                                   |
| `learning_rate`               | `1e-6`       | Optimizer learning rate.                                                                   |
| `beta`                        | `0.04`       | KL penalty coefficient. Set to `0.0`to disable.                                          |
| `epsilon`                     | `0.2`        | PPO clipping ratio.                                                                        |
| `gradient_accumulation_steps` | `1`          | Micro-steps before an optimizer step.                                                      |
| `gradient_checkpointing`      | `False`      | Reduces GPU memory at the cost of recomputation.                                           |
| `bf16`                        | `False`      | Train in bfloat16. Recommended on Ampere+ GPUs.                                            |
| `output_dir`                  | required       | Directory for checkpoints and logs.                                                        |
| `logging_steps`               | `1`          | Log metrics every N steps.                                                                 |
| `save_steps`                  | `500`        | Save a checkpoint every N steps.                                                           |
| `seed`                        | `42`         | Global random seed.                                                                        |

---

## API Reference

### `AtroposGRPOTrainer` class

```python
class AtroposGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: str | PreTrainedModel,
        args: AtroposGRPOConfig | None = None,
        reward_funcs=None,         # optional, defaults to no-op pass-through
        train_dataset=None,        # optional, defaults to placeholder IterableDataset
        eval_dataset=None,         # optional, passed to GRPOTrainer as-is
        **kwargs,                  # forwarded to GRPOTrainer.__init__
    )
```

**Parameters:**

* **`model`** — A HuggingFace model ID string (e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`) or an already-instantiated `PreTrainedModel`. If a string, the model is loaded from the Hub.
* **`args`** — An `AtroposGRPOConfig` instance. If `None`, a default config is created using the model name as `output_dir`.
* **`reward_funcs`** — One or more TRL-compatible reward functions. When running with Atropos, rewards are produced by the environment, so this defaults to `_atropos_passthrough_reward` (returns all zeros). Pass custom reward functions only if you want to add trainer-side reward signals on top of Atropos scores.
* **`train_dataset`** — A HuggingFace `Dataset` or `IterableDataset`. This is ignored during training (the Atropos API is the real data source) but must satisfy the TRL trainer contract. If `None`, a lightweight `_AtroposPlaceholderDataset` is created automatically.
* **`eval_dataset`** — Optional evaluation dataset. When provided, evaluation runs the standard `GRPOTrainer` evaluation loop using the model's own generation (not Atropos).
* **`**kwargs`** — Any additional keyword arguments are forwarded to `GRPOTrainer.__init__` (e.g. `processing_class`, `peft_config`, `callbacks`).

**Key methods (all inherited unless noted):**

| Method                       | Source     | Description                                                                                                    |
| ---------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| `train()`                  | Overridden | Checks Atropos API reachability, registers, then calls `super().train()`.                                    |
| `_prepare_inputs()`        | Overridden | Syncs weights, fetches Atropos batch, converts to tensor dict, returns slice.                                  |
| `_sync_weights_to_vllm()`  | New        | Calls `vllm_generation.sync_weights()`with the `_last_loaded_step`guard.                                   |
| `_convert_atropos_batch()` | New        | Deserialises Atropos trajectory dicts into the tensor dict `compute_loss`expects.                            |
| `_compute_ref_logps()`     | New        | Computes reference model logprobs for the KL term (used when `beta > 0`).                                    |
| `get_train_dataloader()`   | Overridden | Returns a placeholder DataLoader when using `_AtroposPlaceholderDataset`; falls through to parent otherwise. |
| `compute_loss()`           | Inherited  | GRPO/PPO loss with importance sampling and KL penalty. Unmodified.                                             |
| `training_step()`          | Inherited  | Accumulates gradients, increments `_step`, records timing. Unmodified.                                       |

### `AtroposAPIClient` class

A thin HTTP client used internally by the trainer. Can also be used standalone for debugging.

```python
from atropos_grpo_trainer import AtroposAPIClient

client = AtroposAPIClient(
    base_url="http://localhost:8000",
    trainer_id="debug",
    timeout=60.0,
    poll_interval=0.5,
    max_retries=3,
)

# Check if the API is up
client.health()  # -> bool

# Register (batch_size = total trajectories per fetch)
client.register(batch_size=32, group_size=8)  # -> dict

# Poll once (returns None if no batch is ready)
batch = client.fetch_batch()  # -> list[dict] | None

# Block until a batch is available
batch = client.wait_for_batch()  # -> list[dict]
```

### `make_atropos_trainer`

A convenience factory function for the most common configuration.

```python
from atropos_grpo_trainer import make_atropos_trainer

trainer = make_atropos_trainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    atropos_api_url="http://localhost:8000",
    group_size=8,
    per_device_train_batch_size=4,
    max_steps=1000,
    output_dir="./my_run",
    vllm_server_host="0.0.0.0",
    vllm_server_port=8001,
    # Pass any AtroposGRPOConfig fields as extra kwargs:
    extra_config_kwargs={
        "beta": 0.01,
        "learning_rate": 5e-7,
        "bf16": True,
    },
)
trainer.train()
```

---

## The Atropos Batch Contract

The Atropos API's `/batch` endpoint returns a list of trajectory dicts. Each dict must contain:

| Field                    | Type                        | Description                                                                                                                                                                                                                     |
| ------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tokens`               | `list[int]`               | Full token sequence: prompt tokens followed by completion tokens.                                                                                                                                                               |
| `masks`                | `list[int]`               | `-100`for each prompt position; the actual token ID at each completion position. Used to recover the prompt/completion boundary. Matches the `masks`field from the server's `ScoredData`model.                              |
| `inference_logprobs`   | `list[float]`             | `1.0`(sentinel) for each prompt position; the actual log-probability at each completion token position. Must be the log-prob of the**sampled**token under the policy that generated it (the TRL vLLM server). |
| `scores`               | `float`or `list[float]` | The environment's reward for this trajectory. If a list, values are summed to a single float.                                                                                                                                   |

The following fields are optional but used for logging:

| Field               | Type    | Description                                                        |
| ------------------- | ------- | ------------------------------------------------------------------ |
| `prompt_text`     | `str` | Human-readable prompt string. Logged to the completions table.     |
| `completion_text` | `str` | Human-readable completion string. Logged to the completions table. |
| `finish_reason`   | `str` | vLLM finish reason (`"stop"`,`"length"`, etc.).                |

**Batch size constraint:** The number of items in the batch must be divisible by `atropos_group_size`. The trainer raises `ValueError` if this is violated.

**Example trajectory dict:**

```python
{
    "tokens":              [10, 20, 30, 40, 50, 60, 1],   # 3 prompt + 4 completion tokens
    "masks":               [-100, -100, -100, 40, 50, 60, 1],
    "inference_logprobs":  [1.0, 1.0, 1.0, -0.3, -1.2, -0.8, -0.05],
    "scores":        1.0,
    "prompt_text":   "What is 2 + 2?",
    "completion_text": "4",
    "finish_reason": "stop",
}
```

---

## Training Loop Internals

Understanding the exact call sequence helps when debugging timing issues.

```
trainer.train()
  └─ _ensure_registered()          # POST /register (once)
  └─ super().train()               # standard HuggingFace Trainer loop
       └─ for each dataloader batch:
            └─ training_step(model, inputs, ...)   # GRPOTrainer (unmodified)
                 └─ compute_loss(model, inputs)    # GRPOTrainer (unmodified)
                      └─ _prepare_inputs(batch)    # ← AtroposGRPOTrainer OVERRIDE
                           │
                           ├─ [if _step % generate_every == 0]
                           │     _sync_weights_to_vllm()
                           │       └─ vllm_generation.sync_weights()
                           │            └─ NCCL broadcast → TRL vLLM server
                           │
                           │     _atropos_client.wait_for_batch()
                           │       └─ poll GET /batch until 200 OK
                           │
                           │     _convert_atropos_batch(raw_batch)
                           │       ├─ parse tokens / masked_tokens / logprobs
                           │       ├─ compute GRPO advantages
                           │       ├─ pad & tensorise
                           │       ├─ [if beta > 0] _compute_ref_logps()
                           │       └─ return tensor dict
                           │
                           │     split_tensor_dict(prepared, steps_per_generation)
                           │     self._buffered_inputs = batches
                           │
                           └─ return _buffered_inputs[_step % steps_per_generation]
```

The `generate_every` window is `steps_per_generation × num_iterations`. Within one window, the same fetched batch is reused across multiple optimizer steps (standard GRPO multi-step reuse).

---

## Weight Synchronisation

Weight sync uses TRL's native `VLLMGeneration.sync_weights()` which does a direct NCCL broadcast from the trainer process to all vLLM worker processes. No intermediate HTTP transfer occurs.

**Sync cadence:** Once at the start of each `generate_every` window, immediately before calling `wait_for_batch()`. This ensures the Atropos environment is always generating from the latest policy by the time the trainer requests the next batch.

**The `_last_loaded_step` guard:** Inherited from `GRPOTrainer`, this integer prevents double-syncing within the same global step (e.g. during gradient accumulation). The guard is checked before every `sync_weights()` call.

**What happens if sync fails:** The trainer logs a warning and continues. The vLLM server retains its previous weights. Training is not aborted because a single failed sync is recoverable (the next window will attempt again).

---

## Advantage Computation

Advantages are computed from environment scores using standard Group Relative Policy Optimisation normalisation:

```
advantages_i = (score_i - mean(group)) / (std(group) + 1e-4)
```

where each group is a set of `atropos_group_size` trajectories that were generated from the same prompt. This is identical to the formula used by `GRPOTrainer` internally, ensuring the downstream `compute_loss` behaves exactly the same.

The `1e-4` epsilon prevents division by zero when all completions in a group receive the same score.

---

## KL Penalty and Reference Model

When `beta > 0`, a KL divergence penalty is added to the GRPO loss to prevent the policy from drifting too far from the reference model. This works identically to the base `GRPOTrainer`.

**Full fine-tuning:** If you are not using PEFT and `beta > 0`, the reference model is a frozen copy of the initial model weights. It is loaded once at startup and held in GPU memory for the duration of training.

**PEFT/LoRA:** When a `peft_config` is passed, the reference model is the base model with the LoRA adapter disabled (zero weights). No extra copy of the model is needed in memory.

**Disabling KL:** Set `beta=0.0` in `AtroposGRPOConfig`. No reference model is loaded and `_compute_ref_logps` is never called.

---

## Evaluation

When an `eval_dataset` is provided and `eval_steps` or `evaluation_strategy` is configured, the trainer runs standard `GRPOTrainer` evaluation. During evaluation `_prepare_inputs` detects `model.training == False` and delegates to `super()._prepare_inputs()`, which calls `_generate_and_score_completions` as normal. This means:

* Evaluation **does not** use the Atropos API.
* Evaluation uses the model's own generation (via vLLM or transformers).
* Reward functions passed to `AtroposGRPOTrainer` are applied during evaluation.

If you want evaluation to also use Atropos, override `_prepare_inputs` and remove the `mode == "eval"` short-circuit.

---

## PEFT / LoRA Support

Pass a `peft_config` to `AtroposGRPOTrainer.__init__` exactly as you would to `GRPOTrainer`:

```python
from peft import LoraConfig
from atropos_grpo_trainer import AtroposGRPOTrainer, AtroposGRPOConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = AtroposGRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    args=config,
    peft_config=peft_config,
)
```

The `_compute_ref_logps` method handles the PEFT case by temporarily disabling the active adapter to get base-model log-probabilities for the KL penalty.

---

## Multi-GPU and Distributed Training

The trainer itself uses `accelerate` for distributed training in the same way as `GRPOTrainer`. Launch with `accelerate launch` or `torchrun`:

```bash
accelerate launch --num_processes 4 train_atropos.py
```

**vLLM server tensor parallelism:** The TRL vLLM server can be launched with `--tensor-parallel-size N` to shard the model across N GPUs. These GPUs must be different from the ones used by the trainer.

```bash
# 2 GPUs for the vLLM server (CUDA_VISIBLE_DEVICES=0,1)
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --port 8001

# 2 GPUs for training (CUDA_VISIBLE_DEVICES=2,3)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 train_atropos.py
```

The `batch_size` sent to `/register` is automatically computed as `per_device_train_batch_size × num_processes`.

---

## Logging and Metrics

The trainer reports all the same metrics as `GRPOTrainer` plus the following:

| Metric key                  | Description                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `train/reward`            | Mean raw environment score across the batch.                        |
| `train/reward_std`        | Standard deviation of raw scores.                                   |
| `train/completion_length` | Mean completion length in tokens.                                   |
| `rewards/atropos_env`     | Per-sample rewards (logged to the completions table).               |
| `advantages`              | Per-sample normalised advantages (logged to the completions table). |

All standard GRPO metrics (`train/loss`, `train/kl`, `train/entropy`, etc.) are inherited from `GRPOTrainer` unchanged.

Enable W&B or TensorBoard as you normally would with TRL:

```python
config = AtroposGRPOConfig(
    output_dir="./my_run",
    report_to="wandb",   # or "tensorboard"
    run_name="my-atropos-run",
    ...
)
```

---

## Checkpointing and Resuming

Checkpointing works identically to `GRPOTrainer` via HuggingFace `Trainer`. Checkpoints are saved every `save_steps` steps to `output_dir/checkpoint-N/`.

```python
config = AtroposGRPOConfig(
    output_dir="./my_run",
    save_steps=200,
    save_total_limit=3,   # keep only the last 3 checkpoints
    ...
)
```

To resume:

```bash
python train_atropos.py --resume_from_checkpoint ./my_run/checkpoint-400
```

Or in code:

```python
trainer.train(resume_from_checkpoint="./my_run/checkpoint-400")
```

Note: `_buffered_inputs` is `None` after resuming, which is correctly handled — `_prepare_inputs` will fetch a fresh batch from Atropos on the first step.

---

## Troubleshooting

**`ConnectionError: Cannot reach Atropos API at http://localhost:8000`**
The Atropos `run-api` server is not running or is not reachable. Start it with `run-api` and verify with `curl http://localhost:8000/`.

**`TimeoutError: No batch available from Atropos API within 300s`**
No Atropos environment is running, or the environment has crashed. Check that `run-api` shows environment connections and that the environment process is alive.

**`ValueError: Atropos batch size N is not divisible by group_size M`**
The `group_size` configured in your Atropos environment does not match `atropos_group_size` in `AtroposGRPOConfig`. They must be equal.

**`ValueError: Atropos trajectory has mismatched lengths`**
The environment sent a malformed trajectory where `len(tokens)`, `len(masked_tokens)`, and `len(logprobs)` are not all equal. This is an environment bug.

**Weight sync failed / vLLM returns old model outputs**
Check that `use_vllm=True` and `vllm_mode="server"` are set in your config. Verify the trainer can reach the vLLM server (`curl http://localhost:8001/health/`). Check for NCCL errors in the logs.

**CUDA out of memory**
Reduce `--gpu-memory-utilization` on the vLLM server, reduce `per_device_train_batch_size`, enable `gradient_checkpointing=True`, or switch to PEFT/LoRA to reduce the training model's footprint.

**`TypeError: AtroposGRPOTrainer requires an AtroposGRPOConfig instance`**
You passed a plain `GRPOConfig`. Promote it:

```python
import dataclasses
from atropos_grpo_trainer import AtroposGRPOConfig
args = AtroposGRPOConfig(**dataclasses.asdict(your_grpo_config))
```

---

## Design Decisions and Caveats

**`train_dataset` is ignored.** The placeholder dataset is a hack to satisfy `Trainer`'s requirement for a non-empty dataset. The actual data comes from the Atropos API. If you pass a real dataset, the parent's sampler logic will run but the resulting batch dict will be discarded in `_prepare_inputs`.

**`reward_funcs` is a no-op by default.** Atropos environments compute rewards internally. The `_atropos_passthrough_reward` function returns all zeros so the base class reward pipeline does not interfere. The `advantages` tensor inserted by `_convert_atropos_batch` overrides any reward-derived advantages anyway.

**`eval_dataset` uses in-process generation.** Evaluation bypasses Atropos entirely and uses the model's own generation. This is intentional — evaluation should reflect the model's independent ability, not the environment's generation pipeline.

**`num_generations` should equal `atropos_group_size`.** These two config fields play the same conceptual role (completions per prompt group) in different systems. They are set independently because `num_generations` drives the TRL loss computation while `atropos_group_size` drives the advantage normalisation. Set them to the same value.
