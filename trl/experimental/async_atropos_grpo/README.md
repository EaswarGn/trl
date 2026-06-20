# AsyncAtroposGRPOTrainer

An asynchronous GRPO trainer that sources pre-scored trajectory rollouts from the [Atropos](https://github.com/NousResearch/atropos) RL environment platform instead of generating them in-process. Built on top of TRL's [`AsyncGRPOTrainer`](../async_grpo/).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Why Async + Atropos?](#why-async--atropos)
3. [The Four-Process Stack](#the-four-process-stack)
4. [Data Flow in Detail](#data-flow-in-detail)
5. [Inheritance Hierarchy](#inheritance-hierarchy)
6. [The `RolloutWorkerProtocol` Pattern](#the-rolloutworkerprotocol-pattern)
7. [How It Differs from `AsyncGRPOTrainer`](#how-it-differs-from-asyncgrpotrainer)
8. [Configuration Reference](#configuration-reference)
9. [API Reference](#api-reference)
10. [The Atropos Batch Contract](#the-atropos-batch-contract)
11. [Loss Computation](#loss-computation)
12. [Weight Synchronization](#weight-synchronization)
13. [Advantage Computation](#advantage-computation)
14. [Distributed Training](#distributed-training)
15. [Metrics and Logging](#metrics-and-logging)
16. [Troubleshooting](#troubleshooting)
17. [Design Decisions and Caveats](#design-decisions-and-caveats)
18. [File Reference](#file-reference)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Atropos ecosystem                                │
│                                                                      │
│  ┌──────────────────────────┐     ┌───────────────────────────────┐  │
│  │ Atropos environment(s)   │────▶│ Atropos API server (run-api)  │  │
│  │ - Sample prompts         │     │ Port 8000                     │  │
│  │ - Generate completions   │     │ - Buffers scored trajectories │  │
│  │   (via vLLM server)      │     │ - Exposes GET /batch          │  │
│  │ - Score each response    │     │ - Accepts POST /register      │  │
│  │ - POST /scored_data      │     └──────────┬────────────────────┘  │
│  └──────────────────────────┘                │                       │
│                                              │ GET /batch             │
└──────────────────────────────────────────────┼───────────────────────┘
                                               │
                    ┌──────────────────────────▼───────────────────────┐
                    │  AtroposRolloutWorker (child process, CUDA-free) │
                    │                                                  │
                    │  - Polls /batch for scored trajectory groups     │
                    │  - Flattens groups into individual trajectories  │
                    │  - Computes group-relative GRPO advantages       │
                    │  - Pushes RolloutSample -> shared mp.Queue       │
                    └──────────────────────────┬──────────────────────┘
                                               │ mp.Queue[RolloutSample]
                    ┌──────────────────────────▼───────────────────────┐
                    │  AsyncAtroposGRPOTrainer (GPU process)           │
                    │                                                  │
                    │  [Training loop inherited from AsyncGRPOTrainer] │
                    │                                                  │
                    │  1. RolloutQueueDataset drains mp.Queue          │
                    │  2. DataCollatorForRollout pads & collates       │
                    │  3. compute_loss() -- GRPO loss (all variants)    │
                    │  4. optimizer.step()                             │
                    │  5. _sync_weight() -- NCCL -> vLLM dev server    │
                    │     (so Atropos envs use latest policy)          │
                    └─────────────────────────────────────────────────┘

                ┌──────────────────────────────────────────┐
                │  vLLM server (separate GPU, dev mode)    │
                │  VLLM_SERVER_DEV_MODE=1                  │
                │  --weight-transfer-config '{"backend":   │
                │    "nccl"}'                              │
                │                                          │
                │  Atropos envs and trainer both connect:  │
                │    - envs: /v1/completions (generation)  │
                │    - trainer: NCCL (weight push)         │
                └──────────────────────────────────────────┘
```

**Key principle:** There is exactly one vLLM instance -- a **vanilla vLLM server started with dev mode** (`VLLM_SERVER_DEV_MODE=1`) and NCCL weight transfer enabled. The Atropos environment microservices call it for generation via its OpenAI-compatible HTTP API. The trainer pushes updated weights to it via NCCL after each weight sync window. This differs from `AsyncGRPOTrainer`'s parent setup where `trl vllm-serve` is used; here we use the raw vLLM server with the `VLLM_SERVER_DEV_MODE` environment variable.

---

## Why Async + Atropos?

The **`AsyncGRPOTrainer`** (from `trl.experimental.async_grpo`) decouples rollout generation from training using a multi-process architecture:

- **GPU process (trainer):** Handles the forward/backward pass, gradient accumulation, optimizer step, logging, and checkpointing.
- **Child process (rollout worker):** Generates completions and scores them, completely free of GPU or CUDA concerns.

This means the trainer never blocks on generation -- it is always consuming previously generated batches from a queue.

The **`AsyncAtroposGRPOTrainer`** takes this one step further: instead of generating completions via its own vLLM calls in a child process, it **polls the Atropos API** for pre-scored trajectory batches produced by external Atropos environment microservices. This enables:

1. **Complex scoring pipelines** -- Atropos environments can run reward models, code execution verifiers, multi-turn interactions, tool use, etc., all outside the trainer process.
2. **Horizontal scaling** -- Multiple environment processes can run in parallel, potentially on different machines.
3. **Weight sync via NCCL** -- The trainer pushes updated weights to the vLLM server's NCCL weight transfer engine, which the Atropos environments call for generation, keeping the policy fresh without the trainer managing any vLLM instance directly.

---

## The Four-Process Stack

A full training run requires four separate processes, typically in four terminal windows or tmux panes. Start them in the order shown.

### Process 1 -- vLLM Server (Dev Mode)

Serves the model for generation and accepts weight updates from the trainer via NCCL. This is a **vanilla vLLM server** started with `VLLM_SERVER_DEV_MODE=1` and the NCCL weight transfer config, **not** `trl vllm-serve`.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 4096 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    --port 9001
```

Key flags:

| Flag / Env Var                                    | Purpose                                                                                                                                                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `VLLM_SERVER_DEV_MODE=1`                        | Enables the weight transfer HTTP endpoints (`/health`, `/get_world_size`, `/init_weight_transfer_engine`, `/update_weights`, `/pause`, `/resume`) that the `WeightTransferClient` calls. Without this, NCCL weight sync will fail. |
| `--max-model-len <N>`                           | Context window. Set to at least `max_prompt_length + max_completion_length`. Lower values reduce GPU memory usage.                                                                                                                             |
| `--logprobs-mode processed_logprobs`            | Ensures logprobs are returned in a format compatible with the trainer's loss computation.                                                                                                                                                        |
| `--weight-transfer-config '{"backend":"nccl"}'` | Enables the NCCL weight transfer engine that the trainer's `WeightTransferClient` connects to.                                                                                                                                                 |

**GPU isolation note (important):** In async server mode, the vLLM server and the trainer must run on **separate GPUs**. Use `CUDA_VISIBLE_DEVICES` to partition your GPUs. For example, with 2 GPUs, run the vLLM server on GPU 0 and the trainer on GPU 1:

```bash
# Terminal 1: vLLM server on GPU 0
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve ...

# Terminal 2: trainer on GPU 1
CUDA_VISIBLE_DEVICES=1 accelerate launch ...
```

With 4 GPUs, a typical split is 2 GPUs for vLLM (tensor parallel size 2) and 2 GPUs for training:

```bash
# Terminal 1: vLLM server on GPUs 0-1
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 vllm serve ... --tensor-parallel-size 2

# Terminal 2: trainer on GPUs 2-3
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 ...
```

Wait until you see `Application startup complete` before starting anything else.

### Process 2 -- Atropos API Server

Buffers scored trajectories from environments and serves them to the trainer.

```bash
run-api
# Listens on http://0.0.0.0:8000 by default
```

### Process 3 -- Atropos Environment

One or more environment processes that generate and score trajectories. Each environment must be pointed at the vLLM server.

```bash
python environments/gsm8k_server.py serve \
  --openai.model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --openai.base_url http://localhost:9001/v1 \
  --openai.server_type vllm_logprob \
  --env.group_size 8 \
  --env.tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --env.max_token_length 3072 \
  --slurm false 
```

The `--openai.base_url` must point to the **vLLM server** (not any separate instance).

### Process 4 -- Trainer

```bash
python trl/trl/experimental/async_atropos_grpo/example.py
```

Or with `accelerate` for multi-GPU:

```bash
accelerate launch --num_processes 4 \
  trl/trl/experimental/async_atropos_grpo/example.py
```

---

## Data Flow in Detail

### Training Loop (single iteration)

```
trainer.train()
  |
  +- super().train()                 # AsyncGRPOTrainer training loop
  |    |
  |    +- _inner_training_loop():
  |    |    +- [AsyncAtropos override]:
  |    |    |    - register_with_atropos()   # main process only
  |    |    |    - barrier (all ranks)
  |    |    |
  |    |    +- on_train_begin callbacks:
  |    |    |    +- _InitialWeightSyncCallback  # NCCL group setup + cold sync
  |    |    |    +- _StartRolloutWorkerCallback # spawn child, start polling
  |    |    |
  |    |    +- get_train_dataloader():
  |    |    |    +- RolloutQueueDataset(rollout_queue, ...)
  |    |    |       (IterableDataset that drains mp.Queue from the worker)
  |    |    |
  |    |    +- for each batch from dataloader:
  |    |    |    |
  |    |    |    +- DataCollatorForRollout.collate(batch)
  |    |    |    |    - Pad input_ids, attention_mask, completion_mask,
  |    |    |    |      old_log_probs to global batch max length
  |    |    |    |    - Stack advantages into tensor
  |    |    |    |    - Compute global_n_tokens for DAPO normalization
  |    |    |    |    - Convert metrics dicts to tensor dict
  |    |    |    |
  |    |    |    +- compute_loss(model, inputs)
  |    |    |    |    - Forward pass through the model
  |    |    |    |    - Compute log_probs and entropy for generated tokens
  |    |    |    |    - GRPO clipped loss:
  |    |    |    |        ratio = exp(log_probs - old_log_probs)
  |    |    |    |        loss = -min(ratio * A, clip(ratio, 1-e, 1+e) * A)
  |    |    |    |    - Normalize by global_n_tokens (DAPO)
  |    |    |    |    - Metric tracking (KL, entropy, clip ratio, etc.)
  |    |    |    |
  |    |    |    +- loss.backward()
  |    |    |    |
  |    |    |    +- optimizer.step()
  |    |    |            |
  |    |    |            +- [every weight_sync_steps]:
  |    |    |                 _sync_weight()
  |    |    |                   +- pause vLLM server
  |    |    |                   +- NCCL broadcast weights to vLLM
  |    |    |                   +- resume vLLM server
  |    |    |                   +- model_version += 1
  |    |    |
  |    |    +- on_train_end:
  |    |         +- _inner_training_loop finally:
  |    |              - rollout_worker.stop()
  |    |              - weight_transfer.destroy()
  |    |
  |    +- (train() finally: done)
```

### Child Process (AtroposRolloutWorker) Lifecycle

```
AtroposRolloutWorker.start()
  |
  +- Spawn child process (mp.spawn)
  |
  +- Child process:
       |
       +- _scrub_child_env()
       |    - CUDA_VISIBLE_DEVICES=""
       |    - Strip RANK, WORLD_SIZE, etc. (avoid accelerate confusion)
       |
       +- _poll_loop():
            |
            while not stop:
              |
              +- heartbeat = time.time()
              |
              +- _wait_for_batch()     # poll GET /batch with timeout
              |    |
              |    +- [returns list of group dicts]
              |
              +- _convert_batch(raw_batch):
              |    |
              |    +- For each group dict:
              |    |    +- Extract tokens, masks, logprobs, scores
              |    |    +- Validate shapes and consistency
              |    |    +- Flatten into per-trajectory records
              |    |
              |    +- Find completion boundary (masks != -100)
              |    +- Split into prompt / completion parts
              |    +- Compute advantages via GRPO normalization:
              |    |    advantages_i = (score_i - mean(group))
              |    |                   / (std(group) + 1e-4)
              |    |
              |    +- Build list of RolloutSample objects
              |
              +- Push samples to mp.Queue
              |
              +- heartbeat = time.time()
```

**Note:** Registration with the Atropos API (POST /register) is handled by the parent trainer (`AsyncAtroposGRPOTrainer.register_with_atropos()`), not in the child process.

---

## Inheritance Hierarchy

```
transformers.TrainingArguments
  +-- _BaseConfig                         (trl.trainer.base_config)
       +-- AsyncGRPOConfig                (trl.experimental.async_grpo)
            +-- AsyncAtroposGRPOConfig    (trl.experimental.async_atropos_grpo)
                                               + atropos_api_url
                                               + atropos_group_size
                                               + atropos_batch_timeout
                                               + atropos_poll_interval
                                               + atropos_max_retries
                                               + atropos_max_inflight_batches
                                               + atropos_env_wandb_project
                                               + atropos_env_wandb_group
                                               + atropos_max_tokens

transformers.Trainer
  +-- _BaseTrainer                        (trl.trainer.base_trainer)
       +-- AsyncGRPOTrainer              (trl.experimental.async_grpo)
            +-- AsyncAtroposGRPOTrainer  (trl.experimental.async_atropos_grpo)
```

---

## The `RolloutWorkerProtocol` Pattern

The `AsyncGRPOTrainer` defines a **`RolloutWorkerProtocol`** -- an interface that any rollout worker must implement:

```python
class RolloutWorkerProtocol(Protocol):
    rollout_buffer: queue.Queue       # shared mp.Queue for RolloutSample objects

    def start(self) -> None: ...       # called on train begin
    def stop(self) -> None: ...        # called on train end
    def update_model_version(self, version: int) -> None: ...
    def check_health(self, stale_after_s: float) -> None: ...
```

This protocol-based design allows the trainer to accept any custom rollout backend:

- **`AsyncRolloutWorker`** (default): Spawns a child process that generates completions via vLLM HTTP and scores them with reward funcs.
- **`AtroposRolloutWorker`** (ours): Spawns a child process that polls the Atropos API for pre-scored trajectory batches.

Both produce the same `RolloutSample` objects:

```python
@dataclass(slots=True)
class RolloutSample:
    prompt: Messages                 # not used in loss, kept for logging
    completion: Messages             # not used in loss, kept for logging
    input_ids: list[int]             # full sequence (prompt + completion)
    completion_mask: list[int]       # 0 for prompt tokens, 1 for completion
    old_log_probs: list[float]       # logprobs from the generating policy
    advantage: float                 # group-relative GRPO advantage
    model_version: int              # which weight version generated this
    metrics: dict[str, float]       # reward, reward_std, etc.
```

The trainer's `compute_loss` is completely agnostic to which worker produced the samples -- it sees the same data format regardless.

---

## How It Differs from `AsyncGRPOTrainer`

| Aspect                                 | `AsyncGRPOTrainer`                                           | `AsyncAtroposGRPOTrainer`                                                                                                                     |
| -------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Rollout generation**           | Child process generates via vLLM HTTP `/v1/completions`      | Child process polls Atropos API `/batch`                                                                                                      |
| **Reward computation**           | Reward funcs applied in child process on generated completions | Scores come pre-computed from Atropos environment                                                                                               |
| **Rollout worker**               | `AsyncRolloutWorker`                                         | `AtroposRolloutWorker`                                                                                                                        |
| **vLLM server**                  | Uses `trl vllm-serve` with NCCL weight transfer              | Uses vanilla `vllm serve` with `VLLM_SERVER_DEV_MODE=1` and NCCL weight transfer                                                            |
| **Dataset requirement**          | `train_dataset` is required                                  | `train_dataset` is optional (prompts come from Atropos environment)                                                                           |
| **`reward_funcs` requirement** | Required                                                       | Optional -- defaults to pass-through that returns all zeros                                                                                     |
| **Weight transfer setup**        | Created automatically by parent                                | Created by subclass after `super().__init__()` because passing a custom `rollout_worker` causes the parent to skip weight transfer creation |

**Everything else is inherited unchanged:**

- `compute_loss()` -- full GRPO loss with clipping
- `_sync_weight()` -- NCCL weight transfer to vLLM dev-mode server
- `get_train_dataloader()` -- `RolloutQueueDataset` draining the shared queue
- `log()` -- metric averaging and NaN handling
- `_streaming_iter()` -- FSDP-compatible weight iteration
- `_inner_training_loop()` -- training loop lifecycle (with Atropos registration override)
- `evaluate()` -- overridden to skip evaluation; Atropos environment handles it
- Callbacks (`_InitialWeightSyncCallback`, `_StartRolloutWorkerCallback`, `StepIntervalCallback`)

---

## Configuration Reference

### `AsyncAtroposGRPOConfig`

Inherits all fields from `AsyncGRPOConfig`, which inherits from `_BaseConfig` (which inherits from `transformers.TrainingArguments`).

#### Atropos-specific fields

| Field                            | Type          | Default                     | Description                                                                            |
| -------------------------------- | ------------- | --------------------------- | -------------------------------------------------------------------------------------- |
| `atropos_api_url`              | `str`       | `"http://localhost:8000"` | Base URL of the Atropos `run-api` server                                             |
| `atropos_group_size`           | `int`       | `8`                       | Completions per prompt group. Must match the environment's `group_size`              |
| `atropos_batch_timeout`        | `float`     | `300.0`                   | Seconds to wait for a batch before `TimeoutError`                                    |
| `atropos_poll_interval`        | `float`     | `1.0`                     | Seconds between `/batch` polls when no data is available                             |
| `atropos_max_retries`          | `int`       | `3`                       | HTTP retries on transient failures                                                     |
| `atropos_max_inflight_batches` | `int`       | `2`                       | Max batches to buffer locally. Larger values smooth latency but risk stale policy data |
| `atropos_env_wandb_project`    | `str`/`None`| `None`                    | Name of wandb project for atropos env metrics, if enabled                              |
| `atropos_env_wandb_group`      | `str`/`None`| `None`                    | Name of wandb group for atropos env metrics                                            |
| `atropos_max_tokens`           | `int`       | `2048`                    | Max prompt+completion tokens. Good start is `max_completion_length` * 2               |

#### Inherited async pipeline fields (from `AsyncGRPOConfig`)

| Field                       | Type      | Default       | Description                                                                 |
| --------------------------- | --------- | ------------- | --------------------------------------------------------------------------- |
| `max_inflight_tasks`      | `int`   | `-1` (auto) | Max concurrent generation tasks. Auto =`max_staleness * samples_per_step` |
| `max_staleness`           | `int`   | `4`         | Max weight update steps a rollout can lag before being discarded            |
| `queue_maxsize`           | `int`   | `1024`      | Max rollout samples to buffer in the queue                                  |
| `weight_sync_steps`       | `int`   | `1`         | Optimizer steps between weight synchronizations to vLLM                     |
| `heartbeat_stale_after_s` | `float` | `300.0`     | Seconds without heartbeat after which the worker is treated as hung         |

#### Inherited vLLM server fields (from `AsyncGRPOConfig`)

| Field                    | Type      | Default                     | Description                                                                   |
| ------------------------ | --------- | --------------------------- | ----------------------------------------------------------------------------- |
| `vllm_server_base_url` | `str`   | `"http://localhost:8000"` | Base URL of the vLLM server (must be running with `VLLM_SERVER_DEV_MODE=1`) |
| `vllm_server_timeout`  | `float` | `240.0`                   | Seconds to wait for the vLLM server to be ready                               |
| `request_timeout`      | `int`   | `600`                     | Timeout for individual HTTP requests to vLLM                                  |

#### Inherited training fields (from `AsyncGRPOConfig`)

| Field                        | Type      | Default   | Description                                                           |
| ---------------------------- | --------- | --------- | --------------------------------------------------------------------- |
| `num_generations`          | `int`   | `8`     | Number of generations per prompt. Should match `atropos_group_size` |
| `max_completion_length`    | `int`   | `2048`  | Max tokens to generate per completion                                 |
| `temperature`              | `float` | `1.0`   | Sampling temperature                                                  |
| `epsilon`                  | `float` | `0.2`   | Lower-bound clipping epsilon                                          |
| `epsilon_high`             | `float` | `0.2`   | Upper-bound clipping epsilon (DAPO recommends `0.28`)               |
| `learning_rate`            | `float` | `1e-6`  | AdamW learning rate                                                   |
| `logging_steps`            | `float` | `1`     | Log every N steps                                                     |
| `log_completions`          | `bool`  | `False` | Whether to log prompt/completion samples                              |
| `num_completions_to_print` | `int`   | `3`     | Number of completions to print when logging                           |

---

## API Reference

### `AsyncAtroposGRPOTrainer` class

```python
class AsyncAtroposGRPOTrainer(AsyncGRPOTrainer):
    def __init__(
        self,
        model: str,
        train_dataset: Dataset | IterableDataset | None = None,
        args: AsyncAtroposGRPOConfig | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Any, Any] | None = None,
        **kwargs,
    )
```

**Parameters:**

- **`model`** (`str`) -- HuggingFace model ID or local path. Used to load the PyTorch model for training.
- **`args`** (`AsyncAtroposGRPOConfig`, optional) -- Configuration. If `None`, a default is created using the model name.
- **`train_dataset`** -- Optional. The actual data comes from the Atropos API, but a dataset satisfies the parent contract.
- **`callbacks`** -- Optional list of `TrainerCallback` objects.
- **`optimizers`** -- Optional tuple of (optimizer, scheduler). If None, defaults are used.

All other parameters are passed through to `AsyncGRPOTrainer.__init__` via `**kwargs`.

**Key methods:**

| Method                                  | Source    | Description                                                                             |
| --------------------------------------- | --------- | --------------------------------------------------------------------------------------- |
| `_inner_training_loop()`              | Override  | Registers with Atropos API on main process, then delegates to parent                    |
| `register_with_atropos()`             | New       | POST /register to Atropos API with trainer configuration                                |
| `_init_weight_transfer_client()`      | New       | Creates `WeightTransferClient` after parent skips it due to custom `rollout_worker` |
| `compute_loss()`                      | Inherited | Full GRPO loss with all variants (DAPO, DR-GRPO, SAPO, VESPO, LUSPO)                    |
| `_sync_weight()`                      | Inherited | NCCL weight broadcast to vLLM dev-mode server                                           |
| `get_train_dataloader()`              | Inherited | `RolloutQueueDataset` draining the shared queue                                       |
| `log()`                               | Inherited | Metric averaging and logging                                                            |
| `evaluate()`                          | Override  | Logs a message and returns empty dict (evaluation handled by Atropos environment)       |

### `AtroposRolloutWorker` class

```python
class AtroposRolloutWorker:
    def __init__(
        self,
        *,
        atropos_api_url: str = "http://localhost:8000",
        group_size: int = 8,
        batch_timeout: float = 300.0,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        max_inflight_batches: int = 2,
        queue_maxsize: int = 0,
        child_ready_timeout: int = 300,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
    )
```

**Public attributes (the `RolloutWorkerProtocol` interface):**

| Attribute          | Type         | Description                                             |
| ------------------ | ------------ | ------------------------------------------------------- |
| `rollout_buffer` | `mp.Queue` | Shared queue where `RolloutSample` objects are pushed |
| `model_version`  | `int`      | Current policy version (shared with child process)      |

| Method                            | Description                                                 |
| --------------------------------- | ----------------------------------------------------------- |
| `start()`                       | Spawn the child process and wait for it to signal readiness |
| `stop()`                        | Signal shutdown and join the child process                  |
| `check_health(stale_after_s)`   | Raise if the child crashed or heartbeat is stale            |
| `update_model_version(version)` | Propagate the new model version to the child                |

---

## The Atropos Batch Contract

The Atropos API's `/batch` endpoint returns a list of **group dicts**. Each group dict contains multiple parallel trajectories that form one prompt group (all generated from the same prompt).

### Required fields (from each group dict)

| Field                  | Type                  | Description                                                                                      |
| ---------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| `tokens`             | `list[list[int]]`   | Full token sequences (prompt + completion). One sequence per trajectory.                         |
| `masks`              | `list[list[int]]`   | Mask for each sequence:`-100` for prompt tokens, the actual token ID for completion tokens.    |
| `inference_logprobs` | `list[list[float]]` | Log-probabilities of each completion token under the generating policy. One list per trajectory. |
| `scores`             | `list[float]`       | Reward score per trajectory in the group.                                                        |

### Optional fields

| Field               | Type    | Description                                                            |
| ------------------- | ------- | ---------------------------------------------------------------------- |
| `env_id`          | `int` | Source environment identifier (logged for multi-environment awareness) |
| `prompt_text`     | `str` | Human-readable prompt (for logging)                                    |
| `completion_text` | `str` | Human-readable completion (for logging)                                |

### Validation rules enforced by `_convert_batch`

1. All fields within a group must have the same sequence count.
2. `len(tokens) == len(masks) == len(logprobs)` for every trajectory.
3. The total number of trajectories must be divisible by `atropos_group_size`.
4. Logprobs of `0.0` are treated as sentinels (meaning "not provided") and replaced with `-100.0` for numerical stability.

### Example group dict

```python
{
    "tokens": [
        [10, 20, 30, 40, 50, 60, 1],   # trajectory 0: 3 prompt + 4 completion
        [10, 20, 30, 40, 50, 60, 70, 80, 1],  # trajectory 1: 3 prompt + 6 completion
    ],
    "masks": [
        [-100, -100, -100, 50, 60, 1],  # -100 for prompt positions
        [-100, -100, -100, 70, 80, 1],
    ],
    "inference_logprobs": [
        [0.0, 0.0, 0.0, -0.3, -1.2, -0.05],  # 0.0 for prompt, real logprobs for completion
        [0.0, 0.0, 0.0, -0.1, -0.8, -2.0, -0.5],
    ],
    "scores": [1.0, 0.5],
    "env_id": 0,
}
```

---

## Loss Computation

The loss computation is **fully inherited** from `AsyncGRPOTrainer.compute_loss()`. It implements the standard GRPO clipped surrogate objective:

```python
log_ratio = log_probs - old_log_probs       # importance sampling ratio
ratio = torch.exp(log_ratio)
clipped = torch.clamp(ratio, 1 - e_low, 1 + e_high)
per_token_loss = -min(ratio * A, clipped * A)
```

Key details:

1. **Truncation to local max length** -- The collator pads to the global batch max length; each rank truncates to its local longest sequence before the forward pass.
2. **DAPO normalization** -- The loss is normalized by `global_n_tokens / world_size` (total completion tokens across all ranks divided by world size), ensuring per-token loss scaling regardless of sequence lengths.
3. **Gradient accumulation scaling** -- The loss is divided by `current_gradient_accumulation_steps` so gradients accumulate correctly across micro-batches.
4. **Metric tracking** -- The `no_grad` block computes and all-reduces:
   - `ratio` (mean importance sampling ratio)
   - `kl` (approximate KL divergence)
   - `entropy` (mean policy entropy)
   - `clip_ratio` (fraction of clipped tokens)
   - Reward metrics (from the worker's metrics dict)
   - `completions/mean_length`
   - `training_tok/s` (training throughput)
   - `forward_time_s` (time for the forward pass)

---

## Weight Synchronization

Weight sync uses the NCCL weight transfer engine built into vLLM. The trainer's `WeightTransferClient` connects to the vLLM server's engine to broadcast updated model weights.

**Important:** The vLLM server must be started with `VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config '{"backend":"nccl"}'`. This enables the required HTTP endpoints that `WeightTransferClient` calls.

**The `_sync_weight` flow:**

1. **Pause** -- POST `/pause` to the vLLM server (blocks new generation requests).
2. **Barrier** -- `accelerator.wait_for_everyone()` ensures all ranks are at the same step.
3. **NCCL send** -- The main process sends weights via `WeightTransferClient.send_weights()` (uses `NCCLWeightTransferEngine.trainer_send_weights`). Non-main processes participate in the `_streaming_iter()` collective for FSDP2.
4. **Barrier** -- Wait for NCCL transfer to complete on all ranks.
5. **Resume** -- POST `/resume` to the vLLM server (unblocks generation requests).
6. **Version bump** -- `model_version += 1`. The rollout worker's `update_model_version()` is called so the child process tags new samples with the correct version (used for staleness tracking).

**Critical implementation detail:** When a custom `rollout_worker` is passed to `AsyncGRPOTrainer.__init__`, the parent **skips creating `WeightTransferClient`** and sets `self.weight_transfer = None`. Our `AsyncAtroposGRPOTrainer` fixes this after `super().__init__()` by calling `_init_weight_transfer_client()`, which collects weight metadata from the loaded model and creates the client.

**What happens on failure:** The trainer logs a warning and continues. The vLLM server retains its previous weights. Training is not aborted because a single failed sync is recoverable.

---

## Advantage Computation

Advantages are computed in the **child process** (`_AtroposPollingLoop._convert_batch`) using standard Group Relative Policy Optimization normalization:

```
advantages_i = (score_i - mean(group)) / (std(group) + 1e-4)
```

Where each group is a set of `atropos_group_size` trajectories that were generated from the same prompt. This is identical to the formula used by `AsyncGRPOTrainer` internally, ensuring the downstream `compute_loss` behaves exactly the same.

The `1e-4` epsilon prevents division by zero when all completions in a group receive the same score.

**Important:** Because advantages are pre-computed in the child process and included in each `RolloutSample`, the trainer's `compute_loss` receives them directly. The `advantage` field in the collated batch is used as-is; no additional reward computation or normalization is performed in the trainer.

---

## Distributed Training

The trainer uses `accelerate` for distributed training, inherited from `AsyncGRPOTrainer` -> `_BaseTrainer` -> `transformers.Trainer`.

**DataLoader:** Uses `split_batches=True` and `dispatch_batches=True` in the accelerator config (set in `AsyncGRPOConfig.__post_init__`). This ensures:

- The main process drives the `RolloutQueueDataset` (only rank 0 has data).
- Accelerate's `DataLoaderDispatcher` broadcasts each batch to all ranks.

**Weight sync:** The NCCL collective in `_sync_weight()` requires all ranks to participate. Barriers ensure alignment.

**Launch commands:**

```bash
# Single GPU
python example.py

# Multi-GPU
accelerate launch --num_processes 4 example.py

# Multi-node (with torchrun)
torchrun --nproc_per_node=4 --nnodes=2 example.py
```

**GPU isolation (important):** The vLLM server and the trainer must run on **separate GPUs**. Partition your GPUs with `CUDA_VISIBLE_DEVICES`:

```bash
# GPUs 0-1 for vLLM server (tensor parallel)
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 4096 \
    --tensor-parallel-size 2 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'

# GPUs 2-3 for training (data parallel)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 example.py
```

---

## Metrics and Logging

The trainer reports all metrics inherited from `AsyncGRPOTrainer` plus the metrics from the Atropos rollout worker.

### Inherited metrics (from `compute_loss`)

| Metric key                        | Description                                        |
| --------------------------------- | -------------------------------------------------- |
| `train/ratio`                   | Mean importance sampling ratio                     |
| `train/kl`                      | Approximate KL divergence (per token)              |
| `train/entropy`                 | Mean policy entropy                                |
| `train/clip_ratio`              | Fraction of clipped tokens                         |
| `train/completions/mean_length` | Mean completion length in tokens                   |
| `train/training_tok/s`          | Training throughput (completion tokens per second) |
| `train/forward_time_s`          | Forward pass time per step                         |
| `train/train_seq_len`           | Local maximum sequence length                      |
| `train/weight_sync_time_s`      | Time for weight synchronization                    |

### Worker metrics (from `RolloutSample.metrics`)

| Metric key                  | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `train/reward`            | Mean raw environment score across the batch            |
| `train/reward_std`        | Standard deviation of raw scores                       |
| `train/reward_mean`       | Mean of scorable rewards                               |
| `train/queue_wait_time_s` | Time the dataloader waited for a sample from the queue |

### Logging configuration

```python
config = AsyncAtroposGRPOConfig(
    logging_steps=1,              # log every step
    log_completions=True,         # print prompt/completion samples
    num_completions_to_print=3,   # show 3 samples per log
    report_to="wandb",            # or "tensorboard"
)
```

---

## Troubleshooting

### `ConnectionError: Cannot reach Atropos API at http://localhost:8000`

The Atropos `run-api` server is not running or is not reachable. Start it with `run-api` and verify with:

```bash
curl http://localhost:8000/
```

### `TimeoutError: No batch available from Atropos API within 300s`

No Atropos environment is running, or the environment has crashed. Check that:

1. `run-api` shows environment connections in its logs.
2. The environment process is alive and generating.
3. The environment's `group_size` matches `atropos_group_size` in the config.

### `ValueError: Atropos batch size N is not divisible by group_size M`

The `group_size` configured in your Atropos environment does not match `atropos_group_size` in `AsyncAtroposGRPOConfig`. They must be equal.

### Weight sync failed / vLLM returns old model outputs

1. Verify the vLLM server was started with `VLLM_SERVER_DEV_MODE=1`. Without this, the weight transfer HTTP endpoints will not be available and `WeightTransferClient` will fail.
2. Verify `--max-model-len` is large enough to accommodate prompt + completion.
3. Check `vllm_server_base_url` in your config matches the server's address.
4. Check for NCCL errors in the logs (look for "weight_sync" log lines).
5. Increase `vllm_server_timeout` if the server takes long to load the model.

### CUDA out of memory

- Reduce `--max-model-len` on the vLLM server.
- Reduce `per_device_train_batch_size`.
- Enable `gradient_checkpointing=True`.
- Use LoRA/PEFT to reduce the training model's memory footprint.
- Separate vLLM and trainer across different GPUs.

### Child process crashes silently

The child process heartbeat becomes stale. The parent will raise `RuntimeError: Rollout worker heartbeat stale`. Check:

1. The child's stderr for tracebacks (visible in the parent's stderr via the `exception_info_queue`).
2. That `atropos_poll_interval` and `atropos_batch_timeout` are reasonable for your environment's latency.

---

## Design Decisions and Caveats

### 1. Custom `rollout_worker` bypasses parent weight transfer

When a custom `rollout_worker` is passed to `AsyncGRPOTrainer.__init__`, the parent skips creating `WeightTransferClient` and sets `self.weight_transfer = None`. Our `AsyncAtroposGRPOTrainer` explicitly handles this by calling `_init_weight_transfer_client()` after `super().__init__()` to create the client (only on the main process). Without this fix, weight sync would be a no-op and Atropos environments would always generate from the initial policy.

### 2. Pre-computed advantages in the worker

Advantages are computed in the child process (`_convert_batch`) rather than in the trainer. This means the advantage computation is fixed to group-relative GRPO normalization. If you need different advantage computation (e.g., batch-level normalization, per-reward-function weighting), override `_convert_batch` in a subclass of `AtroposRolloutWorker`.

### 3. Pass-through reward function

When no `reward_funcs` are provided, a pass-through that returns all zeros is used. This is because scores come from the Atropos environment and arrive as pre-computed advantages in each `RolloutSample`. The advantages tensor in the collated batch completely overrides any reward-derived values.

If you provide `reward_funcs`, they will be called during evaluation (which uses the standard in-process generation path), but are effectively ignored during training.

### 4. Staleness detection

Each `RolloutSample` carries a `model_version` reflecting the policy that generated it. The `RolloutQueueDataset` compares this against the current `model_version` (which increments with each weight sync). Samples with staleness > `max_staleness` are silently dropped.

This means if the vLLM server is slow to accept weight updates, some Atropos-generated trajectories may be discarded. Tune `max_staleness` and `weight_sync_steps` to match your environment's generation latency.

### 5. Dataset is optional

`train_dataset` is optional because prompts come from the Atropos environment's own dataset. However, the parent `AsyncGRPOTrainer` requires a dataset for the dataloader contract. If `train_dataset` is provided, it is forwarded to the parent and affects only the dataloader setup (the actual data comes from the queue). If omitted, the parent's `RolloutQueueDataset` drives the loop.

### 6. No in-process vLLM

Unlike `AsyncRolloutWorker` which calls vLLM HTTP from the child process, `AtroposRolloutWorker` does not need vLLM installed in the child process. It only needs `requests` and `numpy`. Weight sync still requires the `WeightTransferClient` (which uses vLLM's NCCL engine) on the trainer side.

### 7. Child process is synchronous

Unlike `AsyncRolloutWorker` which uses `asyncio` in the child process, `AtroposRolloutWorker` uses synchronous `requests`. This is simpler and sufficient because there is only one source of data (the Atropos API) rather than parallel generation tasks.

### 8. The stop event

The child process checks `_stop_event.is_set()` in its main loop and in `_wait_for_batch`. This ensures prompt shutdown when the trainer calls `stop()`, preventing the child from hanging indefinitely on a `_wait_for_batch` call after training ends.

### 9. Registration happens in the parent, not the child

Unlike the module docstring originally stated, the child process does not register with the Atropos API. Registration (`POST /register`) is performed by `AsyncAtroposGRPOTrainer.register_with_atropos()` in the parent trainer, called at the start of `_inner_training_loop()` on the main process only.

---

## File Reference

| File                              | Purpose                       | Key Classes                                       |
| --------------------------------- | ----------------------------- | ------------------------------------------------- |
| `__init__.py`                   | Public API exports            | --                                                |
| `async_atropos_grpo_config.py`  | Configuration dataclass       | `AsyncAtroposGRPOConfig`                        |
| `atropos_rollout_worker.py`     | Atropos-backed rollout worker | `AtroposRolloutWorker`, `_AtroposPollingLoop` |
| `async_atropos_grpo_trainer.py` | Trainer subclass              | `AsyncAtroposGRPOTrainer`                       |
| `example.py`                    | Minimal training example      | --                                                |
| `README.md`                     | This file                     | --                                                |

### Dependencies

- `trl` (with `async_grpo` and `weight_transfer` modules)
- `transformers`
- `accelerate`
- `datasets`
- `numpy`
- `requests`
- `torch`

The vLLM server (with `VLLM_SERVER_DEV_MODE=1`), Atropos API server (`run-api`), and Atropos environment microservices are external dependencies not included in this package.