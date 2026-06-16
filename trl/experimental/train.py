from atropos_grpo import make_atropos_trainer

trainer = make_atropos_trainer(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    atropos_api_url="http://localhost:8000",
    group_size=4,
    per_device_train_batch_size=4,
    max_steps=100,
    output_dir="./my_run",
    vllm_server_host="0.0.0.0",
    vllm_server_port=8001,
    # Pass any AtroposGRPOConfig fields as extra kwargs:
    extra_config_kwargs={
        "beta": 0.01,
        "learning_rate": 5e-7,
        "bf16": True,
        "max_completion_length": 2048
    },
)
trainer.train()