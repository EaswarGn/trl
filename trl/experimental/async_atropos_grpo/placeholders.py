from torch.utils.data import IterableDataset

class _DummyIterableDataset(IterableDataset):
    """Minimal empty iterable dataset used as a placeholder.

    ``AsyncGRPOTrainer`` guards against ``train_dataset is None`` but never
    actually iterates ``self.train_dataset`` — its ``get_train_dataloader``
    builds a ``RolloutQueueDataset`` from ``self.rollout_queue`` instead.
    This dummy satisfies that guard callers who do not supply a dataset.
    """

    def __iter__(self):
        return iter([])
    
# Use pass-through reward funcs — Atropos provides the actual scores.
def _passthrough_reward(**kw) -> list[float]:
    prompts = kw.get("prompts", kw.get("prompt", None))
    if prompts is not None:
        return [0.0] * len(prompts)
    return []