"""Keep end-of-epoch weights when checkpoints are saved by steps.

A Stage-1 epoch on 2 x H200 outlasts the 8 h job limit, so runs save every N steps with a
small `save_total_limit` (each checkpoint is ~45 GB of DeepSpeed state). The authors pick
the best *epoch* checkpoint by Math-500, so this callback forces a save at every epoch end
and hard-links that checkpoint's `hf/` and `lora_adapter/` into `<keep_dir>/epoch-<k>-step-<n>/`,
where checkpoint rotation cannot delete them. Enabled by `LSFT_KEEP_EPOCHS_DIR`.
"""
import logging
import os
import shutil

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

__all__ = ["KeepEpochCheckpoints"]


class KeepEpochCheckpoints(TrainerCallback):
    def __init__(self, keep_dir: str) -> None:
        self.keep_dir = keep_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.epoch is None:
            return
        if abs(state.epoch - round(state.epoch)) > 1e-3:
            return
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(self.keep_dir, f"epoch-{round(state.epoch)}-step-{state.global_step}")
        if os.path.exists(dst):
            return
        # LSFT_KEEP_HF_EVERY=k keeps the merged hf/ (~15 GB) only every k-th epoch; the LoRA
        # adapter is always kept (a long Stage-2 run has 70 epochs).
        every = int(os.environ.get("LSFT_KEEP_HF_EVERY", "1"))
        for sub in ("hf", "lora_adapter"):
            if sub == "hf" and round(state.epoch) % every:
                continue
            if os.path.isdir(os.path.join(src, sub)):
                shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub), copy_function=os.link)
        logger.warning("kept epoch %s weights: %s", round(state.epoch), dst)
