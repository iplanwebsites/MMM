"""Lists the Experiment baselines and training."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING

from miditok import TokenizerConfig
from miditok.pytorch_data import DataCollator, DatasetMIDI
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    MistralConfig,
)

from utils.classes import Baseline, DataConfig, TokenizationConfig
from utils.constants import (
    BATCH_SIZE_PER_DEVICE_TRAIN,
    BATCH_SIZE_PER_DEVICE_VALID,
    BF16,
    BF16_EVAL,
    DATA_AUGMENTATION_OFFSETS,
    DDP_BUCKET_CAP_MB,
    DDP_FIND_UNUSED_PARAMETERS,
    DEEPSPEED,
    EMBEDDING_SIZE,
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    EVAL_ACCUMULATION_STEPS,
    EVALUATION_STRATEGY,
    FEEDFORWARD_SIZE,
    FP16,
    FP16_EVAL,
    FULL_DETERMINISM,
    GRAD_ACC_STEPS,
    GRADIENT_CHECKPOINTING,
    GRADIENT_CLIP_NORM,
    HUB_PRIVATE_REPO,
    HUB_STRATEGY,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOG_LEVEL,
    LOG_STEPS_INTVL,
    LOGGING_STRATEGY,
    LR_SCHEDULER,
    MAX_POSITION_EMBEDDINGS,
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    NUM_ATTENTION_HEADS,
    NUM_BEAMS,
    NUM_INFERENCES_EVAL,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_TRAIN_EPOCHS,
    REPETITION_PENALTY,
    SAVE_STEPS,
    SAVE_STRATEGY,
    SAVE_TOTAL_LIMIT,
    SEED,
    SLIDING_WINDOWS,
    TEMPERATURE_SAMPLING,
    TEST_SPLIT,
    TOKENIZER_PARAMS,
    TOP_K,
    TOP_P,
    TORCH_COMPILE,
    TORCH_COMPILE_BACKEND,
    TORCH_COMPILE_MODE,
    USE_CUDA,
    USE_MPS,
    VALID_INTVL,
    VALID_SPLIT,
    VOCAB_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from transformers import PreTrainedModel


class MMM(Baseline):
    """MMM model baseline."""

    def create_dataset(self, files_paths: Sequence[Path]) -> DatasetMIDI:
        """
        Create a ``pytorch.utils.data.Dataset`` to use to train/test a model.

        :param files_paths: paths of the files to use.
        """
        return DatasetMIDI(files_paths, self.tokenizer, self.data_config.max_seq_len)

    def create_data_collator(self, pad_on_left: bool = False) -> DataCollator:
        """Create a data collator to use with a ``pytorch.utils.data.DataLoader``."""
        return DataCollator(
            self.pad_token_id,
            pad_on_left=pad_on_left,
            copy_inputs_as_labels=True,
        )

    def create_model(self, pretrained: str | None = None, **kwargs) -> PreTrainedModel:
        """
        Create the model of the baseline.

        :param pretrained: path of the model to load. If ``None`` is given, the model is
            created untrained.
        :param kwargs: any additional keyword arguments that should be provided.
        """
        if pretrained is not None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_config(self.model_config, **kwargs)
        # model = BetterTransformer.transform(model, keep_original_model=False)
        model.generation_config = self.generation_config
        return model


# TODO pretrain without Attr controls then finetune with?
# Create config objects
training_config_kwargs = {
    "output_dir": "",  # overridden by Baseline class
    "overwrite_output_dir": False,
    "do_train": True,
    "do_eval": True,
    "do_predict": False,
    "evaluation_strategy": EVALUATION_STRATEGY,
    "per_device_train_batch_size": BATCH_SIZE_PER_DEVICE_TRAIN,
    "per_device_eval_batch_size": BATCH_SIZE_PER_DEVICE_VALID,
    "gradient_accumulation_steps": GRAD_ACC_STEPS,
    "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
    "eval_steps": VALID_INTVL,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": GRADIENT_CLIP_NORM,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    # "max_steps": args["training_steps"],
    "lr_scheduler_type": LR_SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,
    "log_level": LOG_LEVEL,
    "logging_strategy": LOGGING_STRATEGY,
    "logging_steps": LOG_STEPS_INTVL,
    "save_strategy": SAVE_STRATEGY,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "use_cpu": not USE_CUDA,
    "seed": SEED,
    "bf16": BF16,
    "bf16_full_eval": BF16_EVAL,
    "fp16": FP16,
    "fp16_full_eval": FP16_EVAL,
    "local_rank": int(os.getenv("LOCAL_RANK", -1)),  # for DDP
    "disable_tqdm": True,
    "load_best_model_at_end": True,
    "label_smoothing_factor": LABEL_SMOOTHING,
    "optim": "adamw_torch",
    "report_to": ["tensorboard"],  # logging_dir will be set within Baseline class
    "ddp_find_unused_parameters": DDP_FIND_UNUSED_PARAMETERS,
    "ddp_bucket_cap_mb": DDP_BUCKET_CAP_MB,
    "deepspeed": DEEPSPEED,  # set with argparse
    "hub_strategy": HUB_STRATEGY,
    "hub_private_repo": HUB_PRIVATE_REPO,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "full_determinism": FULL_DETERMINISM,
    "use_mps_device": USE_MPS,
    "torch_compile": TORCH_COMPILE,
    "torch_compile_backend": TORCH_COMPILE_BACKEND,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "predict_with_generate": False,
}
data_config = DataConfig(
    VALID_SPLIT, TEST_SPLIT, DATA_AUGMENTATION_OFFSETS, MIN_SEQ_LEN, MAX_SEQ_LEN
)
tok_config = TokenizationConfig(
    "MMM", TokenizerConfig(**deepcopy(TOKENIZER_PARAMS)), VOCAB_SIZE
)
model_config = MistralConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=EMBEDDING_SIZE,
    intermediate_size=FEEDFORWARD_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    sliding_window=SLIDING_WINDOWS,
    use_cache=False,  # for gradient checkpointing during training
)
generation_config = GenerationConfig(
    max_new_tokens=NUM_INFERENCES_EVAL,
    num_beams=NUM_BEAMS,
    do_sample=True,
    temperature=TEMPERATURE_SAMPLING,
    top_k=TOP_K,
    top_p=TOP_P,
    epsilon_cutoff=EPSILON_CUTOFF,
    eta_cutoff=ETA_CUTOFF,
    repetition_penalty=REPETITION_PENALTY,
    use_cache=True,
)

# exp -> Model size, baseline -> pretrained + finetune
dataset = "MMD"
mmm = MMM(
    dataset,
    SEED,
    deepcopy(tok_config),
    deepcopy(model_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)
