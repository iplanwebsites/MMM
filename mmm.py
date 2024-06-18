"""Lists the Experiment baselines and training."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import TYPE_CHECKING

from miditok import TokenizerConfig
from miditok.pytorch_data import DataCollator
from tokentamer import Controller, ControllerConfig
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    MistralConfig,
)

from utils.classes import Baseline, DataConfig, TokenizationConfig
from utils.constants import (
    AC_BAR_DENSITY,
    AC_BAR_NOTE_DURATION,
    AC_BAR_POLYPHONY,
    AC_PITCH_LEVEL,
    AC_TRACK_DENSITY,
    AC_TRACK_NOTE_DURATION,
    AC_TRACK_POLYPHONY,
    ACS_RANDOM_RATIO_RANGE,
    BAR_DENSITY_MAX,
    BARS_IDX_RANDOM_RATIO_RANGE,
    BATCH_SIZE_PER_DEVICE_TRAIN,
    BATCH_SIZE_PER_DEVICE_VALID,
    BF16,
    BF16_EVAL,
    DATA_AUGMENTATION_OFFSETS,
    DDP_BUCKET_CAP_MB,
    DDP_FIND_UNUSED_PARAMETERS,
    DEEPSPEED,
    DISABLE_TQDM,
    EMBEDDING_SIZE,
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    EVAL_ACCUMULATION_STEPS,
    EVAL_STRATEGY,
    FEEDFORWARD_SIZE,
    FP16,
    FP16_EVAL,
    FULL_DETERMINISM,
    GRAD_ACC_STEPS,
    GRADIENT_CHECKPOINTING,
    GRADIENT_CLIP_NORM,
    HALF_PRECISION_BACKEND,
    HUB_PRIVATE_REPO,
    HUB_STRATEGY,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOAD_BEST_MODEL_AT_END,
    LOG_LEVEL,
    LOG_STEPS_INTVL,
    LOGGING_STRATEGY,
    LR_SCHEDULER,
    MAX_POSITION_EMBEDDINGS,
    MAX_SEQ_LEN,
    NEFTUNE_NOISE_ALPHA,
    NUM_ATTENTION_HEADS,
    NUM_BEAMS,
    NUM_INFERENCES_EVAL,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_TRAIN_EPOCHS,
    POLYPHONY_MAX,
    POLYPHONY_MIN,
    PUSH_TO_HF_HUB,
    REPETITION_PENALTY,
    REPORT_TO,
    SAVE_SAFETENSOR,
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
    TRACK_DENSITY_MAX,
    TRACK_DENSITY_MIN,
    TRACKS_IDX_RANDOM_RATIO_RANGE,
    TRACKS_SELECTION_RANDOM_RATIO_RANGE,
    TRAINING_STEPS,
    USE_CUDA,
    USE_MPS,
    VALID_DELAY,
    VALID_SPLIT,
    VOCAB_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from utils.data_loading import DatasetMMM

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from miditok import MusicTokenizer
    from symusic import Score
    from transformers import PreTrainedModel


CONTROLLER_CONFIG = ControllerConfig(
    bar_polyphony=AC_BAR_POLYPHONY,
    track_polyphony=AC_TRACK_POLYPHONY,
    polyphony_min=POLYPHONY_MIN,
    polyphony_max=POLYPHONY_MAX,
    pitch_level=AC_PITCH_LEVEL,
    track_density_level=AC_TRACK_DENSITY,
    track_density_max=TRACK_DENSITY_MAX,
    track_density_min=TRACK_DENSITY_MIN,
    bar_density_level=AC_BAR_DENSITY,
    bar_density_max=BAR_DENSITY_MAX,
    bar_note_duration=AC_BAR_NOTE_DURATION,
    track_note_duration=AC_TRACK_NOTE_DURATION,
)


class MMM(Baseline):
    """MMM model baseline."""

    def create_tokenizer(self) -> MusicTokenizer:
        """
        Create the tokenizer of the baseline.

        :return: tokenizer of the baseline.
        """
        tokenizer = super().create_tokenizer()
        self.controller = Controller(CONTROLLER_CONFIG, tokenizer)
        return tokenizer

    def create_dataset(self, files_paths: Sequence[Path]) -> DatasetMMM:
        """
        Create a ``pytorch.utils.data.Dataset`` to use to train/test a model.

        :param files_paths: paths of the files to use.
        """
        return DatasetMMM(
            files_paths,
            self.controller,
            self.data_config.max_seq_len,
            TRACKS_SELECTION_RANDOM_RATIO_RANGE,
            ACS_RANDOM_RATIO_RANGE,
            TRACKS_IDX_RANDOM_RATIO_RANGE,
            BARS_IDX_RANDOM_RATIO_RANGE,
        )

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
            created untrained. (default: ``None``)
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

    def generate_new_track(self, score: Score, program: int) -> None:
        """
        Generate a new track of a given Score.

        The new track will be generated with the model and
        TODO might need to provide a few additional arguments like a GenerationConfig

        :param score: ``symusic.Score`` to generate a new track from.
        :param program: program of the track to generate. Give ``-1`` for drums.
        """
        # TODO implement: tokenize --> inject Infilling --> generate --> detok...

    def generate_infilling(
        self, score: Score, bar_idx: tuple[int, int], track_idx: Sequence[int]
    ) -> None:
        """
        Generate a new portion of a ``symusic.Score``.

        The portion to infill will be generated with the model and added to the score
        inplace for the selected tracks. Notes originally present in the portion to
        infill will be removed.
        TODO might need to provide a few additional arguments like a GenerationConfig

        :param score: ``symusic.Score`` to generate a new track from.
        :param bar_idx: tuple of bar numbers of the portion of the score to infill.
        :param track_idx: indexes of the tracks to infill.
        """
        # TODO implement: tokenize --> inject Infilling --> generate --> detok...


# TODO pretrain without Attr controls then finetune with?
# Create config objects
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
training_config_kwargs = {
    "output_dir": "",  # overridden by Baseline class
    "overwrite_output_dir": False,
    "do_train": True,
    "do_eval": True,
    "do_predict": False,
    "eval_strategy": EVAL_STRATEGY,
    "per_device_train_batch_size": BATCH_SIZE_PER_DEVICE_TRAIN,
    "per_device_eval_batch_size": BATCH_SIZE_PER_DEVICE_VALID,
    "gradient_accumulation_steps": GRAD_ACC_STEPS,
    "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
    "eval_delay": VALID_DELAY,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": GRADIENT_CLIP_NORM,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": TRAINING_STEPS,
    "lr_scheduler_type": LR_SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,
    "log_level": LOG_LEVEL,
    "logging_strategy": LOGGING_STRATEGY,
    "logging_steps": LOG_STEPS_INTVL,
    "save_strategy": SAVE_STRATEGY,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "save_safetensors": SAVE_SAFETENSOR,
    "use_cpu": not (USE_CUDA or USE_MPS),
    "seed": SEED,
    "data_seed": SEED,
    "bf16": BF16,
    "bf16_full_eval": BF16_EVAL,
    "fp16": FP16,
    "fp16_full_eval": FP16_EVAL,
    "half_precision_backend": HALF_PRECISION_BACKEND,
    "local_rank": int(os.getenv("LOCAL_RANK", -1)),  # for DDP
    "disable_tqdm": DISABLE_TQDM,
    "load_best_model_at_end": LOAD_BEST_MODEL_AT_END,
    "label_smoothing_factor": LABEL_SMOOTHING,
    "optim": "adamw_torch",
    "report_to": REPORT_TO,
    "ddp_find_unused_parameters": DDP_FIND_UNUSED_PARAMETERS,
    "ddp_bucket_cap_mb": DDP_BUCKET_CAP_MB,
    "push_to_hub": PUSH_TO_HF_HUB,
    "deepspeed": DEEPSPEED,  # set with argparse
    "hub_strategy": HUB_STRATEGY,
    "hub_private_repo": HUB_PRIVATE_REPO,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "full_determinism": FULL_DETERMINISM,
    "use_mps_device": USE_MPS,
    "torch_compile": TORCH_COMPILE,
    "torch_compile_backend": TORCH_COMPILE_BACKEND,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "neftune_noise_alpha": NEFTUNE_NOISE_ALPHA,
    "predict_with_generate": False,
}
data_config = DataConfig(
    VALID_SPLIT, TEST_SPLIT, DATA_AUGMENTATION_OFFSETS, MAX_SEQ_LEN
)
tok_config = TokenizationConfig(
    "MMM", TokenizerConfig(**deepcopy(TOKENIZER_PARAMS)), VOCAB_SIZE
)
attn_implem = "flash_attention_2" if "flash_attn" in sys.modules else None
model_config = MistralConfig(  # TODO mixtral?
    vocab_size=VOCAB_SIZE,
    hidden_size=EMBEDDING_SIZE,
    intermediate_size=FEEDFORWARD_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    sliding_window=SLIDING_WINDOWS,
    use_cache=False,  # for gradient checkpointing during training
    attn_implementation=attn_implem,
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
dataset = "GigaMIDI"
mmm = MMM(
    dataset,
    SEED,
    deepcopy(tok_config),
    deepcopy(model_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)
