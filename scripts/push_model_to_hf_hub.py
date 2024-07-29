#!/usr/bin/python3 python

"""Push models to HF hub."""

from pathlib import Path

IGNORE_PATTERNS = [
    "*.DS_Store",
    "gen/*",
    "gen_lvl/*",
    "checkpoint-*",
    "config.json",
    "pytorch_model.bin",
    "generation_config.json",
]
MODEL_CARD_PATH = Path("docs", "model_card.md")


if __name__ == "__main__":
    import shutil
    from argparse import ArgumentParser

    from huggingface_hub import repo_exists, upload_folder
    from transformers.trainer_utils import get_last_checkpoint

    from utils.baseline import mmm, mmm_seq2seq

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--model", type=str, required=True, default=None)
    parser.add_argument("--hf-repo-name", type=str, required=True, default=None)
    parser.add_argument("--hf-token", type=str, required=True, default=None)
    args = vars(parser.parse_args())

    for baseline in [mmm, mmm_seq2seq]:
        if baseline.name == args["model"]:
            # Set checkpoints, lvl for lowest valid loss for pretraining
            checkpoint = get_last_checkpoint(baseline.run_path)

            if repo_exists(args["hf_repo_name"], token=args["hf_token"]):
                continue

            # Load model
            model_ = baseline.create_model(checkpoint)

            # Push to hub (model + tokenizer + training files)
            # We do not use the trainer as it does not push the weights as safe
            # tensors (yet)
            # https://github.com/huggingface/transformers/issues/25992
            model_.push_to_hub(
                repo_id=args["hf_repo_name"],
                private=True,
                token=args["hf_token"],
                commit_message=f"Uploading {baseline.name}",
                safe_serialization=True,
            )
            baseline.tokenizer.push_to_hub(
                repo_id=args["hf_repo_name"], token=args["hf_token"]
            )
            shutil.copy2(MODEL_CARD_PATH, baseline.run_path / "README.md")
            upload_folder(
                repo_id=args["hf_repo_name"],
                token=args["hf_token"],
                folder_path=baseline.run_path,
                ignore_patterns=IGNORE_PATTERNS,
            )
