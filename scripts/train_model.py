#!/usr/bin/python3 python

"""Train the MMM model."""

if __name__ == "__main__":
    from argparse import ArgumentParser

    from utils.baseline import mmm, mmm_seq2seq
    from utils.training import whole_training_process

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--seq2seq", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--no-torch-compile", dest="torch_compile", action="store_false"
    )
    parser.set_defaults(seq2seq=False)
    parser.set_defaults(deepspeed=False)
    parser.set_defaults(torch_compile=True)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument("--hf-repo-name", type=str, required=False, default=None)
    parser.add_argument("--hf-token", type=str, required=False, default=None)
    args = vars(parser.parse_args())

    baseline = mmm_seq2seq if args["seq2seq"] else mmm

    # Tweak configuration
    if args["hf_repo_name"]:
        baseline.training_config_kwargs["push_to_hub"] = True
        baseline.training_config_kwargs["hub_model_id"] = args["hf_repo_name"]
        baseline.training_config_kwargs["hub_token"] = args["hf_token"]
    if args["deepspeed"]:
        baseline.training_config_kwargs["deepspeed"] = "slurm/ds_config.json"
    for attr in ("per_device_train_batch_size", "per_device_eval_batch_size"):
        if args[attr]:
            baseline.training_config_kwargs[attr] = args[attr]
    baseline.training_config_kwargs["torch_compile"] = args["torch_compile"]

    # TODO introduce metrics: measure effectiveness of attribute controls
    """from metrics import Metrics, apply_argmax_to_preds
    metrics_names = {
        "accuracy": (apply_argmax_to_preds, {}, {}),
    }
    # Metrics(metrics_names, exp_id=exp_.name)"""

    whole_training_process(baseline, do_test=False)
