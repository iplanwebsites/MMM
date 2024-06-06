#!/usr/bin/python3 python

"""Train the MMM model."""


if __name__ == "__main__":
    from argparse import ArgumentParser

    from mmm import mmm
    from utils.constants import DEEPSPEED
    from utils.training import whole_training_process

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument(
        "--deepspeed", type=str, help="", required=False, default=DEEPSPEED
    )
    parser.add_argument("--torch-compile", help="", required=False, action="store_true")
    parser.add_argument(
        "--no-torch-compile", help="", dest="torch_compile", action="store_false"
    )
    parser.add_argument(
        "--per-device-batch-size-train",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--per-device-batch-size-test",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--hf-repo-name", type=str, help="", required=False, default=None
    )
    parser.add_argument("--hf-token", type=str, help="", required=False, default="?")
    parser.set_defaults(deepspeed=True)
    parser.set_defaults(torch_compile=True)
    args = vars(parser.parse_args())

    # Tweak configuration
    if args["hf_repo_name"]:
        mmm.training_config_kwargs["hub_model_id"] = args["hf_repo_name"]
        mmm.training_config_kwargs["hub_token"] = args["hf_token"]
        # TODO make sure the trainer creates with safe tensors
    if args["deepspeed"]:
        mmm.training_config_kwargs["deepspeed"] = "slurm/ds_config.json"
    for attr in ("per_device_batch_size_train", "per_device_batch_size_test"):
        if args[attr]:
            mmm.training_config_kwargs[attr] = args[attr]

    """from metrics import Metrics, apply_argmax_to_preds
    metrics_names = {
        "accuracy": (apply_argmax_to_preds, {}, {}),
    }
    # Metrics(metrics_names, exp_id=exp_.name)"""

    whole_training_process(mmm, do_test=False)
