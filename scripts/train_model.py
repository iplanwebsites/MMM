#!/usr/bin/python3 python

"""Train the MMM model."""

if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from inspect import signature

    from transformers import Seq2SeqTrainingArguments
    from utils.baselines import baselines
    from utils.training import whole_training_process

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--model", type=str, default="MMM_mistral")
    for param in signature(Seq2SeqTrainingArguments).parameters.values():
        key = param.name.replace("_", "-")
        if param.annotation is bool:
            parser.add_argument(f"--{key}", action=BooleanOptionalAction, default=None)
        else:
            parser.add_argument(f"--{key}", type=param.annotation, default=None)
    args = vars(parser.parse_args())

    # Identify model to train and tweak its training configuration
    baseline = baselines[args.pop("model")]
    for arg, value in args.items():
        if value is not None:
            baseline.training_config_kwargs[arg] = value

    # TODO introduce metrics: measure effectiveness of attribute controls
    """from metrics import Metrics, apply_argmax_to_preds
    metrics_names = {
        "accuracy": (apply_argmax_to_preds, {}, {}),
    }
    # Metrics(metrics_names, exp_id=exp_.name)"""

    # Training the model
    whole_training_process(baseline, do_test=False)
