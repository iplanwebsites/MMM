# MMM
Multi-track music machine implementation

## Steps to reproduce

Before running these commands, make sure to load a virtual Python environment if needed.

### Install dependencies:
```bash
pip install ".[train]"
```

### Prepare data and train

#### On a Slurm cluster

It will use DeepSpeed to train the model on multiple GPUs.

```bash
sbatch slurm/preprocess_dataset.sh
sbatch slurm/train_model.sh hf_token
```

#### Pure Python

```bash
python scripts/preprocess_dataset.py
python scripts/train_model.py
```

## Data preprocessing

1. Train the tokenizer on the whole non-preprocessed dataset;
2. Split the MIDI files in train/valid/test subsets;
3. Filter invalid files (corrupted, empty files, files with less than 4 bars);
4. Split each file into chunks that make approximately 2048 tokens;
5. Augment each chunk on +-6 pitch intervals and -+2 velocities;
6.
