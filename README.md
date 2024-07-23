# MMM
Multi-track music machine implementation

## Usage example

```Python
from pathlib import Path
from mmm import MMM  # model
from miditok import MMM as MMM_T  # tokenizer
from symusic import Score  # MIDI file parsing

# Creating the model and the tokenizer
model = MMM.from_pretrained("metacreation/MMM")
tokenizer = MMM_T.from_pretrained("metacreation/MMM")

# Loading a MIDI file
score = Score(Path("to", "file.mid"))
tokens = tokenizer(score)
# TODO complete inference example
# gen_tokens = model.generate
# gen_score = tokenizer.decode(gen_tokens)
```

## Steps to reproduce

Before running these commands, make sure to load a virtual Python environment if needed.

### On Compute Canada

On Compute Canada (Narval), all these steps can be reproduced by running:

```bash
# Unzip the dataset (you might need to remove macOS fork files in the archive) and install flash attention 2 simultaneously
sbatch slurm/unzip_gigamidi.sh && sbatch slurm/install_flashattention.sh
```

```bash
# Remove non-valid files then train the model
sbatch --wait slurm/preprocess_dataset.sh
sh scripts/train_model_loop.sh
```

### Install dependencies:

```bash
pip install ".[train]"
```

#### FlashAttention2

To install [FlashAttention2](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features):

```bash
pip install ninja
pip install flash-attn --no-build-isolation
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
python scripts/train_tokenizer.py
python scripts/train_model.py
```

## Data preprocessing

1. Filter non-valid files: corrupted or less than 8 bars;
2. Train the tokenizer on a subset of 100k files from the dataset, including Attribute Controls tokens computed for k randomly selected tracks and b randomly selected bars;
3. Split the dataset in train/valid/test subsets;
4. Split each file into chunks that make approximately 2048 tokens;
5. Augment each chunk on up/down to +-6 pitch intervals and -+2 velocities;
