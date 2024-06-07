"""Constants file."""
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.cuda import is_bf16_supported

SEED = 444
MODEL_NAME = "MMM"
HF_USERNAME = "Metacreation"

# For MMD preprocessing
MIN_NUM_BARS_FILE_VALID = 4
DATA_AUGMENTATION_OFFSETS = (6, 2, 0)  # pitch, velocity, duration
DATA_CHUNK_NUM_OVERLAP_BARS = 1


# Tokenizer params (same as MidiTok expect for new constants)
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": True,
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 95 + 1)),  # TODO plot distribution of programs
}
VOCAB_SIZE = 30000
TRAINING_MAX_NB_FILES = 100000

TIME_DIVISION = 384


# MODEL SIZE
MAX_POSITION_EMBEDDINGS = 8192
EMBEDDING_SIZE = 768
FEEDFORWARD_SIZE = EMBEDDING_SIZE * 4
NUM_LAYERS = 24
NUM_ATTENTION_HEADS = 12
NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS // 2
SLIDING_WINDOWS = 384


# DATA CONFIGS
MIN_SEQ_LEN = 512
MAX_SEQ_LEN = 2048
VALID_SPLIT = 0.02
TEST_SPLIT = 0.05


# TRAINING PARAMS
DROPOUT = 0.1
BATCH_SIZE_PER_DEVICE_TRAIN = 64  # multiple of 64 for A100, 8 for other GPUs (V100)
BATCH_SIZE_PER_DEVICE_VALID = 128
VALID_DELAY = 500
GRAD_ACC_STEPS = 1
EVALUATION_STRATEGY = "steps"
EVAL_ACCUMULATION_STEPS = None  # in case of CUDA OOM during eval
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 3.0
LABEL_SMOOTHING = 0.0
NUM_TRAIN_EPOCHS = 20
TRAINING_STEPS = -1  # unused
USE_CUDA = True
USE_MPS = False
BF16 = True  # Ampere and newer
BF16_EVAL = True
FP16 = False
FP16_EVAL = False
HALF_PRECISION_BACKEND = "auto"
DEEPSPEED = None  # set with argparse
TORCH_COMPILE = True
TORCH_COMPILE_BACKEND = None  # default to "inductor"
TORCH_COMPILE_MODE = None
GRADIENT_CHECKPOINTING = True
# https://pytorch.org/docs/stable/distributed.html
# DDP_BACKEND = None
DDP_FIND_UNUSED_PARAMETERS = False
DDP_BUCKET_CAP_MB = None  # default to 25mb
FULL_DETERMINISM = True
LOG_LEVEL = "debug"
LOGGING_STRATEGY = "steps"
LOG_STEPS_INTVL = 50
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 8
SAVE_SAFETENSOR = True
LOAD_BEST_MODEL_AT_END = False
DISABLE_TQDM = True
LEARNING_RATE = 6e-5
LR_SCHEDULER = "cosine_with_restarts"
WARMUP_RATIO = 0.10
REPORT_TO = ["tensorboard"]  # logging_dir will be set within Baseline class
PUSH_TO_HF_HUB = False
HUB_STRATEGY = "every_save"
HUB_PRIVATE_REPO = True
NEFTUNE_NOISE_ALPHA = True


# TEST PARAMS (for generation)
NUM_INFERENCES_EVAL = 256  # for evaluation metrics
NUM_INFERENCES_TEST = MAX_SEQ_LEN  # for tests after training, saving the results
# we use MAX_SEQ_LEN to make levenshtein computations more faithful


# GENERATION CONFIG (for validation and tests)
NUM_BEAMS = 1
TEMPERATURE_SAMPLING = 0.8
REPETITION_PENALTY = 1.2
TOP_K = 10
TOP_P = 0.95
EPSILON_CUTOFF = None
ETA_CUTOFF = None


# in case no GPU is available
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = USE_CUDA = False
elif (BF16 or BF16_EVAL) and not is_bf16_supported():
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
if USE_CUDA or not mps_available():
    USE_MPS = False
