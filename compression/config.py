from pathlib import Path
import torch

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "fish4knowledge"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset
# ==========================================================

NUM_CLASSES = 23

IMAGE_SIZE = 224

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==========================================================
# Training
# ==========================================================

SEED = 42

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

LABEL_SMOOTHING = 0.1

# ==========================================================
# Hardware
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 8

PIN_MEMORY = DEVICE.type == "cuda"

USE_AMP = DEVICE.type == "cuda"

# ==========================================================
# Checkpoint
# ==========================================================

SAVE_BEST_ONLY = True

CHECKPOINT_NAME = "best_model.pth"

# ==========================================================
# Evaluation
# ==========================================================

TOPK = (1, 5)

CONFUSION_MATRIX_DPI = 300

FIGURE_SIZE = (12, 10)

# ==========================================================
# Model
# ==========================================================

EMBED_DIM = 384

NUM_HEADS = 8

TRANSFORMER_DEPTH = (3, 3, 2)

KNN_NEIGHBORS = 6

DROPOUT = 0.1

# ==========================================================
# BlockTT
# ==========================================================

class BlockTTConfig:

    def __init__(
        self,
        max_rank,
        row_blocks=2,
        col_blocks=2,
    ):
        self.max_rank = max_rank
        self.row_blocks = row_blocks
        self.col_blocks = col_blocks

# ==========================================================
# Model
# ==========================================================

EMBED_DIM = 384

NUM_HEADS = 8

TRANSFORMER_DEPTH = (3, 3, 2)

KNN_NEIGHBORS = 6

DROPOUT = 0.1

# ==========================================================
# BlockTT
# ==========================================================

class BlockTTConfig:

    def __init__(
        self,
        max_rank,
        row_blocks=2,
        col_blocks=2,
    ):
        self.max_rank = max_rank
        self.row_blocks = row_blocks
        self.col_blocks = col_blocks