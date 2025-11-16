# training/selfplay/config.py

from pathlib import Path

# Path to your ChessHacks project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to your current best supervised model
# (used both by the engine and as the INIT weights for RL fine-tuning)
MODEL_PATH = Path(r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\src\model_save\best.pt")

# Path to Stockfish binary (adjust if needed)
STOCKFISH_PATH = Path(r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\src\stockfish-windows-x86-64-avx2.exe")

# Output directory for self-play datasets
SELFPLAY_OUT_DIR = Path(r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\selfPlay\selfPlayData")
SELFPLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# training/selfplay/config.py  (example snippet)

from pathlib import Path


NUM_GAMES = 10              # default for RL; can override with --games-per-cycle
TOP_K_MOVES = 10
MAX_MOVES_PER_GAME = 120

SF_TIME_LIMIT = 0.05        # base; rl_loop will cap at 0.03 for speed
SELFPLAY_ENGINE_DEPTH = 3   # shallower than your tournament engine depth
POLICY_TEMP_CP = 400.0

# NN / policy config (must match your engine / model)
NUM_PLANES = 18
NUM_PROMOS = 5
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0

# Self-play engine search depth (smaller than tournament depth to keep it fast)
