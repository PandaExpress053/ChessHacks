# fix_best_checkpoint.py

from pathlib import Path
import torch

# Adjust if your path is slightly different, but this matches your config.py
MODEL_DIR = Path(r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\src\model_save")

last_path = MODEL_DIR / "last.pt"
best_path = MODEL_DIR / "best.pt"

print(f"[FIX] MODEL_DIR = {MODEL_DIR}")
print(f"[FIX] last.pt  = {last_path.exists()}")
print(f"[FIX] best.pt  = {best_path.exists()}")

if not last_path.exists():
    raise FileNotFoundError(f"last.pt not found at {last_path}")

ckpt = torch.load(last_path, map_location="cpu")

# If it's a rich checkpoint, extract the model weights
if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
    print("[FIX] Extracted 'model' subkey from last.pt")

    # Save clean state_dict so engine can load it
    torch.save(state_dict, best_path)
    print(f"[FIX] Wrote clean state_dict to {best_path}")

else:
    # If it's already a state_dict, just copy it through
    print("[FIX] last.pt does not look like a rich checkpoint; copying as-is.")
    torch.save(ckpt, best_path)
    print(f"[FIX] Overwrote {best_path} with last.pt contents")

print("[FIX] Done. Now restart your serve.py / dev server.")
