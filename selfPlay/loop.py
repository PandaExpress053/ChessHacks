#!/usr/bin/env python3
"""
selfPlay/loop.py

Reinforcement Learning Loop using self-play NPZ generator.

Per cycle:
  1) Decide which weights to start from:
       - If best_selfplay.pt exists: start from that (continue RL).
       - Else: start from best.pt (supervised baseline).

  2) Generate self-play NPZ with generate_selfplay_npz(...)
       - Uses PolicyOnlyEngine + StockfishEvaluator internally.

  3) (Optional) Generate extra supervised mate-in-N data for this cycle
       - Uses generate_mateN_npz_for_cycle(...) (Stockfish-labeled)

  4) Train a new RL model on ALL NPZs in this cycle directory
       - train_selfplay_model(...) reads NPZs and writes best_selfplay.pt.

IMPORTANT:
  - best.pt (supervised baseline) is NEVER overwritten in RL.
  - RL_MODEL_PATH = best_selfplay.pt (lives next to best.pt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Imports from your training package
# ----------------------------------------------------------------------
from selfPlay.config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,          # supervised best.pt
    NUM_GAMES,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    SF_TIME_LIMIT,
)

from selfPlay.selfplayGEN import (
    generate_selfplay_npz,
    BLUNDER_PLAY_PROB,
    SF_GUIDE_GAME_PROB,
)

from selfPlay.trainselfplay import train_selfplay_model
from selfPlay.mategen import generate_mateN_npz_for_cycle


# RL weights: live next to best.pt
SUPER_MODEL_PATH = Path(MODEL_PATH)
RL_MODEL_PATH = SUPER_MODEL_PATH.with_name("best_selfplay.pt")

SELFPLAY_OUT_DIR = Path(SELFPLAY_OUT_DIR)


# ======================================================================
# Utils
# ======================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_start_weights() -> Path:
    """
    Decide which weights to start RL from.

    Priority:
      1) If best_selfplay.pt exists, use that (continue RL).
      2) Otherwise, use best.pt (supervised baseline).

    Note: This does NOT control which weights selfplay_generator uses
    internally (PolicyOnlyEngine may still use MODEL_PATH from config),
    but it DOES control which weights we train from and where we save
    the updated RL model.
    """
    if RL_MODEL_PATH.exists():
        print(f"[RL] Found existing RL weights: {RL_MODEL_PATH}")
        return RL_MODEL_PATH
    else:
        print(f"[RL] No RL weights yet; starting from supervised: {SUPER_MODEL_PATH}")
        return SUPER_MODEL_PATH


# ======================================================================
# Main RL loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play RL loop")

    # Number of RL cycles
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of RL self-play + train cycles.",
    )

    # Games per cycle (default from config)
    parser.add_argument(
        "--games-per-cycle",
        type=int,
        default=NUM_GAMES,
        help="Number of self-play games per RL cycle.",
    )

    # Optional device override
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Default: auto-detect.",
    )

    # ------------------------------------------------------------------
    # Mate-in-N supervised augmentation (optional)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mate-supervised",
        action="store_true",
        help="Also generate supervised mate-in-N data each RL cycle.",
    )
    parser.add_argument(
        "--mate-samples",
        type=int,
        default=2000,
        help="Number of mate-in-N positions per cycle (if enabled).",
    )
    parser.add_argument(
        "--mate-max-dist",
        type=int,
        default=8,
        help="Keep only positions with |mate_distance| <= this.",
    )
    parser.add_argument(
        "--sf-engine",
        type=str,
        default=None,
        help="Path to Stockfish binary for mate-in-N generator "
             "(required if --mate-supervised is used).",
    )

    # Optional schedule scale for blunders / SF guidance over cycles
    parser.add_argument(
        "--blunder-scale",
        type=float,
        default=1.0,
        help="Scale factor for blunder probability across cycles "
             "(1.0 = no change).",
    )
    parser.add_argument(
        "--sf-guide-scale",
        type=float,
        default=1.0,
        help="Scale factor for SF-guided-game probability across cycles "
             "(1.0 = no change).",
    )

    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else get_device()
    print(f"[RL] Using device (for training): {device}")

    if args.mate_supervised and args.sf_engine is None:
        raise SystemExit(
            "You enabled --mate-supervised but did not provide --sf-engine "
            "(path to Stockfish)."
        )

    start_weights_path = choose_start_weights()

    # ------------------------------------------------------------------
    # Main RL cycles
    # ------------------------------------------------------------------
    for cycle in range(args.cycles):
        print("=" * 70)
        print(f"[RL] Starting cycle {cycle} / {args.cycles - 1}")
        print("=" * 70)

        # Directory for this cycle's data
        cycle_dir = SELFPLAY_OUT_DIR / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RL] Cycle output directory: {cycle_dir}")

        # --------------------------------------------------------------
        # 1) Generate self-play NPZ for this cycle
        # --------------------------------------------------------------
        out_npz = cycle_dir / f"cycle_{cycle:03d}_selfplay.npz"

        # Optionally schedule blunder/SF-guided probabilities over cycles
        eff_blunder = BLUNDER_PLAY_PROB * (args.blunder_scale ** cycle)
        eff_sf_guide = SF_GUIDE_GAME_PROB * (args.sf_guide_scale ** cycle)

        print(f"[RL] Generating self-play NPZ at {out_npz}")
        print(
            f"[RL]  blunder_prob={eff_blunder:.3f}, "
            f"sf_guide_prob={eff_sf_guide:.3f}"
        )

        generate_selfplay_npz(
            out_path=out_npz,
            num_games=args.games_per_cycle,
            top_k=TOP_K_MOVES,
            max_moves=MAX_MOVES_PER_GAME,
            sf_time_limit=SF_TIME_LIMIT,
            blunder_prob=eff_blunder,
            sf_guide_prob=eff_sf_guide,
            # conversion_cp_threshold left default (800) for now
        )

        # --------------------------------------------------------------
        # 1b) (OPTIONAL) Generate mate-in-N supervised NPZ
        # --------------------------------------------------------------
        if args.mate_supervised:
            print(f"[RL] Generating supervised mate-in-N shard for cycle {cycle}...")
            generate_mateN_npz_for_cycle(
                cycle_idx=cycle,
                out_dir=cycle_dir,
                engine_path=args.sf_engine,
                num_positions=args.mate_samples,
                max_mate_distance=args.mate_max_dist,
                time_limit=SF_TIME_LIMIT,
                prefix="mateN",
            )

        # --------------------------------------------------------------
        # 2) Train RL model on this cycle's data
        # --------------------------------------------------------------
        print("[RL] Training self-play model on this cycle's data...")

        # NOTE: This assumes train_selfplay_model looks at ALL .npz in `data_dir`.
        # If its signature is different, we can adjust once we see it.
        train_selfplay_model(
            data_dir=cycle_dir,
            init_model_path=start_weights_path,
            rl_model_path=RL_MODEL_PATH,
            device=device,
        )

        # New starting weights for the next cycle
        start_weights_path = RL_MODEL_PATH
        print(f"[RL] Finished cycle {cycle}. Updated RL weights: {RL_MODEL_PATH}")

    print("[RL] All cycles completed.")


if __name__ == "__main__":
    main()
