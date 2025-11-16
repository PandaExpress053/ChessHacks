# training/selfplay/rl_loop.py

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.pgn
import torch

from .config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,          # supervised best.pt
    NUM_GAMES,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    SF_TIME_LIMIT,
)

from .selfplayGEN import (
    generate_selfplay_npz,
    BLUNDER_PLAY_PROB,
    SF_GUIDE_GAME_PROB,
)

from .trainselfplay import train_selfplay_model
from src.main import PolicyOnlyEngine

# Some configs might not define SELFPLAY_ENGINE_DEPTH;
# fall back to a sensible default.
try:
    from .config import SELFPLAY_ENGINE_DEPTH as _DEPTH
    SELFPLAY_ENGINE_DEPTH = _DEPTH
except Exception:
    SELFPLAY_ENGINE_DEPTH = 4  # safe default

SELFPLAY_OUT_DIR = Path(SELFPLAY_OUT_DIR)
SUPER_MODEL_PATH = Path(MODEL_PATH)

# RL weights live alongside best.pt
RL_MODEL_PATH = SUPER_MODEL_PATH.with_name("best_selfplay.pt")


# ======================================================================
# ENGINE HELPERS (USE RL WEIGHTS IF AVAILABLE)
# ======================================================================

def _build_rl_engine() -> PolicyOnlyEngine:
    """
    Build a PolicyOnlyEngine and, if RL weights exist, load them on top.
    This way:
      - src/main.py and submission still use best.pt normally.
      - self-play + visualization use best_selfplay.pt if it exists.
    """
    engine = PolicyOnlyEngine()  # loads SUPER_MODEL_PATH inside

    if RL_MODEL_PATH.is_file():
        try:
            state = torch.load(RL_MODEL_PATH, map_location="cpu")
            if isinstance(state, dict) and "policy_head.weight" not in state:
                state = state.get("model", state)
            engine.model.load_state_dict(state, strict=False)
            print(f"[RL_ENGINE] Loaded RL weights from {RL_MODEL_PATH}")
        except Exception as e:
            print(f"[RL_ENGINE] Failed to load RL weights ({e}), using base MODEL_PATH.")

    return engine


# ======================================================================
# RL SCHEDULE: DECAY SF HELP + BLUNDERS OVER CYCLES
# ======================================================================

def compute_schedule_for_cycle(
    cycle_idx: int,
    total_cycles: int,
) -> tuple[float, float, float]:
    """
    Returns (blunder_prob, sf_guide_prob, conversion_cp_threshold)
    for this RL cycle.

    Goals:
      - Early cycles:
          * small blunder rate (to learn to punish, not live in chaos)
          * some SF-guided games (but at least 33% pure model-vs-model)
          * low conversion threshold so SF converts won positions to mate
      - Later cycles:
          * blunders -> almost 0
          * sf_guide_prob -> 0 (100% model-vs-model)
          * conversion threshold high (almost no SF autopilot)
    """
    if total_cycles <= 1:
        progress = 0.0
    else:
        progress = cycle_idx / float(total_cycles - 1)  # 0 → 1

    # Help factor: 1 at first cycle, 0 at last cycle
    help_factor = 1.0 - progress

    # Make blunders relatively rare and decay quickly:
    # base BLUNDER_PLAY_PROB is 0.15 → effective at most ~0.075 early
    blunder_prob = BLUNDER_PLAY_PROB * 0.5 * (help_factor ** 1.5)
    blunder_prob = min(blunder_prob, 0.10)

    # SF-guided games: bounded so guided_games <= 0.67 ⇒ >= 33% pure model-vs-model
    sf_guide_prob = 0.67 * help_factor  # from ~0.67 → 0.0 across cycles

    # Conversion threshold:
    #   early: ~400 cp → SF converts many winning positions to mate
    #   late: grows, so SF rarely takes over
    base_conv_cp = 400.0
    min_help_for_conv = 0.25
    denom = max(help_factor, min_help_for_conv)
    conversion_cp_threshold = base_conv_cp / denom  # 400 → 1600+ as help shrinks

    return blunder_prob, sf_guide_prob, conversion_cp_threshold


# ======================================================================
# ONE RL CYCLE: GENERATE + TRAIN
# ======================================================================

def run_one_cycle(
    cycle_idx: int,
    total_cycles: int,
    num_games: int,
    top_k: int = TOP_K_MOVES,
    max_moves: int = MAX_MOVES_PER_GAME,
    sf_time_limit: float = SF_TIME_LIMIT,
):
    """
    One RL iteration:
      1) Generate self-play data with current RL weights
      2) Fine-tune RL model on that data, saving into RL_MODEL_PATH
    """
    dataset_path = SELFPLAY_OUT_DIR / "selfplay_sf_topk_dataset.npz"

    blunder_prob, sf_guide_prob, conv_cp = compute_schedule_for_cycle(
        cycle_idx=cycle_idx,
        total_cycles=total_cycles,
    )

    print("=" * 80)
    print(
        f"[CYCLE {cycle_idx}] SCHEDULE:"
        f" blunder_prob={blunder_prob:.3f},"
        f" sf_guide_prob={sf_guide_prob:.3f},"
        f" conversion_cp_threshold={conv_cp:.1f} cp"
    )

    # Self-play generation (cap SF time a bit for speed)
    effective_sf_time = min(sf_time_limit, 0.03)

    print(f"[CYCLE {cycle_idx}] Generating self-play dataset → {dataset_path}")
    generate_selfplay_npz(
        out_path=dataset_path,
        num_games=num_games,
        top_k=top_k,
        max_moves=max_moves,
        sf_time_limit=effective_sf_time,
        blunder_prob=blunder_prob,
        sf_guide_prob=sf_guide_prob,
        conversion_cp_threshold=conv_cp,
    )

    print(f"[CYCLE {cycle_idx}] Training on self-play data (fine-tune RL model)...")
    train_selfplay_model(dataset_path, RL_MODEL_PATH)
    print(f"[CYCLE {cycle_idx}] Done training, RL weights updated at {RL_MODEL_PATH}")


# ======================================================================
# VISUALIZATION HELPERS
# ======================================================================

def visualize_self_play(max_ply: int = 120):
    """
    Visualize a single self-play game with the CURRENT RL weights.
    Pure RL engine (no SF move selection).
    """
    print("-" * 80)
    print("[VISUALIZE] Playing a self-play game with current RL model...")

    engine = _build_rl_engine()
    engine.reset()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "RL-Engine"
    game.headers["Black"] = "RL-Engine"
    node = game

    ply = 0
    while not board.is_game_over() and ply < max_ply:
        best_move, _ = engine.search_best_move(
            board,
            max_depth=SELFPLAY_ENGINE_DEPTH,
        )
        if best_move is None:
            print("[VISUALIZE] No legal move returned; stopping.")
            break

        board.push(best_move)
        node = node.add_variation(best_move)
        ply += 1

    if board.is_game_over():
        game.headers["Result"] = board.result()
    else:
        game.headers["Result"] = "*"

    print("[VISUALIZE] PGN of self-play game:\n")
    print(game)
    print()


def dump_checkmate_pgn(
    cycle_idx: int,
    max_tries: int = 3,
    max_ply: int = 200,
):
    """
    Try up to max_tries times to generate a self-play game that ends in checkmate
    using the CURRENT RL model, and dump it as a PGN file for inspection.
    """
    engine = _build_rl_engine()

    for attempt in range(1, max_tries + 1):
        print(f"[CYCLE {cycle_idx}] Generating PGN example game (attempt {attempt}/{max_tries})")

        engine.reset()
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["White"] = "RL-Engine"
        game.headers["Black"] = "RL-Engine"

        node = game
        ply = 0

        while not board.is_game_over() and ply < max_ply:
            best_move, _ = engine.search_best_move(
                board,
                max_depth=SELFPLAY_ENGINE_DEPTH,
            )
            if best_move is None:
                break

            board.push(best_move)
            node = node.add_variation(best_move)
            ply += 1

        if board.is_checkmate():
            game.headers["Result"] = board.result()
            pgn_str = str(game)

            pgn_path = SELFPLAY_OUT_DIR / f"cycle_{cycle_idx:03d}_example.pgn"
            with open(pgn_path, "w", encoding="utf-8") as f:
                f.write(pgn_str)

            print(f"[CYCLE {cycle_idx}] Saved checkmate PGN to {pgn_path}\n")
            print(pgn_str)
            print()
            return

        else:
            result_desc = board.result() if board.is_game_over() else "incomplete"
            print(f"[CYCLE {cycle_idx}] Game did not end in checkmate (result={result_desc}).")

    print(f"[CYCLE {cycle_idx}] Failed to generate a checkmate game in {max_tries} attempts.")


# ======================================================================
# MAIN CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RL self-play cycles (generate → train → visualize)."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of (generate → train → visualize) RL cycles to run.",
    )
    parser.add_argument(
        "--games-per-cycle",
        type=int,
        default=NUM_GAMES,
        help="Number of self-play games per generation run.",
    )
    parser.add_argument(
        "--visualize-max-ply",
        type=int,
        default=120,
        help="Max number of plies to play in the visualization game.",
    )

    args = parser.parse_args()

    for c in range(args.cycles):
        run_one_cycle(
            cycle_idx=c,
            total_cycles=args.cycles,
            num_games=args.games_per_cycle,
            top_k=TOP_K_MOVES,
            max_moves=MAX_MOVES_PER_GAME,
            sf_time_limit=SF_TIME_LIMIT,
        )

        visualize_self_play(max_ply=args.visualize_max_ply)
        dump_checkmate_pgn(cycle_idx=c, max_tries=3, max_ply=200)

    print("=" * 80)
    print(
        f"RL loop finished.\n"
        f"  Supervised best.pt : {SUPER_MODEL_PATH}\n"
        f"  RL best_selfplay.pt: {RL_MODEL_PATH}\n"
    )


if __name__ == "__main__":
    main()
