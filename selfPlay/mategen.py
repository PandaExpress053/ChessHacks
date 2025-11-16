#!/usr/bin/env python3
"""
mateN_supervised_gen.py
=======================

Synthetic *mate-in-N* supervised data generator.

Provides:

1) Standalone CLI:
   - Generates one or more .npz files containing ONLY mate-in-N positions
     (for the side to move), Stockfish-labeled.

2) RL helper function:
   - generate_mateN_npz_for_cycle(...)
   - Intended to be called from rl_loop.py so that each RL cycle can also
     get a chunk of supervised mate-in-N data in the same directory as
     self-play shards.

Output schema (matches typical supervised SF-labeled dataset):

    X                float32 (N, 18, 8, 8)
    y_policy_best    int64   (N,)
    cp_before        float32 (N,)
    cp_after_best    float32 (N,)
    delta_cp         float32 (N,)
    game_result      float32 (N,)

- All values are from the POV of the side to move at the sampled position.
- Only positions where Stockfish reports mate-in-N (for side to move or
  against it) with |N| <= max_mate_distance are kept.

IMPORTANT:
    You MUST hook board_to_planes() to your existing feature extractor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Optional

import numpy as np
import chess
import chess.engine


# ======================================================================
# CONFIG CONSTANTS (can be tweaked)
# ======================================================================

NUM_PLANES = 18
NUM_PROMOS = 5  # [None, Q, R, B, N]
MATE_SCORE = 100_000

PROMO_MAP = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

# Synthetic endgame-ish piece sets. You can expand/tune these if you like.
ENDGAME_SETS = [
    # Simple mates
    {chess.WHITE: [chess.KING, chess.QUEEN], chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING, chess.ROOK],  chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING],              chess.BLACK: [chess.KING, chess.QUEEN]},
    # Pawn endings
    {chess.WHITE: [chess.KING, chess.PAWN],                          chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING, chess.PAWN, chess.PAWN],              chess.BLACK: [chess.KING, chess.PAWN]},
    {chess.WHITE: [chess.KING, chess.PAWN, chess.PAWN, chess.PAWN],  chess.BLACK: [chess.KING, chess.PAWN]},
]


# ======================================================================
# board_to_planes IMPORT – YOU MUST ADAPT THIS
# ======================================================================

# ⚠️ IMPORTANT:
# Change this import to wherever your real board_to_planes lives.
# Example possibilities (commented so it doesn't break if path is different):
#
# from training.utils.features import board_to_planes
# from src.features import board_to_planes
#
# For now, we provide a stub that raises if you forget to replace it.

def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Replace this with your actual implementation used in supervised training.

    It must:
        - Take a chess.Board
        - Return a numpy array of shape (18, 8, 8), dtype float32
    """
    raise NotImplementedError(
        "mateN_supervised_gen.py: hook board_to_planes() to your real feature extractor."
    )


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def encode_policy_index(move: chess.Move) -> int:
    """
    Encode a move into a policy index compatible with your existing scheme:

        idx = from_square * 64 * NUM_PROMOS + to_square * NUM_PROMOS + promo_idx
    """
    promo_idx = PROMO_MAP.get(move.promotion, 0)
    return move.from_square * 64 * NUM_PROMOS + move.to_square * NUM_PROMOS + promo_idx


def random_endgame_position() -> chess.Board:
    """
    Generate a random, valid endgame-ish position.

    We sample from ENDGAME_SETS and place kings/pieces onto random squares,
    rejecting invalid positions (e.g., illegal king placements).
    """
    board = chess.Board(None)
    occupied = set()
    piece_set = random.choice(ENDGAME_SETS)

    def place_piece(pt: chess.PieceType, color: chess.Color):
        nonlocal board, occupied
        while True:
            sq = random.randrange(64)
            if sq in occupied:
                continue
            # No pawns on first/last ranks
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, chess.Piece(pt, color))
            occupied.add(sq)
            return

    # Place kings first
    place_piece(chess.KING, chess.WHITE)
    place_piece(chess.KING, chess.BLACK)

    # Place remaining pieces
    for color in (chess.WHITE, chess.BLACK):
        for pt in piece_set[color]:
            if pt == chess.KING:
                continue
            place_piece(pt, color)

    board.turn = random.choice([chess.WHITE, chess.BLACK])

    if not board.is_valid():
        # Retry on invalid position
        return random_endgame_position()

    return board


# ======================================================================
# CORE GENERATION LOGIC
# ======================================================================

def _generate_mateN_positions(
    engine: chess.engine.SimpleEngine,
    num_positions: int,
    max_mate_distance: int,
    time_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core loop that collects num_positions *mate-in-N* positions.

    Returns:
        X, y_policy_best, cp_before, cp_after_best, delta_cp, game_result
    """
    X_list = []
    pol_list = []
    cp_before_list = []
    cp_after_list = []
    delta_list = []
    result_list = []

    while len(X_list) < num_positions:
        board = random_endgame_position()

        # Analyse current position
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info["score"].pov(board.turn)

        # Only keep mate-in-N for side to move (positive = STM mates; negative = STM gets mated)
        if not score.is_mate():
            continue

        mate_dist = score.mate()
        if mate_dist is None or mate_dist == 0 or abs(mate_dist) > max_mate_distance:
            continue

        pv = info.get("pv")
        if not pv:
            continue
        best_move = pv[0]

        # Evaluate before/after best move in centipawns
        cp_before = score.score(mate_score=MATE_SCORE)

        b2 = board.copy()
        b2.push(best_move)
        info2 = engine.analyse(b2, chess.engine.Limit(time=time_limit))
        score2 = info2["score"].pov(board.turn)
        cp_after = score2.score(mate_score=MATE_SCORE)
        delta_cp = cp_after - cp_before

        # Game result from POV of STM:
        #   mate_dist > 0  => STM is delivering mate -> +1
        #   mate_dist < 0  => STM is getting mated   -> -1
        game_result = 1.0 if mate_dist > 0 else -1.0

        planes = board_to_planes(board).astype(np.float32)
        policy_idx = encode_policy_index(best_move)

        X_list.append(planes)
        pol_list.append(policy_idx)
        cp_before_list.append(cp_before)
        cp_after_list.append(cp_after)
        delta_list.append(delta_cp)
        result_list.append(game_result)

        if len(X_list) % 100 == 0:
            print(
                f"[mateN] Collected {len(X_list)}/{num_positions} positions...",
                end="\r",
                flush=True,
            )

    print()  # newline after progress

    return (
        np.array(X_list, dtype=np.float32),
        np.array(pol_list, dtype=np.int64),
        np.array(cp_before_list, dtype=np.float32),
        np.array(cp_after_list, dtype=np.float32),
        np.array(delta_list, dtype=np.float32),
        np.array(result_list, dtype=np.float32),
    )


# ======================================================================
# RL HELPER – to be called from rl_loop.py
# ======================================================================

def generate_mateN_npz_for_cycle(
    cycle_idx: int,
    out_dir: Path,
    engine_path: str,
    num_positions: int = 2000,
    max_mate_distance: int = 8,
    time_limit: float = 0.05,
    prefix: str = "mateN",
) -> Path:
    """
    Generate a single mate-in-N supervised NPZ shard for a given RL cycle.

    Parameters
    ----------
    cycle_idx : int
        RL cycle index (for naming the file).
    out_dir : Path
        Directory where the shard will be written. Typically the same
        directory as your self-play shards for this RL cycle.
    engine_path : str
        Path to Stockfish binary.
    num_positions : int
        Number of mate-in-N positions to collect.
    max_mate_distance : int
        Only keep positions where |mate_distance| <= this.
    time_limit : float
        Time in seconds for each Stockfish analyse() call.
    prefix : str
        Filename prefix (default "mateN").

    Returns
    -------
    Path
        The path to the written .npz file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_path = out_dir / f"cycle_{cycle_idx:03d}_{prefix}_supervised.npz"

    print(f"[mateN] Using Stockfish at: {engine_path}")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        (
            X,
            y_policy_best,
            cp_before,
            cp_after_best,
            delta_cp,
            game_result,
        ) = _generate_mateN_positions(
            engine=engine,
            num_positions=num_positions,
            max_mate_distance=max_mate_distance,
            time_limit=time_limit,
        )

        np.savez_compressed(
            shard_path,
            X=X,
            y_policy_best=y_policy_best,
            cp_before=cp_before,
            cp_after_best=cp_after_best,
            delta_cp=delta_cp,
            game_result=game_result,
        )
        print(f"[mateN] Wrote {shard_path}")
        return shard_path

    finally:
        engine.quit()
        print("[mateN] Engine closed.")


# ======================================================================
# STANDALONE CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mate-in-N supervised NPZ data."
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write NPZ shards into.",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=2000,
        help="Number of mate-in-N positions per shard (default: 2000).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to generate (default: 1).",
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=0,
        help="Cycle index to start from when naming shards (default: 0).",
    )
    parser.add_argument(
        "--max-mate-distance",
        type=int,
        default=8,
        help="Keep only positions with |mate_distance| <= this (default: 8).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=0.05,
        help="Stockfish analyse() time limit in seconds (default: 0.05).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mateN",
        help="Filename prefix (default: 'mateN').",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One engine instance reused across shards for speed
    print(f"[mateN CLI] Using Stockfish at: {args.engine}")
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)

    try:
        for i in range(args.num_shards):
            cycle_idx = args.start_cycle + i
            shard_path = out_dir / f"cycle_{cycle_idx:03d}_{args.prefix}_supervised.npz"

            print(
                f"[mateN CLI] Generating shard {i+1}/{args.num_shards} "
                f"(cycle_idx={cycle_idx}) -> {shard_path}"
            )

            (
                X,
                y_policy_best,
                cp_before,
                cp_after_best,
                delta_cp,
                game_result,
            ) = _generate_mateN_positions(
                engine=engine,
                num_positions=args.num_positions,
                max_mate_distance=args.max_mate_distance,
                time_limit=args.time_limit,
            )

            np.savez_compressed(
                shard_path,
                X=X,
                y_policy_best=y_policy_best,
                cp_before=cp_before,
                cp_after_best=cp_after_best,
                delta_cp=delta_cp,
                game_result=game_result,
            )
            print(f"[mateN CLI] Wrote {shard_path}")

    finally:
        engine.quit()
        print("[mateN CLI] Engine closed.")


if __name__ == "__main__":
    main_cli()
