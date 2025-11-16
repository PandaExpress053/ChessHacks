#!/usr/bin/env python3
"""
SYNTHETIC MATE-IN-N DATA GENERATOR (SUPERVISED)
===============================================

Produces NPZ shards with ONLY mate-in-N positions (for side to move),
all labels coming from Stockfish.

Output keys (same as your supervised SF-labeled data):

    X                float32 (N, 18, 8, 8)
    y_policy_best    int64   (N,)
    cp_before        float32 (N,)
    cp_after_best    float32 (N,)
    delta_cp         float32 (N,)
    game_result      float32 (N,)

- game_result is from POV of side to move at the sampled position.
- Only positions where SF reports mate for side to move (mate-in-N) are kept.
"""

import chess
import chess.engine
import numpy as np
import random
from pathlib import Path

# =============================================================
# CONFIG
# =============================================================

# 👉 Change this to your actual Stockfish binary path
ENGINE_PATH = r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\src\stockfish-windows-x86-64-avx2.exe"

# 👉 Where to save the .npz shards
OUT_DIR = Path(
    r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\processed"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 2000          # positions per shard
NUM_SHARDS = 5             # number of shards to generate
TIME_LIMIT = 0.05          # SF time per query
MATE_SCORE = 100000        # centipawn equivalent for mate
MAX_MATE_DISTANCE = 8      # keep only mate-in-N where |N| <= this

NUM_PLANES = 18
NUM_PROMOS = 5  # [None, Q, R, B, N]

# 👉 Replace this import with whatever you use in your pipeline
# e.g. from training.utils.features import board_to_planes
PIECE_PLANES = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Same encoding as training: (18, 8, 8).
    """
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1.0

    # side to move
    if board.turn == chess.WHITE:
        P[12, :, :] = 1.0
    else:
        P[12, :, :] = 0.0

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        P[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        P[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        P[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        P[16, :, :] = 1.0

    # en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1.0

    return P

PROMO_MAP = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}


# =============================================================
# Helpers
# =============================================================

def encode_policy_index(move: chess.Move) -> int:
    """
    Policy index encoding consistent with your other data:

        idx = from_sq * 64 * NUM_PROMOS + to_sq * NUM_PROMOS + promo_idx
    """
    promo_idx = PROMO_MAP.get(move.promotion, 0)
    return move.from_square * 64 * NUM_PROMOS + move.to_square * NUM_PROMOS + promo_idx


ENDGAME_SETS = [
    # Simple mates / basic endgames – can tweak these
    {chess.WHITE: [chess.KING, chess.QUEEN], chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING, chess.ROOK],  chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING], chess.BLACK: [chess.KING, chess.QUEEN]},
    {chess.WHITE: [chess.KING, chess.PAWN], chess.BLACK: [chess.KING]},
    {chess.WHITE: [chess.KING, chess.PAWN, chess.PAWN], chess.BLACK: [chess.KING, chess.PAWN]},
]


def random_endgame_position() -> chess.Board:
    """Generate a random, valid endgame-ish position."""
    board = chess.Board(None)
    occupied = set()
    piece_set = random.choice(ENDGAME_SETS)

    def place(pt: chess.PieceType, color: chess.Color):
        while True:
            sq = random.randrange(64)
            if sq in occupied:
                continue
            # No pawns on first/last rank
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, chess.Piece(pt, color))
            occupied.add(sq)
            break

    # Kings first
    place(chess.KING, chess.WHITE)
    place(chess.KING, chess.BLACK)

    # Remaining pieces
    for color in (chess.WHITE, chess.BLACK):
        for pt in piece_set[color]:
            if pt == chess.KING:
                continue
            place(pt, color)

    board.turn = random.choice([chess.WHITE, chess.BLACK])

    # If invalid (e.g. illegal king in check), retry
    if not board.is_valid():
        return random_endgame_position()

    return board


# =============================================================
# Main
# =============================================================

def main():
    print(f"[MATE N GEN] Using Stockfish at: {ENGINE_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    try:
        for shard_idx in range(1, NUM_SHARDS + 1):
            X_list = []
            pol_list = []
            cp_before_list = []
            cp_after_list = []
            delta_list = []
            result_list = []

            print(f"[MATE N GEN] Generating shard {shard_idx}/{NUM_SHARDS}...")

            while len(X_list) < SHARD_SIZE:
                board = random_endgame_position()

                # Ask Stockfish about current position
                info = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT))
                score = info["score"].pov(board.turn)

                # We only want mate-in-N for side to move
                if not score.is_mate():
                    continue

                mate_dist = score.mate()  # POV: positive => side to move mates opp, negative => gets mated
                if mate_dist is None:
                    continue
                if mate_dist == 0 or abs(mate_dist) > MAX_MATE_DISTANCE:
                    continue

                # Best move from PV
                pv = info.get("pv")
                if not pv:
                    continue
                best_move = pv[0]

                # Eval before / after best move (centipawns)
                cp_before = score.score(mate_score=MATE_SCORE)

                b2 = board.copy()
                b2.push(best_move)
                info2 = engine.analyse(b2, chess.engine.Limit(time=TIME_LIMIT))
                score2 = info2["score"].pov(board.turn)
                cp_after = score2.score(mate_score=MATE_SCORE)
                delta_cp = cp_after - cp_before

                # Game result from side-to-move POV:
                #   mate_dist > 0 => side to move mates => +1
                #   mate_dist < 0 => side to move is mated => -1
                game_result = 1.0 if mate_dist > 0 else -1.0

                # Features
                planes = board_to_planes(board).astype(np.float32)
                policy_idx = encode_policy_index(best_move)

                X_list.append(planes)
                pol_list.append(policy_idx)
                cp_before_list.append(cp_before)
                cp_after_list.append(cp_after)
                delta_list.append(delta_cp)
                result_list.append(game_result)

                if len(X_list) % 100 == 0:
                    print(f"  collected {len(X_list)}/{SHARD_SIZE} positions...", end="\r")

            shard_path = OUT_DIR / f"mateN_shard_{shard_idx:05d}.npz"
            np.savez_compressed(
                shard_path,
                X=np.array(X_list, dtype=np.float32),
                y_policy_best=np.array(pol_list, dtype=np.int64),
                cp_before=np.array(cp_before_list, dtype=np.float32),
                cp_after_best=np.array(cp_after_list, dtype=np.float32),
                delta_cp=np.array(delta_list, dtype=np.float32),
                game_result=np.array(result_list, dtype=np.float32),
            )
            print(f"\n[MATE N GEN] Wrote {shard_path}")

    finally:
        engine.quit()
        print("[MATE N GEN] Done.")


if __name__ == "__main__":
    main()
