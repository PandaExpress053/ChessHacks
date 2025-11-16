# training/selfplay/selfplay_generator.py

import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import chess
import torch  # used by the engine, harmless here

# ======================================================================
# BASE SELF-PLAY TUNING CONSTANTS
# ======================================================================

# How often to deliberately play a bad move (to learn to handle blunders)
BLUNDER_PLAY_PROB = 0.15         # slightly less chaos than 0.25
BLUNDER_CP_DROP_THRESHOLD = 150  # centipawns worse than best to qualify as a blunder

# Fraction of games where Stockfish helps “steer” winning positions
SF_GUIDE_GAME_PROB = 0.30        # 30% of games are SF-guided (before schedule scaling)

# Threshold at which SF considers the side to be clearly winning
WINNING_CP_THRESHOLD = 300.0     # in centipawns (e.g. +300 or -300)

# When exploring, only sample moves that aren't too much worse than SF-best
SAFE_CP_DROP = 200.0             # in centipawns; moves > 2 pawns worse are avoided in normal exploration


from .config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    NUM_GAMES,
    SF_TIME_LIMIT,
    POLICY_TEMP_CP,
    CP_SCALE,
    SELFPLAY_ENGINE_DEPTH,
)

from .stockfishWRAP import StockfishEvaluator

# Engine / model side
from src.main import (
    PolicyOnlyEngine,
    board_to_planes,
    move_to_index,
    POLICY_DIM,
)


# ======================================================================
# UTILS
# ======================================================================

def softmax_from_cp(cp_values: np.ndarray, temp_cp: float) -> np.ndarray:
    """
    Convert centipawn evals to a probability distribution using a temperature
    measured in centipawns.
    """
    if cp_values.size == 0:
        return cp_values

    scaled = cp_values.astype(np.float32) / float(temp_cp)
    max_s = scaled.max()
    exps = np.exp(scaled - max_s)
    Z = exps.sum()
    if Z <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / Z


def _interesting_weight(
    board: chess.Board,
    best_sf_move: chess.Move,
    cp_before: float,
    cp_after_best: float,
) -> int:
    """
    Reward shaping via sample weighting:
      - midgame / endgame emphasized
      - GOOD checks / GOOD eval swings emphasized
      - BAD checks / big eval drops de-emphasized

    Returns an integer >= 1 used to duplicate the sample.
    """

    weight = 1

    # Game phase: emphasize midgame & endgame
    fullmove = board.fullmove_number
    if fullmove >= 12:
        weight += 1        # midgame
    if fullmove >= 25:
        weight += 1        # deeper endgame

    # Check / mate bonuses / penalties
    board.push(best_sf_move)
    is_check = board.is_check()
    board.pop()

    cp_gain = cp_after_best - cp_before  # SF eval improvement for best_sf_move

    # Good checking move (check + eval improves nicely)
    if is_check and cp_gain > 50:
        weight += 1

    # Bad check: gives check but *loses* eval → de-emphasize
    if is_check and cp_gain < -50:
        weight -= 1

    # Big eval swings
    if cp_gain > 150:
        weight += 1  # strong improvement
    if cp_gain < -150:
        weight -= 2  # harsher penalty for bad sacs / big blunders

    # Near-mate or mate (our wrapper uses ±100000 for mates)
    if abs(cp_after_best) >= 90_000:
        weight += 3   # upgraded from +2 → extra emphasis on mating patterns

    # Cap the weight to avoid insane duplication
    return max(1, min(weight, 5))


def _would_cause_threefold(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if playing `move` would allow a threefold repetition claim.
    (Currently unused, but kept for potential future use.)
    """
    board.push(move)
    can = board.can_claim_threefold_repetition()
    board.pop()
    return can


# ======================================================================
# ONE SELF-PLAY GAME
# ======================================================================

def play_one_selfplay_game(
    engine: PolicyOnlyEngine,
    sf: StockfishEvaluator,
    top_k: int,
    max_moves: int,
    # Effective per-call knobs, so rl_loop can decay them over cycles:
    blunder_prob: float,
    sf_guide_prob: float,
    conversion_cp_threshold: float,
) -> Tuple[
    list[np.ndarray],
    list[int],
    list[np.ndarray],
    list[float],
    list[float],
    list[float],
    list[int],
]:
    """
    Play a single self-play game with your NN+alpha-beta engine as the actor
    and Stockfish as the critic on top-K root moves.

    Behavior:
      - All games are kept (no checkmate-only discard), but
        checkmating / attacking / improving positions are weighted more
        via _interesting_weight().
      - Some games are "Stockfish-guided": when SF thinks a side is
        clearly winning, we more aggressively follow the SF-best move.
      - Once eval is heavily in your favour (|cp_before| >= conversion_cp_threshold),
        we enter a "conversion mode" and let Stockfish convert to checkmate
        for the rest of the game.
      - We still inject blunders and NN exploration for robustness,
        but normal exploration only samples from SF-"safe" moves.

    Returns lists over plies:
      - planes_list:      list of (18, 8, 8) float32
      - policy_best_idx:  list of ints (Stockfish-best among top-K)
      - policy_topk_idx:  list of np.array(K,) of ints
      - cp_before_list:   list of floats
      - cp_after_best:    list of floats
      - result_list:      list of floats (game outcome from stm POV)
      - weight_list:      list of ints (sample duplication weight)
    """

    board = chess.Board()

    planes_list: list[np.ndarray] = []
    policy_best_idx_list: list[int] = []
    policy_topk_idx_list: list[np.ndarray] = []
    cp_before_list: list[float] = []
    cp_after_best_list: list[float] = []
    weight_list: list[int] = []

    move_count = 0

    # for computing result from White POV:
    board_history: list[chess.Board] = []

    # Decide if this game will be SF-guided (per-call prob)
    sf_guided = (random.random() < sf_guide_prob)

    # Conversion mode: once eval is very large, let SF autopilot to mate
    conversion_mode = False

    while not board.is_game_over() and move_count < max_moves:
        board_history.append(board.copy(stack=False))

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # If already in conversion mode: just let Stockfish best drive the game
        if conversion_mode:
            cp_after_conv = []
            for mv in legal_moves:
                board.push(mv)
                cp_after_m = sf.evaluate_cp(board)
                board.pop()
                cp_after_conv.append(cp_after_m)
            cp_after_conv = np.array(cp_after_conv, dtype=np.float32)

            best_idx = int(cp_after_conv.argmax())
            best_sf_move = legal_moves[best_idx]
            best_sf_cp = float(cp_after_conv[best_idx])

            planes = board_to_planes(board)
            planes_list.append(planes)
            policy_best_idx_list.append(move_to_index(best_sf_move))
            policy_topk_idx_list.append(
                np.array([move_to_index(mv) for mv in legal_moves], dtype=np.int64)
            )
            cp_before = sf.evaluate_cp(board)
            cp_before_list.append(float(cp_before))
            cp_after_best_list.append(best_sf_cp)

            w = _interesting_weight(board, best_sf_move, cp_before, best_sf_cp)
            weight_list.append(w)

            board.push(best_sf_move)
            move_count += 1
            continue

        # Otherwise, normal NN+SF logic:

        # Ask your engine for move + root probs (search-based)
        best_move, probs_dict = engine.search_best_move(
            board,
            max_depth=SELFPLAY_ENGINE_DEPTH,
        )
        if best_move is None:
            # Safety fallback: random legal move
            best_move = random.choice(legal_moves)

        # Ensure probs over legal moves
        probs = np.array(
            [probs_dict.get(mv, 0.0) for mv in legal_moves],
            dtype=np.float32,
        )
        Z = probs.sum()
        if Z <= 0:
            probs[:] = 1.0 / len(legal_moves)
        else:
            probs /= Z

        # Choose top-K moves by engine's probability (actor's proposal set)
        sorted_indices = np.argsort(-probs)
        k = min(top_k, len(legal_moves))
        topk_indices = sorted_indices[:k]
        topk_moves = [legal_moves[i] for i in topk_indices]

        # Stockfish evaluates current position and each top-K move
        cp_before = sf.evaluate_cp(board)

        cp_after = []
        for mv in topk_moves:
            board.push(mv)
            cp_after_m = sf.evaluate_cp(board)
            board.pop()
            cp_after.append(cp_after_m)
        cp_after = np.array(cp_after, dtype=np.float32)

        # choose SF-best among top-K
        best_sf_idx_local = int(cp_after.argmax())
        best_sf_move = topk_moves[best_sf_idx_local]
        best_sf_cp = float(cp_after[best_sf_idx_local])

        # build sparse policy target info (indices into global POLICY_DIM space)
        policy_topk_idx = np.array(
            [move_to_index(mv) for mv in topk_moves],
            dtype=np.int64,
        )

        # (Optional) soft distribution over top-K from cp_after (not stored yet)
        _policy_topk_prob = softmax_from_cp(cp_after, temp_cp=POLICY_TEMP_CP)

        # record features & labels (except game result)
        planes = board_to_planes(board)
        planes_list.append(planes)
        policy_best_idx_list.append(move_to_index(best_sf_move))
        policy_topk_idx_list.append(policy_topk_idx)
        cp_before_list.append(float(cp_before))
        cp_after_best_list.append(best_sf_cp)

        # reward-shaping weight for this sample
        w = _interesting_weight(board, best_sf_move, cp_before, best_sf_cp)
        weight_list.append(w)

        # ------------------------------------------------------------------
        # Check for CONVERSION MODE trigger
        # ------------------------------------------------------------------
        if abs(cp_before) >= conversion_cp_threshold:
            conversion_mode = True

        if abs(best_sf_cp) >= 90_000:
            conversion_mode = True

        # If conversion_mode just got activated this turn, execute SF best now
        if conversion_mode:
            board.push(best_sf_move)
            move_count += 1
            continue

        # ------------------------------------------------------------------
        # Prepare SF-"safe" subset of moves for exploration
        # ------------------------------------------------------------------
        best_cp = best_sf_cp
        cp_drop = best_cp - cp_after  # >0 means worse than best

        safe_indices = np.where(cp_drop <= SAFE_CP_DROP)[0]
        if safe_indices.size == 0:
            safe_indices = np.arange(len(topk_moves))

        safe_topk_moves = [topk_moves[i] for i in safe_indices]

        # ------------------------------------------------------------------
        # Decide which move to actually PLAY in self-play
        # ------------------------------------------------------------------

        r = random.random()
        chosen: chess.Move

        # 1) Blunder mode: inject clearly bad moves with some probability
        if r < blunder_prob and len(topk_moves) > 1:
            cp_drop_for_blunders = best_cp - cp_after  # positive if worse
            blunder_indices = np.where(
                cp_drop_for_blunders >= BLUNDER_CP_DROP_THRESHOLD
            )[0]

            if blunder_indices.size == 0:
                worst_idx_local = int(cp_after.argmin())
                chosen = topk_moves[worst_idx_local]
            else:
                idx_local = int(np.random.choice(blunder_indices))
                chosen = topk_moves[idx_local]

        else:
            # 2) Normal / SF-guided mode
            if board.fullmove_number <= 12:
                explore_prob = 0.4  # 60% best, 40% explore
            else:
                explore_prob = 0.2  # 80% best, 20% explore

            clearly_winning = abs(cp_before) >= WINNING_CP_THRESHOLD

            if sf_guided and clearly_winning:
                if random.random() < 0.85:
                    chosen = best_sf_move
                else:
                    if random.random() < (1.0 - explore_prob):
                        chosen = best_move
                    else:
                        chosen = random.choice(safe_topk_moves)
            else:
                if random.random() < (1.0 - explore_prob):
                    chosen = best_move
                else:
                    chosen = random.choice(safe_topk_moves)

        board.push(chosen)
        move_count += 1

    # ------------------------------------------------------------------
    # GAME RESULT
    # ------------------------------------------------------------------

    if not board.is_game_over():
        print("[SELFPLAY] Game ended by move limit; treating as draw.")
        result_str = "1/2-1/2"
    else:
        result_str = board.result()  # e.g. "1-0", "0-1", "1/2-1/2"

    if result_str == "1-0":
        z_white = 1.0
    elif result_str == "0-1":
        z_white = -1.0
    else:
        z_white = 0.0

    # For each stored position, result from side-to-move POV
    result_list: list[float] = []
    for b in board_history[: len(planes_list)]:  # ensure same length
        stm_factor = 1.0 if b.turn == chess.WHITE else -1.0
        result_list.append(z_white * stm_factor)

    return (
        planes_list,
        policy_best_idx_list,
        policy_topk_idx_list,
        cp_before_list,
        cp_after_best_list,
        result_list,
        weight_list,
    )


# ======================================================================
# NPZ GENERATION (WITH OPTIONAL SCHEDULE OVERRIDES)
# ======================================================================

def generate_selfplay_npz(
    out_path: Path,
    num_games: int = NUM_GAMES,
    top_k: int = TOP_K_MOVES,
    max_moves: int = MAX_MOVES_PER_GAME,
    sf_time_limit: float = SF_TIME_LIMIT,
    # Optional per-call overrides so rl_loop can decay them over cycles:
    blunder_prob: Optional[float] = None,
    sf_guide_prob: Optional[float] = None,
    conversion_cp_threshold: Optional[float] = None,
):
    """
    Generate a self-play + Stockfish-labeled dataset and save it to `out_path`.
    The network used is loaded via PolicyOnlyEngine / MODEL_PATH.

    All games are included; checkmating / attacking / improving positions are
    emphasized via sample weighting.

    If blunder_prob / sf_guide_prob / conversion_cp_threshold are provided,
    they override the base constants for this generation run; this allows
    rl_loop.py to gradually reduce SF help and blunders over RL cycles.
    """
    eff_blunder_prob = BLUNDER_PLAY_PROB if blunder_prob is None else blunder_prob
    eff_sf_guide_prob = SF_GUIDE_GAME_PROB if sf_guide_prob is None else sf_guide_prob
    eff_conversion_cp_threshold = (
        800.0 if conversion_cp_threshold is None else conversion_cp_threshold
    )

    print(f"[SELFPLAY] Loading NN engine from {MODEL_PATH}")
    print(
        f"[SELFPLAY] blunder_prob={eff_blunder_prob:.3f}, "
        f"sf_guide_prob={eff_sf_guide_prob:.3f}, "
        f"conversion_cp_threshold={eff_conversion_cp_threshold:.1f} cp"
    )

    engine = PolicyOnlyEngine()
    engine.reset()
    sf = StockfishEvaluator(time_limit=sf_time_limit)

    X_all: list[np.ndarray] = []
    policy_best_idx_all: list[int] = []
    policy_topk_idx_all: list[np.ndarray] = []
    cp_before_all: list[float] = []
    cp_after_best_all: list[float] = []
    delta_cp_all: list[float] = []
    result_all: list[float] = []

    try:
        for gi in range(num_games):
            print(f"[SELFPLAY] Game {gi + 1}/{num_games}")
            (
                planes_list,
                policy_best_idx_list,
                policy_topk_idx_list,
                cp_before_list,
                cp_after_best_list,
                result_list,
                weight_list,
            ) = play_one_selfplay_game(
                engine=engine,
                sf=sf,
                top_k=top_k,
                max_moves=max_moves,
                blunder_prob=eff_blunder_prob,
                sf_guide_prob=eff_sf_guide_prob,
                conversion_cp_threshold=eff_conversion_cp_threshold,
            )

            if not planes_list:
                continue

            for i in range(len(planes_list)):
                repeat = max(1, int(weight_list[i]))
                for _ in range(repeat):
                    X_all.append(planes_list[i])
                    policy_best_idx_all.append(policy_best_idx_list[i])
                    policy_topk_idx_all.append(policy_topk_idx_list[i])
                    cp_before_all.append(cp_before_list[i])
                    cp_after_best_all.append(cp_after_best_list[i])
                    delta_cp_all.append(cp_after_best_list[i] - cp_before_list[i])
                    result_all.append(result_list[i])

        if not X_all:
            print("[SELFPLAY] No positions generated; aborting.")
            return

        X_arr = np.stack(X_all).astype(np.float32)
        policy_best_idx_arr = np.array(policy_best_idx_all, dtype=np.int64)

        max_k = max(arr.shape[0] for arr in policy_topk_idx_all)

        policy_topk_idx_padded = []
        for arr in policy_topk_idx_all:
            if arr.shape[0] < max_k:
                pad_len = max_k - arr.shape[0]
                pad = -np.ones(pad_len, dtype=np.int64)
                arr = np.concatenate([arr, pad])
            policy_topk_idx_padded.append(arr)

        policy_topk_idx_arr = np.stack(policy_topk_idx_padded).astype(np.int64)

        cp_before_arr = np.array(cp_before_all, dtype=np.float32)
        cp_after_best_arr = np.array(cp_after_best_all, dtype=np.float32)
        delta_cp_arr = np.array(delta_cp_all, dtype=np.float32)
        result_arr = np.array(result_all, dtype=np.float32)

        print(f"[SELFPLAY] Saving {X_arr.shape[0]} positions to {out_path}")
        np.savez_compressed(
            out_path,
            X=X_arr,
            policy_best_idx=policy_best_idx_arr,
            policy_topk_idx=policy_topk_idx_arr,
            cp_before=cp_before_arr,
            cp_after_best=cp_after_best_arr,
            delta_cp=delta_cp_arr,
            result=result_arr,
        )

    finally:
        sf.close()
        engine.reset()


if __name__ == "__main__":
    out_file = SELFPLAY_OUT_DIR / "selfplay_sf_topk_dataset.npz"
    generate_selfplay_npz(out_file)
