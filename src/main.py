from .utils import chess_manager, GameContext
from chess import Move
import chess
import random
import time
from pathlib import Path
import math

import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

# Relative path: src/main.py → src/model_save/best.pt
MODEL_PATH = Path(r"C:/Users/ethan/Downloads/ChessHacks/e/ChessHacks/src/model_save/best.pt")

NUM_PLANES = 18
NUM_PROMOS = 5   # [None, Q, R, B, N]
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0  # must match training

MATE_SCORE = 100000.0  # used for checkmate scores

# --- SEARCH TUNABLES (speed/strength knobs) ---
MAX_SEARCH_DEPTH = 20            # 1 = eval every legal move once, 2 = light lookahead
ROOT_TOP_K = 25                   # only these many moves get full-depth search (None = all)
# For debugging, disable time limit so we see full search behavior.
# For competition, you can set this back to e.g. 0.18.
TIME_LIMIT_SECONDS = 1       # per-move soft limit; set to None to disable
CP_SOFTMAX_TEMPERATURE = 200.0   # for converting search scores to probs
USE_POLICY_ROOT_ORDERING = True  # use NN policy to order root moves
BONUS_CAPTURE_ORDERING = False   # NOTE: disabled – was causing bad behavior

# --- EVAL / DEBUG OPTIONS -----------------------------------
# If your CP head was trained as "eval from WHITE POV", set this True.
# If it was trained as "side-to-move POV", leave False.
CP_IS_WHITE_POV = False

# Policy-bias and blunder guard at root (all learned signals)
POLICY_SCORE_LAMBDA = 0.1   # λ in score_root = search_value + λ * policy_logit
BLUNDER_MARGIN_CP = 20    # moves worse than (best_value - margin) are discarded

# Print debug info for each move at the root (helpful to see blunders)
DEBUG_SEARCH = False
DEBUG_TOP_K = 10  # how many root moves to show in debug dumps

ENGINE = None  # will be set if NN loads


# ============================================================
# BOARD → PLANES (must match training)
# ============================================================

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


def move_to_index(move: chess.Move) -> int:
    """
    Same policy index scheme as training:
      (from * 64 + to) * 5 + promo_idx
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)  # 0..4
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


# ============================================================
# NN engine with policy + cp eval + NEGAMAX search (patched)
# ============================================================

try:
    from .model import ResNet20_PolicyDelta
except Exception as e:
    ResNet20_PolicyDelta = None
    print(f"[ENGINE] Could not import ResNet20_PolicyDelta from .model: {e}")


class PolicyOnlyEngine:
    """
    NN engine with:
      - ResNet20_PolicyDelta
      - Policy head for move ordering
      - CP head for evaluation
      - Negamax + alpha–beta search to depth MAX_SEARCH_DEPTH
      - Eval & policy caching
      - Optional per-move time limit

    IMPORTANT:
      We treat cp_pred as either WHITE-POV or side-to-move-POV,
      controlled by CP_IS_WHITE_POV.
    """

    def __init__(self):
        if ResNet20_PolicyDelta is None:
            raise RuntimeError("Model class not available")

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ENGINE] Using device: {self.device}")

        self.model = ResNet20_PolicyDelta(
            in_planes=NUM_PLANES,
            policy_dim=POLICY_DIM,
            cp_scale=CP_SCALE,
        ).to(self.device)

        state = torch.load(MODEL_PATH, map_location="cpu")

        # If someone accidentally pointed MODEL_PATH at a rich checkpoint (with "model", "optim", etc.),
        # extract the actual weights.
        if isinstance(state, dict) and "model" in state and "optim" in state:
            print("[ENGINE] Detected training checkpoint at MODEL_PATH; extracting 'model' subkey.")
            state = state["model"]

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Simple transposition/eval caches
        self.eval_cache = {}    # (zkey, side_to_move) -> eval
        self.policy_cache = {}  # zkey -> logits
        self.search_deadline = None

        # Node counter for debugging
        self.nodes = 0
        self.time_hit = False

        print(f"[ENGINE] Loaded model from {MODEL_PATH}")

    # -------------------------------
    # Helpers: Zobrist key
    # -------------------------------

    def _zkey(self, board: chess.Board) -> int:
        try:
            return board.transposition_key()
        except Exception:
            return hash(board.fen())

    # -------------------------------
    # Low-level NN helpers
    # -------------------------------

    def _forward(self, board: chess.Board):
        """
        Single NN forward for a board.
        Returns (logits, cp_raw) where:
          - logits shape: (POLICY_DIM,)
          - cp_raw: scalar eval (raw CP head output)
        """
        planes = board_to_planes(board)  # (18, 8, 8)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # (1,18,8,8)

        with torch.inference_mode():
            # logits, delta_real, delta_pred, cp_real, cp_pred, res_pred
            logits, _, _, _, cp_pred, _ = self.model(x)

        logits = logits[0].detach().cpu().numpy()   # (POLICY_DIM,)
        cp_raw = float(cp_pred[0].item())           # scalar
        return logits, cp_raw

    def _policy_logits(self, board: chess.Board) -> np.ndarray:
        """
        Cached policy logits.
        """
        z = self._zkey(board)
        if z in self.policy_cache:
            return self.policy_cache[z]
        logits, _ = self._forward(board)
        self.policy_cache[z] = logits
        return logits

    def _nn_eval_cp_stm(self, board: chess.Board) -> float:
        """
        Cached NN evaluation "from side-to-move POV" at this node.

        If CP_IS_WHITE_POV is True, we assume cp_raw is from WHITE POV,
        and flip sign when it's Black to move.
        If False, we assume cp_raw is already STM POV.
        """
        z = self._zkey(board)
        key = (z, board.turn)
        if key in self.eval_cache:
            return self.eval_cache[key]

        _, cp_raw = self._forward(board)

        if CP_IS_WHITE_POV:
            cp_stm = cp_raw if board.turn == chess.WHITE else -cp_raw
        else:
            cp_stm = cp_raw

        self.eval_cache[key] = cp_stm
        return cp_stm

    # -------------------------------
    # Evaluation & negamax (patched)
    # -------------------------------

    def _evaluate_terminal_stm(self, board: chess.Board) -> float:
        """
        Evaluation for terminal nodes (game over), from *side to move* POV.
        Positive is good for the side to move.
        """
        outcome = board.outcome()
        if outcome is None:
            return 0.0

        if outcome.winner is None:
            # draw
            return 0.0

        # If winner is the side to move, that's +MATE; otherwise -MATE.
        return MATE_SCORE if outcome.winner == board.turn else -MATE_SCORE

    def _evaluate_stm(self, board: chess.Board) -> float:
        """
        Static evaluation function used at leaves.
        Returns a score from the *side to move* POV.
        """
        if board.is_game_over():
            return self._evaluate_terminal_stm(board)

        cp_stm = self._nn_eval_cp_stm(board)
        return cp_stm

    def _time_exceeded(self) -> bool:
        if TIME_LIMIT_SECONDS is None:
            return False
        if self.search_deadline is None:
            return False
        if time.time() > self.search_deadline:
            self.time_hit = True
            return True
        return False

    def _negamax(self, board: chess.Board, depth: int,
                 alpha: float, beta: float) -> float:
        """
        Negamax + alpha–beta pruning.
        Score is always from *side-to-move's* POV at this node.
        """
        # Count nodes for debugging
        self.nodes += 1

        # Check soft time limit
        if self._time_exceeded():
            # Return a static eval so search can bail out gracefully
            return self._evaluate_stm(board)

        # Leaf or terminal
        if depth == 0 or board.is_game_over():
            return self._evaluate_stm(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # No moves – treat as terminal
            return self._evaluate_stm(board)

        # Very cheap move ordering inside the tree
        if BONUS_CAPTURE_ORDERING:
            def move_inner_key(mv: chess.Move):
                # Captures first
                return board.is_capture(mv)
            legal_moves.sort(key=move_inner_key, reverse=True)

        value = -float("inf")
        for mv in legal_moves:
            board.push(mv)
            score = -self._negamax(
                board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
            )
            board.pop()

            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break  # beta cutoff

        return value

    # -------------------------------
    # Root move selection (with debug, policy bias, blunder guard)
    # -------------------------------

    def search_best_move(self, board: chess.Board, max_depth: int):
        """
        Root search with negamax + alpha–beta.
        Returns (best_move, move_prob_dict).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        root_color = board.turn  # for logging/debug

        # Reset per-move diagnostics
        self.nodes = 0
        self.time_hit = False

        # Start per-move timer
        if TIME_LIMIT_SECONDS is not None:
            self.search_deadline = time.time() + TIME_LIMIT_SECONDS
        else:
            self.search_deadline = None

        # Optional: root move ordering using policy logits
        policy_logits = None
        if USE_POLICY_ROOT_ORDERING:
            try:
                policy_logits = self._policy_logits(board)
            except Exception as e:
                print(f"[ENGINE] Policy logits failed for ordering: {e}")
                policy_logits = None

        def root_order_key(mv: chess.Move):
            score = 0.0
            if policy_logits is not None:
                idx = move_to_index(mv)
                if 0 <= idx < POLICY_DIM:
                    score += policy_logits[idx]
            if BONUS_CAPTURE_ORDERING and board.is_capture(mv):
                score += 10000.0
            return score

        if policy_logits is not None or BONUS_CAPTURE_ORDERING:
            legal_moves.sort(key=root_order_key, reverse=True)

        # If ROOT_TOP_K is set, only search those deeply; others get shallow eval
        if ROOT_TOP_K is not None and ROOT_TOP_K < len(legal_moves):
            primary_moves = legal_moves[:ROOT_TOP_K]
            secondary_moves = legal_moves[ROOT_TOP_K:]
        else:
            primary_moves = legal_moves
            secondary_moves = []

        best_search_move = None
        best_search_value = -float("inf")
        move_values = {}
        move_leaf_evals = {}  # STM static eval after the move (for debug)

        # --- primary moves: full depth search ---
        for mv in primary_moves:
            board.push(mv)

            # Static eval at child (side to move there)
            leaf_eval = self._evaluate_stm(board)

            value = -self._negamax(
                board,
                depth=max_depth - 1,
                alpha=-float("inf"),
                beta=float("inf"),
            )
            board.pop()

            move_values[mv] = value
            move_leaf_evals[mv] = leaf_eval

            if value > best_search_value or best_search_move is None:
                best_search_value = value
                best_search_move = mv

            if self._time_exceeded():
                # No time to do more fancy stuff
                break

        # --- secondary moves: cheap 1-ply eval (if time remains) ---
        if not self._time_exceeded() and secondary_moves:
            for mv in secondary_moves:
                board.push(mv)
                leaf_eval = self._evaluate_stm(board)
                # Depth 1 search = just leaf eval (negamax will hit depth=0)
                value = -self._negamax(
                    board,
                    depth=0,
                    alpha=-float("inf"),
                    beta=float("inf"),
                )
                board.pop()

                move_values[mv] = value
                move_leaf_evals[mv] = leaf_eval

                if value > best_search_value or best_search_move is None:
                    best_search_value = value
                    best_search_move = mv

                if self._time_exceeded():
                    break

        if not move_values:
            return None, {}

        # -------------------------
        # Policy-aware bias + blunder guard
        # -------------------------

        # 1) Blunder guard: discard moves far worse than the best search value
        best_value = best_search_value
        safe_moves = [
            mv for mv, val in move_values.items()
            if val >= best_value - BLUNDER_MARGIN_CP
        ]
        if not safe_moves:
            safe_moves = list(move_values.keys())

        # 2) Combine search value with policy logit (policy-aware scoring)
        combined_scores = {}
        for mv in safe_moves:
            v = move_values[mv]
            policy_term = 0.0
            if policy_logits is not None:
                idx = move_to_index(mv)
                if 0 <= idx < POLICY_DIM:
                    policy_term = POLICY_SCORE_LAMBDA * float(policy_logits[idx])
            combined_scores[mv] = v + policy_term

        best_move = max(combined_scores.items(), key=lambda kv: kv[1])[0]

        # Debug dump: show root move scores
        if DEBUG_SEARCH:
            print("========== ROOT DEBUG ==========")
            print(f"Side to move: {'White' if root_color == chess.WHITE else 'Black'}")
            print(f"FEN: {board.fen()}")
            print(f"CP_IS_WHITE_POV: {CP_IS_WHITE_POV}")
            print(f"Root legal moves: {len(legal_moves)}")
            print(f"Nodes searched: {self.nodes}")
            print(f"Time limit hit: {self.time_hit}")
            print(f"Best pure search value: {best_search_value:.1f}")
            print(f"Blunder margin: {BLUNDER_MARGIN_CP:.1f}")
            print(f"Safe moves count: {len(safe_moves)} / {len(move_values)}")
            # Sort moves by search value for inspection
            debug_items = sorted(
                move_values.items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            for mv, val in debug_items[:DEBUG_TOP_K]:
                leaf = move_leaf_evals.get(mv, float('nan'))
                comb = combined_scores.get(mv, float('nan'))
                print(
                    f"Move {mv.uci():6s}  search={val:8.1f}  "
                    f"leaf={leaf:8.1f}  comb={comb:8.1f}"
                )
            print(f"Chosen move: {best_move.uci()}")
            print("================================")

        # Convert move_values to a probability distribution for logging
        if move_values:
            max_v = max(move_values.values())
            exps = {}
            for mv, v in move_values.items():
                scaled = (v - max_v) / max(CP_SOFTMAX_TEMPERATURE, 1e-6)
                scaled = max(-60.0, min(60.0, scaled))
                exps[mv] = math.exp(scaled)

            Z = sum(exps.values())
            if Z > 0:
                move_probs = {mv: exps.get(mv, 0.0) / Z for mv in legal_moves}
            else:
                # fallback uniform
                p = 1.0 / len(legal_moves)
                move_probs = {mv: p for mv in legal_moves}
        else:
            p = 1.0 / len(legal_moves)
            move_probs = {mv: p for mv in legal_moves}

        return best_move, move_probs

    # (optional) pure policy move, kept as utility / backup
    def select_move_policy_only(self, board: chess.Board):
        """
        One-shot policy softmax, no search.
        Returns (best_move, move_prob_dict).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        logits = self._policy_logits(board)
        scores = []

        for mv in legal_moves:
            idx = move_to_index(mv)
            if 0 <= idx < POLICY_DIM:
                scores.append(float(logits[idx]))
            else:
                scores.append(-1e9)

        max_s = max(scores)
        exps = [np.exp(s - max_s) if s > -1e8 else 0.0 for s in scores]
        Z = float(sum(exps))

        if Z <= 0.0:
            p = 1.0 / len(legal_moves)
            move_probs = {mv: p for mv in legal_moves}
            best_move = random.choice(legal_moves)
            return best_move, move_probs

        probs = [e / Z for e in exps]
        move_probs = {mv: p for mv, p in zip(legal_moves, probs)}

        best_idx = int(np.argmax(probs))
        best_move = legal_moves[best_idx]
        return best_move, move_probs

    def reset(self):
        # Clear caches for a new game
        self.eval_cache.clear()
        self.policy_cache.clear()
        self.search_deadline = None
        self.nodes = 0
        self.time_hit = False


# Try to construct the engine once at import
try:
    if ResNet20_PolicyDelta is not None:
        ENGINE = PolicyOnlyEngine()
    else:
        ENGINE = None
except Exception as e:
    ENGINE = None
    print(f"[ENGINE] Failed to initialize NN engine, falling back to random. Error: {e}")


# ============================================================
# Submission platform hooks
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    # tiny delay if you want logs to flush; can be set to 0
    time.sleep(0.005)

    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # If NN engine didn't initialize, fall back to random baseline
    if ENGINE is None:
        print("Safety (no NN engine, using random)")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # -------------------------------
    # MAIN PATH: negamax search
    # -------------------------------
    try:
        best_move, move_probs = ENGINE.search_best_move(board, MAX_SEARCH_DEPTH)
    except Exception as e:
        # Extra safety: if search crashes, fall back to policy-only
        print(f"[ENGINE] Search failed ({e}), falling back to policy-only.")
        best_move, move_probs = ENGINE.select_move_policy_only(board)

    if best_move is None:
        # Extra safety fallback
        print("Safety (no best_move from engine, using random)")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Ensure probabilities are defined over exactly current legal moves
    probs_filtered = {mv: move_probs.get(mv, 0.0) for mv in legal_moves}
    Z = sum(probs_filtered.values())
    if Z > 0:
        probs_filtered = {mv: p / Z for mv, p in probs_filtered.items()}
    else:
        # Degenerate case → uniform
        p = 1.0 / len(legal_moves)
        probs_filtered = {mv: p for mv in legal_moves}

    # This is what the website uses to show the move probability bar chart.
    ctx.logProbabilities(probs_filtered)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    if ENGINE is not None:
        ENGINE.reset()
