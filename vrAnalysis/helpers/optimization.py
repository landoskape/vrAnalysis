import math
from typing import Callable, Optional, Tuple, List, Dict


def golden_section_search(
    func: Callable[[float], float],
    a: float,
    b: float,
    tolerance_param: float = 1e-2,
    tolerance_score: Optional[float] = None,
    max_iterations: int = 50,
    minimize: bool = True,
    logspace: bool = False,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Perform golden section search for 1D function minimization or maximization.

    This can optionally search in log-space while always calling `func` with the
    parameter in the original domain.

    Parameters
    ----------
    func : Callable[[float], float]
        Function to optimize. Takes a single float (original-domain parameter) and
        returns a float (score).
    a : float
        Lower bound of the search interval (original domain).
    b : float
        Upper bound of the search interval (original domain). Must satisfy a < b.
    tolerance_param : float, default=1e-2
        Tolerance for the search interval in the *search* coordinate.
        - If logspace=False: tolerance in raw units.
        - If logspace=True: tolerance in log units.
    tolerance_score : float, optional
        Optional tolerance for relative improvement in function value.
        For minimization:
            (prev_best - recent_best) / |prev_best| < tolerance_score
        For maximization:
            (recent_best - prev_best) / |prev_best| < tolerance_score
        Only checked after at least 6 function evaluations.
    max_iterations : int, default=50
        Maximum number of iterations.
    minimize : bool, default=True
        If True, minimize the function. If False, maximize the function.
    logspace : bool, default=False
        If True, perform the search in log-space:
        - `a` and `b` are interpreted as positive original-domain bounds.
        - Internally we search over log(param), but `func` is always called with
          the original param.

    Returns
    -------
    best_param : float
        Best parameter in the original domain.
    best_score : float
        Function value at best_param (original score, not negated).
    history : list[tuple[float, float]]
        List of (param, score) pairs in the original domain for all evaluations.

    Raises
    ------
    ValueError
        If a >= b or if logspace=True and a/b are non-positive.
    """
    if a >= b:
        raise ValueError(f"Lower bound a ({a}) must be less than upper bound b ({b})")
    if logspace and (a <= 0 or b <= 0):
        raise ValueError("logspace=True requires a > 0 and b > 0")

    # Golden ratio constant: (sqrt(5) - 1) / 2 ≈ 0.618
    golden_ratio = (math.sqrt(5) - 1) / 2

    # Define the search coordinate: raw or log-space
    if logspace:
        search_a = math.log(a)
        search_b = math.log(b)
    else:
        search_a, search_b = a, b

    # Cache: search_coordinate -> (score_for_min, original_score, original_param)
    evaluated_points: Dict[float, Tuple[float, float, float]] = {}

    def evaluate(param_value: float) -> Tuple[float, float, float]:
        """Evaluate function with caching.

        Returns
        -------
        score_for_min : float
            Score in the "minimization" space (negated if maximize).
        original_score : float
            Original score returned by `func`.
        x : float
            Parameter in the original domain.
        """
        if param_value in evaluated_points:
            return evaluated_points[param_value]

        # Map from search coordinate to original-domain parameter
        x = math.exp(param_value) if logspace else param_value

        original_score = func(x)
        score_for_min = -original_score if not minimize else original_score

        evaluated_points[param_value] = (score_for_min, original_score, x)
        return score_for_min, original_score, x

    # Initialize with two interior points in search space
    c = search_b - golden_ratio * (search_b - search_a)
    d = search_a + golden_ratio * (search_b - search_a)
    fc, fc_orig, xc = evaluate(c)
    fd, fd_orig, xd = evaluate(d)

    # History in original domain
    history: List[Tuple[float, float]] = [(xc, fc_orig), (xd, fd_orig)]

    # Global best (tracked in minimization space but we keep original too)
    best_score_min = fc
    best_param_search = c
    best_param_original = xc
    best_score_original = fc_orig
    if fd < best_score_min:
        best_score_min = fd
        best_param_search = d
        best_param_original = xd
        best_score_original = fd_orig

    iteration = 0

    # Golden section search loop
    while iteration < max_iterations:
        iteration += 1

        # Parameter tolerance in search coordinates
        if iteration > 1 and (search_b - search_a) < tolerance_param:
            break

        # Narrow the interval
        if fc < fd:
            # Minimum is in [search_a, d]
            search_b = d
            d, fd, fd_orig, xd = c, fc, fc_orig, xc
            c = search_b - golden_ratio * (search_b - search_a)
            fc, fc_orig, xc = evaluate(c)
            history.append((xc, fc_orig))
        else:
            # Minimum is in [c, search_b]
            search_a = c
            c, fc, fc_orig, xc = d, fd, fd_orig, xd
            d = search_a + golden_ratio * (search_b - search_a)
            fd, fd_orig, xd = evaluate(d)
            history.append((xd, fd_orig))

        # Update global best (in minimization space)
        if fc < best_score_min:
            best_score_min = fc
            best_param_search = c
            best_param_original = xc
            best_score_original = fc_orig
        if fd < best_score_min:
            best_score_min = fd
            best_param_search = d
            best_param_original = xd
            best_score_original = fd_orig

        # Optional score-based early stopping
        if tolerance_score is not None and len(history) > 6:
            recent_scores = [s for _, s in history[-3:]]
            prev_scores = [s for _, s in history[:-3]]

            if minimize:
                best_recent = min(recent_scores)
                prev_best = min(prev_scores)
                if prev_best != 0:
                    rel_improvement = (prev_best - best_recent) / abs(prev_best)
                    if rel_improvement < tolerance_score:
                        break
            else:
                best_recent = max(recent_scores)
                prev_best = max(prev_scores)
                if prev_best != 0:
                    rel_improvement = (best_recent - prev_best) / abs(prev_best)
                    if rel_improvement < tolerance_score:
                        break

    return best_param_original, best_score_original, history
