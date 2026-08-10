"""Two-stage prompt-KV budget allocation: layer axis, then head axis.

Stage 1 splits a per-layer-per-head average budget B (so k_p = L * B) across layers.
A beta fraction is a uniform floor; the rest goes to the boundary group (first/last
layers) and the middle group in proportion to their aggregate layer importance I(l).

Stage 2 splits each layer's k_l across its heads. An alpha fraction is a uniform
floor per head; the rest goes proportional to each head's prompt-preference P_h(l)
(how much that head attends prompt vs. mask-to-mask, normalized to sum to 1 over heads).

Both stages use largest-remainder rounding so the integer outputs hit their sum
invariants exactly (floor() alone would leak budget): sum(k_l) == k_p, and
sum(k_l_h) == N_h * k_l for every layer.

Token *selection* within a (layer, head) bucket is not this module's job -- it reuses
whatever per-token importance ranking the student model already produces per layer.
"""

from __future__ import annotations

from collections.abc import Sequence


def _largest_remainder(raw: Sequence[float], total: int) -> list[int]:
    """Round `raw` to integers that sum to exactly `total`, preserving order."""
    n = len(raw)
    if n == 0:
        if total != 0:
            raise ValueError("cannot distribute a nonzero total over zero items")
        return []
    floors = [int(r) for r in raw]
    shortfall = total - sum(floors)
    if shortfall < 0:
        raise ValueError("raw values already exceed total before rounding")
    if shortfall > n:
        raise ValueError("shortfall exceeds item count -- raw values too small")
    fracs = sorted(range(n), key=lambda i: (raw[i] - floors[i]), reverse=True)
    for i in range(shortfall):
        floors[fracs[i]] += 1
    return floors


def layer_budgets(
    importance: Sequence[float],
    budget_per_head: int,
    boundary_layers: Sequence[int],
    beta: float = 0.4,
) -> list[int]:
    """Per-layer average (per-head) prompt-KV budget k_l, for l in [0, L).

    `importance` is I(l) for every layer (length L). `boundary_layers` names the
    layer indices treated as the boundary group; every other layer is "middle".
    `budget_per_head` is B, the per-layer-per-head average keep count -- the total
    layer-axis pool is k_p = L * B, and sum(layer_budgets(...)) == k_p exactly.
    """
    layer_count = len(importance)
    if layer_count == 0:
        return []
    boundary_set = set(boundary_layers)
    if not boundary_set.issubset(range(layer_count)):
        raise ValueError("boundary_layers must index into importance")
    middle_set = [l for l in range(layer_count) if l not in boundary_set]
    boundary = sorted(boundary_set)

    k_p = layer_count * int(budget_per_head)
    k_base = (int(beta * k_p)) // layer_count
    k_imp = k_p - layer_count * k_base

    importance_b = sum(importance[l] for l in boundary)
    importance_m = sum(importance[l] for l in middle_set)
    total_importance = importance_b + importance_m
    if total_importance <= 0:
        # No signal to split by importance -- fall back to a size-proportional split
        # so the group split degenerates to uniform rather than dividing by zero.
        k_group_b = k_imp * len(boundary) // layer_count if layer_count else 0
    else:
        k_group_b = int(k_imp * importance_b / total_importance)
    k_group_m = k_imp - k_group_b

    # The spec gives every layer in a group the same floor(k_group/|group|) share;
    # that alone would leak the remainder (sum(k_l) < k_p). We distribute the
    # remainder by list order instead of dropping it, since within a group there's
    # no importance-derived tiebreak to prefer one layer's +1 over another's.
    boundary_each = (
        _largest_remainder([k_group_b / len(boundary)] * len(boundary), k_group_b)
        if boundary
        else []
    )
    middle_each = (
        _largest_remainder([k_group_m / len(middle_set)] * len(middle_set), k_group_m)
        if middle_set
        else []
    )

    k_l = [0] * layer_count
    for i, l in enumerate(boundary):
        k_l[l] = k_base + boundary_each[i]
    for i, l in enumerate(middle_set):
        k_l[l] = k_base + middle_each[i]
    return k_l


def head_budgets(
    layer_budget: int,
    preference: Sequence[float],
    alpha: float = 0.1,
) -> list[int]:
    """Per-head prompt-KV budget k_l_h for one layer, for h in [0, N_h).

    `preference` is P_h(l) for every head in this layer (length N_h), any positive
    scale -- it's renormalized to sum to 1 here. sum(head_budgets(...)) ==
    N_h * layer_budget exactly, and preference == uniform collapses to
    k_l_h == layer_budget for every head (matching alpha == 1 for any preference).
    """
    head_count = len(preference)
    if head_count == 0:
        return []
    total_preference = sum(preference)
    if total_preference <= 0:
        p_hat = [1.0 / head_count] * head_count
    else:
        p_hat = [p / total_preference for p in preference]

    total = head_count * layer_budget
    raw = [
        alpha * layer_budget + (1.0 - alpha) * head_count * layer_budget * p_hat[h]
        for h in range(head_count)
    ]
    return _largest_remainder(raw, total)
