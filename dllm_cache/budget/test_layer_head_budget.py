"""Standalone sanity checks for layer_head_budget.py -- no pytest, no GPU.

Run directly: python dllm_cache/budget/test_layer_head_budget.py
"""

from __future__ import annotations

import random

from dllm_cache.budget.layer_head_budget import head_budgets, layer_budgets


def check(name: str, cond: bool) -> None:
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(1)


def test_layer_budgets_sum_conserved() -> None:
    rng = random.Random(0)
    for trial in range(50):
        layer_count = rng.randint(4, 40)
        budget_per_head = rng.randint(1, 64)
        importance = [rng.random() for _ in range(layer_count)]
        boundary = list(range(min(2, layer_count))) + list(
            range(max(layer_count - 2, 0), layer_count)
        )
        boundary = sorted(set(boundary))
        k_l = layer_budgets(importance, budget_per_head, boundary, beta=0.4)
        check(
            f"trial {trial}: sum(k_l) == L*B ({sum(k_l)} vs {layer_count * budget_per_head})",
            sum(k_l) == layer_count * budget_per_head,
        )
        check(f"trial {trial}: k_l all non-negative", all(k >= 0 for k in k_l))


def test_layer_budgets_beta_one_is_uniform_when_divisible() -> None:
    layer_count = 8
    budget_per_head = 16  # divisible by layer_count -> k_imp == 0 when beta == 1
    importance = [random.random() for _ in range(layer_count)]
    boundary = [0, 1, layer_count - 2, layer_count - 1]
    k_l = layer_budgets(importance, budget_per_head, boundary, beta=1.0)
    check("beta=1, divisible budget -> every layer gets exactly B", all(k == budget_per_head for k in k_l))


def test_layer_budgets_beta_zero_ignores_floor() -> None:
    layer_count = 6
    budget_per_head = 10
    # All importance in the middle group -> boundary layers should get ~k_base (~0).
    importance = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
    boundary = [0, 1, 4, 5]
    k_l = layer_budgets(importance, budget_per_head, boundary, beta=0.0)
    check("beta=0: boundary layers starve when importance is all-middle", all(k_l[l] == 0 for l in boundary))
    check("beta=0: sum still conserved", sum(k_l) == layer_count * budget_per_head)


def test_head_budgets_sum_conserved() -> None:
    rng = random.Random(1)
    for trial in range(50):
        head_count = rng.randint(2, 32)
        layer_budget = rng.randint(0, 100)
        preference = [rng.random() for _ in range(head_count)]
        k_h = head_budgets(layer_budget, preference, alpha=0.1)
        check(
            f"trial {trial}: sum(k_l_h) == N_h*k_l ({sum(k_h)} vs {head_count * layer_budget})",
            sum(k_h) == head_count * layer_budget,
        )
        check(f"trial {trial}: k_l_h all non-negative", all(k >= 0 for k in k_h))


def test_head_budgets_alpha_one_is_uniform() -> None:
    head_count = 12
    layer_budget = 37
    preference = [random.random() for _ in range(head_count)]
    k_h = head_budgets(layer_budget, preference, alpha=1.0)
    check("alpha=1: every head gets exactly k_l regardless of preference", all(k == layer_budget for k in k_h))


def test_head_budgets_uniform_preference_is_uniform() -> None:
    head_count = 9
    layer_budget = 21
    preference = [3.0] * head_count  # uniform, any positive scale
    k_h = head_budgets(layer_budget, preference, alpha=0.1)
    check("uniform P_hat -> every head gets exactly k_l", all(k == layer_budget for k in k_h))


def test_head_budgets_zero_preference_falls_back_to_uniform() -> None:
    head_count = 5
    layer_budget = 8
    preference = [0.0] * head_count
    k_h = head_budgets(layer_budget, preference, alpha=0.1)
    check("all-zero preference -> uniform fallback, sum conserved", sum(k_h) == head_count * layer_budget)


def test_head_budgets_skewed_preference_favors_high_head() -> None:
    layer_budget = 40
    preference = [0.01, 0.01, 0.01, 0.97]
    k_h = head_budgets(layer_budget, preference, alpha=0.1)
    check("skewed preference -> favored head gets the most budget", k_h[3] == max(k_h))
    check("skewed preference -> favored head well above the floor", k_h[3] > layer_budget)


if __name__ == "__main__":
    test_layer_budgets_sum_conserved()
    test_layer_budgets_beta_one_is_uniform_when_divisible()
    test_layer_budgets_beta_zero_ignores_floor()
    test_head_budgets_sum_conserved()
    test_head_budgets_alpha_one_is_uniform()
    test_head_budgets_uniform_preference_is_uniform()
    test_head_budgets_zero_preference_falls_back_to_uniform()
    test_head_budgets_skewed_preference_favors_high_head()
    print("all layer_head_budget checks passed")
