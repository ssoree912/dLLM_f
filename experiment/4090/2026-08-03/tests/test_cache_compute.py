from __future__ import annotations

from step_distill.cache_compute import (
    CacheFlopRequest,
    CacheSchedule,
    ModelShape,
    estimate_cache_flops,
)


def test_top128_reduces_attention_flops_by_four_point_five_at_cache1024() -> None:
    # Given
    shape = ModelShape(
        layers=32,
        hidden_size=4096,
        mlp_hidden_size=12288,
        vocabulary_size=126464,
    )
    all_request = CacheFlopRequest(
        shape=shape,
        schedule=CacheSchedule(
            steps=128,
            refresh_interval=4,
            suffix_length=128,
            cache_budget=1024,
            attention_budget=1024,
        ),
        prompt_length=2048,
    )
    topk_request = CacheFlopRequest(
        shape=shape,
        schedule=CacheSchedule(
            steps=128,
            refresh_interval=4,
            suffix_length=128,
            cache_budget=1024,
            attention_budget=128,
        ),
        prompt_length=2048,
    )

    # When
    all_cache = estimate_cache_flops(all_request)
    top128 = estimate_cache_flops(topk_request)

    # Then
    assert all_cache.attention_flops / top128.attention_flops == 4.5
    assert all_cache.dense_flops == top128.dense_flops
    assert all_cache.logit_flops == top128.logit_flops
