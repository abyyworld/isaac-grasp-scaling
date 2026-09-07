"""Measuring how much a re-run moves a point.

The curve's scatter decomposition infers training variance by subtracting the
binomial term from the residual about a fitted line, which charges any curvature
in the true relationship to training noise. Repeated runs at one size assume
nothing: same data, same evaluation scenes, different training seed.

These tests pin the arithmetic, including the two ways it can quietly mislead:
counting evaluation noise as training noise, and pooling sizes as though a
two-seed group were as informative as a three-seed one.
"""

from __future__ import annotations

import math

import pytest

from isaacgrasp.scaling import measure_seed_variance


def _point(samples, seed, rate, n=1500):
    return {"train_samples": samples, "train_seed": seed,
            "unseen": {"rate": rate, "n": n}}


def test_a_single_seed_per_size_supports_no_estimate():
    """One run cannot tell you how much a re-run would move."""
    result = measure_seed_variance([_point(384, 0, 0.43), _point(1536, 0, 0.45)])
    assert result["per_size"] == []
    assert "pooled" not in result


def test_identical_replicates_leave_no_training_variance():
    """Same number three times means the only spread is the sampling floor."""
    points = [_point(1536, seed, 0.45) for seed in (0, 1, 2)]
    result = measure_seed_variance(points)
    row = result["per_size"][0]
    assert row["n_seeds"] == 3
    assert row["observed_sd_pp"] == pytest.approx(0.0)
    assert row["training_sd_pp"] == 0.0


def test_observed_spread_is_the_sample_standard_deviation():
    points = [_point(1536, 0, 0.40), _point(1536, 1, 0.50), _point(1536, 2, 0.45)]
    row = measure_seed_variance(points)["per_size"][0]
    expected = math.sqrt(((40 - 45) ** 2 + (50 - 45) ** 2 + 0) / 2)
    assert row["observed_sd_pp"] == pytest.approx(expected)
    assert row["mean_pp"] == pytest.approx(45.0)
    assert row["range_pp"] == pytest.approx(10.0)
    assert row["seeds"] == [0, 1, 2]


def test_evaluation_noise_is_removed_from_the_training_estimate():
    """Spread no larger than the binomial floor is not evidence of run variance."""
    points = [_point(1536, 0, 0.450), _point(1536, 1, 0.455), _point(1536, 2, 0.452)]
    row = measure_seed_variance(points)["per_size"][0]
    assert row["evaluation_sd_pp"] > row["observed_sd_pp"]
    assert row["training_sd_pp"] == 0.0, "must floor at zero, not go imaginary"


def test_more_evaluation_episodes_shift_spread_into_the_training_term():
    """The same observed spread means more when each measurement is tighter."""
    rates = (0.40, 0.50, 0.45)
    loose = measure_seed_variance([_point(1536, i, r, n=100)
                                   for i, r in enumerate(rates)])["per_size"][0]
    tight = measure_seed_variance([_point(1536, i, r, n=5000)
                                   for i, r in enumerate(rates)])["per_size"][0]
    assert loose["observed_sd_pp"] == pytest.approx(tight["observed_sd_pp"])
    assert tight["training_sd_pp"] > loose["training_sd_pp"]


def test_pooling_weights_by_degrees_of_freedom():
    """Three seeds at one size must count twice as much as two at another."""
    points = [
        _point(384, 0, 0.40), _point(384, 1, 0.50), _point(384, 2, 0.45),
        _point(1536, 0, 0.44), _point(1536, 1, 0.46),
    ]
    result = measure_seed_variance(points)
    pooled = result["pooled"]
    assert pooled["degrees_of_freedom"] == 3, "2 from the triple, 1 from the pair"

    small = next(r for r in result["per_size"] if r["train_samples"] == 384)
    large = next(r for r in result["per_size"] if r["train_samples"] == 1536)
    expected = math.sqrt((2 * small["observed_sd_pp"] ** 2
                          + 1 * large["observed_sd_pp"] ** 2) / 3)
    assert pooled["observed_sd_pp"] == pytest.approx(expected)


def test_sizes_are_reported_in_order():
    points = [_point(6144, 0, 0.48), _point(6144, 1, 0.50),
              _point(384, 0, 0.43), _point(384, 1, 0.45)]
    sizes = [row["train_samples"] for row in measure_seed_variance(points)["per_size"]]
    assert sizes == [384, 6144]


def test_points_without_an_episode_count_are_skipped():
    """A point written before the count was recorded must not crash the estimate."""
    points = [_point(1536, 0, 0.44), _point(1536, 1, 0.46),
              {"train_samples": 1536, "train_seed": 2, "unseen": {"rate": 0.45}}]
    row = measure_seed_variance(points)["per_size"][0]
    assert row["n_seeds"] == 2
