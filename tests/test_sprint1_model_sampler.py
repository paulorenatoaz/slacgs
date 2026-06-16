"""Tests for the Sprint 1 Gaussian model, dataset, and sampler (Checkpoint 1)."""

import numpy as np
import pytest

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler, derive_seed


# --- Scenario 1 model fixtures ------------------------------------------------

MU0 = [-0.70, -0.55, -0.30]
MU1 = [0.70, 0.55, 0.30]
SIGMA = [
    [1.00, 0.35, 0.05],
    [0.35, 1.00, 0.05],
    [0.05, 0.05, 1.00],
]


def make_model():
    return GaussianSimulationModel(
        means={0: MU0, 1: MU1},
        covariances={0: SIGMA, 1: SIGMA},
    )


# --- GaussianSimulationModel validity ----------------------------------------

def test_valid_model_infers_d_and_labels():
    model = make_model()
    assert model.d == 3
    assert model.num_channels == 3
    assert model.K == 2
    assert model.num_classes == 2
    assert model.class_labels == (0, 1)
    assert np.allclose(model.mean(0), MU0)
    assert np.allclose(model.covariance(1), SIGMA)


def test_model_stores_numpy_arrays():
    model = make_model()
    assert isinstance(model.mean(0), np.ndarray)
    assert isinstance(model.covariance(0), np.ndarray)


def test_model_mismatched_class_labels_raises():
    with pytest.raises(ValueError):
        GaussianSimulationModel(
            means={0: MU0, 1: MU1},
            covariances={0: SIGMA, 2: SIGMA},
        )


def test_model_single_class_raises():
    with pytest.raises(ValueError):
        GaussianSimulationModel(means={0: MU0}, covariances={0: SIGMA})


def test_model_mean_wrong_length_raises():
    with pytest.raises(ValueError):
        GaussianSimulationModel(
            means={0: [0.0, 0.0], 1: MU1},
            covariances={0: SIGMA, 1: SIGMA},
        )


def test_model_covariance_wrong_shape_raises():
    bad_cov = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        GaussianSimulationModel(
            means={0: MU0, 1: MU1},
            covariances={0: bad_cov, 1: SIGMA},
        )


def test_model_non_symmetric_covariance_raises():
    asym = [
        [1.00, 0.35, 0.05],
        [0.30, 1.00, 0.05],
        [0.05, 0.05, 1.00],
    ]
    with pytest.raises(ValueError):
        GaussianSimulationModel(
            means={0: MU0, 1: MU1},
            covariances={0: asym, 1: SIGMA},
        )


def test_model_non_positive_definite_covariance_raises():
    npd = [
        [1.0, 2.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError):
        GaussianSimulationModel(
            means={0: MU0, 1: MU1},
            covariances={0: npd, 1: SIGMA},
        )


def test_model_restrict_to_subset():
    model = make_model()
    restricted = model.restrict_to_subset((0, 2))
    assert restricted.d == 2
    assert np.allclose(restricted.mean(0), [MU0[0], MU0[2]])
    expected_cov = np.array(SIGMA)[np.ix_([0, 2], [0, 2])]
    assert np.allclose(restricted.covariance(0), expected_cov)


def test_model_restrict_empty_subset_raises():
    model = make_model()
    with pytest.raises(ValueError):
        model.restrict_to_subset(())


# --- Dataset ------------------------------------------------------------------

def test_dataset_shape_validation():
    X = np.zeros((5, 3))
    y = np.zeros(5)
    ds = Dataset(X, y)
    assert ds.n_samples == 5
    assert ds.d == 3


def test_dataset_mismatched_rows_raises():
    with pytest.raises(ValueError):
        Dataset(np.zeros((5, 3)), np.zeros(4))


def test_dataset_bad_ndim_raises():
    with pytest.raises(ValueError):
        Dataset(np.zeros(5), np.zeros(5))


def test_dataset_select_channels():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.array([0, 0, 1, 1])
    ds = Dataset(X, y)
    sub = ds.select_channels((0, 2))
    assert sub.d == 2
    assert np.allclose(sub.X, X[:, [0, 2]])
    assert np.allclose(sub.y, y)


def test_dataset_select_channels_out_of_range_raises():
    ds = Dataset(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(ValueError):
        ds.select_channels((0, 5))


# --- Sampler ------------------------------------------------------------------

def test_sampler_train_balanced():
    sampler = GaussianClassConditionalSampler(make_model(), base_seed=42)
    ds = sampler.sample_train(n_per_class=10, replication_id=0)
    assert ds.n_samples == 20
    counts = {label: int(np.sum(ds.y == label)) for label in (0, 1)}
    assert counts == {0: 10, 1: 10}


def test_sampler_train_deterministic():
    s1 = GaussianClassConditionalSampler(make_model(), base_seed=7)
    s2 = GaussianClassConditionalSampler(make_model(), base_seed=7)
    d1 = s1.sample_train(n_per_class=16, replication_id=3)
    d2 = s2.sample_train(n_per_class=16, replication_id=3)
    assert np.allclose(d1.X, d2.X)
    assert np.array_equal(d1.y, d2.y)


def test_sampler_different_replication_differs():
    sampler = GaussianClassConditionalSampler(make_model(), base_seed=7)
    d_r0 = sampler.sample_train(n_per_class=16, replication_id=0)
    d_r1 = sampler.sample_train(n_per_class=16, replication_id=1)
    assert not np.allclose(d_r0.X, d_r1.X)


def test_sampler_train_prefix_nested():
    sampler = GaussianClassConditionalSampler(make_model(), base_seed=123)
    small = sampler.sample_train(n_per_class=16, replication_id=5)
    large = sampler.sample_train(n_per_class=64, replication_id=5)

    # Per class, the first 16 rows of the large draw must match the small draw.
    for label in (0, 1):
        small_rows = small.X[small.y == label]
        large_rows = large.X[large.y == label]
        assert small_rows.shape[0] == 16
        assert large_rows.shape[0] == 64
        assert np.allclose(small_rows, large_rows[:16])


def test_sampler_test_set_fixed_and_cached():
    sampler = GaussianClassConditionalSampler(
        make_model(), base_seed=1, test_samples_per_class=200
    )
    t1 = sampler.sample_test()
    t2 = sampler.sample_test()
    assert t1 is t2  # cached
    assert t1.n_samples == 400
    counts = {label: int(np.sum(t1.y == label)) for label in (0, 1)}
    assert counts == {0: 200, 1: 200}


def test_sampler_test_independent_of_replication_seed_path():
    # Two samplers with same base seed produce identical fixed test sets.
    s1 = GaussianClassConditionalSampler(make_model(), base_seed=99, test_samples_per_class=50)
    s2 = GaussianClassConditionalSampler(make_model(), base_seed=99, test_samples_per_class=50)
    assert np.allclose(s1.sample_test().X, s2.sample_test().X)


def test_derive_seed_is_deterministic_and_split_sensitive():
    a = derive_seed(0, 0, split="train", replication_id=0)
    b = derive_seed(0, 0, split="train", replication_id=0)
    c = derive_seed(0, 0, split="test")
    assert a == b
    assert a != c
