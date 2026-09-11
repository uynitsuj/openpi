import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _MarkerDataset:
    """Map-style stub whose samples identify their source dataset."""

    def __init__(self, marker: int, size: int):
        self._marker = marker
        self._size = size

    def __getitem__(self, index):
        return {"source": self._marker, "index": int(index)}

    def __len__(self) -> int:
        return self._size


def test_weighted_mixture_dataset():
    # Component sizes are deliberately lopsided: sampling must follow the
    # weights, not the sizes.
    small, large = _MarkerDataset(0, 10), _MarkerDataset(1, 1000)
    mixture = _data_loader.WeightedMixtureDataset([small, large], [0.9, 0.1], seed=0)

    assert len(mixture) == 1010

    draws = [mixture[i]["source"] for i in range(1000)]
    large_frac = sum(draws) / len(draws)
    assert 0.05 < large_frac < 0.15  # ≈ 0.1 despite `large` being 100x bigger

    # Deterministic: same (seed, index) → same sample; same seed → same stream.
    replay = _data_loader.WeightedMixtureDataset([small, large], [0.9, 0.1], seed=0)
    assert [replay[i]["source"] for i in range(1000)] == draws
    assert replay[3] == mixture[3]

    # Explicit virtual length override.
    sized = _data_loader.WeightedMixtureDataset([small, large], [1.0, 1.0], length=50)
    assert len(sized) == 50


def test_with_fake_mixture():
    config = _config.get_config("debug_mixture")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2, shuffle=True)
    # The mixture reports the primary component's data config.
    assert loader.data_config().repo_id == "fake"

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
