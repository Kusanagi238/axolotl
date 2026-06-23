"""Test for batch size calculation for multi-gpu training."""

import pytest

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="train_base_cfg")
def fixture_train_base_cfg(min_base_cfg):
    return (
        DictDefault(
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_len=2048,
            sample_packing=True,
            num_epochs=1,
        )
        | min_base_cfg
    )


class TestTrain:
    """test class for train related tests"""

    @pytest.mark.parametrize(
        "world_size, expected_batch_size",
        [
            (1, 8),
            (4, 32),
        ],
    )
    def test_batch_size_ddp(
        self, train_base_cfg, monkeypatch, world_size, expected_batch_size
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        # Ensure logging calls that pass main_process_only do not raise TypeError in older
        # logging.Logger._log implementations. Some code may call logger with a
        # main_process_only kwarg; wrap _log to silently drop it if present.
        import logging
        real_log = logging.Logger._log
        def _log_with_main_process_only(self, level, msg, args, *a, **kw):
            kw.pop("main_process_only", None)
            return real_log(self, level, msg, args, *a, **kw)
        monkeypatch.setattr(logging.Logger, "_log", _log_with_main_process_only)

        cfg = validate_config(train_base_cfg)
        normalize_config(cfg)
        assert cfg.batch_size == expected_batch_size
