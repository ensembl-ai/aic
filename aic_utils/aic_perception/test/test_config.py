"""Configuration loading and strict validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aic_perception.config import CONFIG_DIR, load_config


@pytest.mark.parametrize("kind", ["collect", "train", "refine", "infer"])
def test_shipped_configs_load(kind: str) -> None:
    """Every shipped configuration validates and resolves absolute paths."""

    config = load_config(kind)
    assert Path(config["dataset"]["directory"]).is_absolute()


def edited_copy(tmp_path: Path, name: str, edit) -> Path:
    """Copy the configuration directory and edit one file."""

    directory = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, directory)
    path = directory / name
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    return directory


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """Typos fail loudly instead of being ignored."""

    directory = edited_copy(
        tmp_path, "train.json", lambda d: d["model"].update(dmodel=1)
    )
    with pytest.raises(KeyError, match="Unknown"):
        load_config("train", directory)


def test_invalid_rendering_mode_is_rejected(tmp_path: Path) -> None:
    """Only supported rendering modes are accepted."""

    directory = edited_copy(
        tmp_path, "collect.json", lambda d: d["docker"].update(rendering="vulkan")
    )
    with pytest.raises(ValueError, match="rendering"):
        load_config("collect", directory)


def test_duplicate_target_ids_are_rejected(tmp_path: Path) -> None:
    """The board schema must map every query to a unique ID."""

    def duplicate(data: dict) -> None:
        """Duplicate the last target."""

        data["schema"]["targets"].append(dict(data["schema"]["targets"][-1]))

    directory = edited_copy(tmp_path, "base.json", duplicate)
    with pytest.raises(ValueError, match="unique"):
        load_config("infer", directory)


def test_dinov2_input_must_be_multiples_of_14(tmp_path: Path) -> None:
    """DINOv2 patches are 14 pixels, so its input size must divide evenly."""

    directory = edited_copy(
        tmp_path, "train.json", lambda d: d["model"].update(backbone="dinov2_vits14", backbone_input=[576, 512])
    )
    with pytest.raises(ValueError, match="multiples of 14"):
        load_config("train", directory)


def test_sc_rank_rule_must_be_a_permutation(tmp_path: Path) -> None:
    """Both SC slots map onto distinct ranks."""

    directory = edited_copy(
        tmp_path, "infer.json", lambda d: d.update(sc_rank_for_slot={"sc_port_0": 0, "sc_port_1": 0})
    )
    with pytest.raises(ValueError, match="sc_rank_for_slot"):
        load_config("infer", directory)
