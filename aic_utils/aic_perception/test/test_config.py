"""Configuration loading and strict validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aic_perception.config import CONFIG_DIR, load_config


@pytest.mark.parametrize("kind", ["collect", "train", "infer"])
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
