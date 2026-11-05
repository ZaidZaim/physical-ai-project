"""Tests run on the code's defaults, not on this laptop's .env (its booth settings, such as AUTO_PLACE=false, broke the place tests)."""
import os

os.environ["PUPPY_NO_DOTENV"] = "1"

import pytest


@pytest.fixture(autouse=True)
def _drive_trim_in_tmp(tmp_path, monkeypatch):
    """The heading hold keeps what it learns in logs/drive_trim.json: tests must neither read the robot's real one nor write it."""
    monkeypatch.setenv("DRIVE_TRIM_FILE", str(tmp_path / "drive_trim.json"))
