"""Scratch-tree path handling and the sweeper.

The upload never reaches disk; the only file this process writes is the overlay PNG a user asked to
see. That makes the two ways this module can fail rather different: a path bug exposes someone else's
picture, and a sweeper bug fills the disk. Both are tested here, and the path tests are all about
what is *refused* — the allowlist is the defence, and ``_under`` is the belt.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.job import new_job_id
from app.core.scratch import Scratch, job_id_safe, path_safe, write_overlay

MALFORMED_IDS = [
    "",
    "..",
    "..\\..\\..\\windows\\system32",
    "../../../etc",
    "1a1a-../evil",
    "/absolute/path-1",
    "1a1a1b65238-04997ce21a0892f6a2e349f3466c5a38-extra",
    "1A1A1B65238-04997CE21A0892F6A2E349F3466C5A38",   # uppercase: not a shape we mint
    "1a1a1b65238|ela",                                # a pipe would re-align the signed message
    "1a1a1b65238-",                                   # nonce missing
    "-04997ce21a0892f6a2e349f3466c5a38",              # timestamp missing
    "%2e%2e%2f%2e%2e%2fapp",                          # percent-encoded traversal
    "1a1a1b65238-04997ce21a0892f6a2e349f3466c5a38.png",
]

MALFORMED_NAMES = [
    "",
    "Ela",                    # uppercase
    "1ela",                   # must start with a letter
    "ela.png",                # the extension is added by the module, not taken from the URL
    "../ela",
    "ela/../../app/config",
    "ela|cnn",
    "ela\x00",
    "a" * 40,                 # longer than the 32-character ceiling
    "ela-bill",               # hyphens are not in our identifier class
]


@pytest.fixture()
def scratch(tmp_path) -> Scratch:
    s = Scratch(str(tmp_path / "scratch"))
    yield s
    s.discard()


# ------------------------------------------------------------------ the allowlist


def test_minted_ids_and_real_technique_names_are_accepted():
    for _ in range(20):
        job = new_job_id()
        assert job_id_safe(job), job
    for name in ("ela", "copy_move", "jpeg_ghost", "block_grid", "cnn", "metadata", "c2pa",
                 "frequency", "noise", "face"):
        assert path_safe(new_job_id(), name)


@pytest.mark.parametrize("job_id", MALFORMED_IDS)
def test_a_job_id_outside_the_minted_shape_is_refused(job_id):
    assert not job_id_safe(job_id)
    assert not path_safe(job_id, "ela")


@pytest.mark.parametrize("name", MALFORMED_NAMES)
def test_a_technique_name_outside_the_identifier_shape_is_refused(name):
    assert not path_safe(new_job_id(), name)


def test_overlay_path_resolves_only_inside_the_root(scratch):
    job = new_job_id()
    resolved = scratch.overlay_path(job, "ela")
    assert resolved is not None
    assert resolved.parent == scratch.root / job
    assert resolved.name == "ela.png"


@pytest.mark.parametrize("job_id,name", [
    ("..", "ela"),
    (new_job_id(), "..\\..\\app\\config"),
    (new_job_id(), "../secrets"),
    ("", "ela"),
    (new_job_id(), ""),
])
def test_overlay_path_returns_none_for_any_unaddressable_pair(scratch, job_id, name):
    """The route turns this into a 404. It must never become a path.

    ``None`` rather than an exception, because a difference in status code between "no such file"
    and "you are not allowed to ask" is information.
    """
    assert scratch.overlay_path(job_id, name) is None


def test_a_stored_rel_path_cannot_climb_out_of_the_tree(scratch):
    job = new_job_id()
    assert scratch.abs_overlay_url_path(job, f"{job}/ela.png") is not None
    assert scratch.abs_overlay_url_path(job, f"{job}/../../app/config.py") is None
    assert scratch.abs_overlay_url_path(job, "C:/Windows/win.ini") is None


def test_ensure_job_dir_refuses_to_create_a_malformed_one(scratch):
    with pytest.raises(ValueError):
        scratch.ensure_job_dir("../outside")
    assert not (scratch.root.parent / "outside").exists()


def test_remove_job_refuses_to_delete_a_malformed_target(scratch, caplog):
    victim = scratch.root.parent / "keep-me"
    victim.mkdir()
    (victim / "important.txt").write_text("still here", encoding="utf-8")
    with caplog.at_level("WARNING", logger="truelens.scratch"):
        scratch.remove_job("../keep-me")
    assert victim.is_dir(), "a malformed id must never reach rmtree"
    assert any("malformed job id" in r.message for r in caplog.records)


def test_remove_job_deletes_its_own_directory(scratch):
    job = new_job_id()
    (scratch.ensure_job_dir(job) / "ela.png").write_bytes(b"x")
    scratch.remove_job(job)
    assert not scratch.job_dir(job).exists()


# ----------------------------------------------------------------------- sweeping


def test_sweep_removes_only_overaged_directories(scratch):
    old, middle, young = new_job_id(), new_job_id(), new_job_id()
    for job in (old, middle, young):
        (scratch.ensure_job_dir(job) / "ela.png").write_bytes(b"x")
    now = time.time()
    os.utime(scratch.job_dir(old), (now - 5_000, now - 5_000))
    os.utime(scratch.job_dir(middle), (now - 200, now - 200))
    assert scratch.sweep_expired(600, now=now) == 1
    assert not scratch.job_dir(old).exists()
    assert scratch.job_dir(middle).is_dir()
    assert scratch.job_dir(young).is_dir()


def test_sweep_ignores_stray_files_at_the_top_level(scratch):
    (scratch.root / "loose.png").write_bytes(b"x")
    assert scratch.sweep_expired(0, now=time.time() + 10_000) == 0
    assert (scratch.root / "loose.png").is_file()


def test_sweep_on_a_missing_root_is_quiet(tmp_path):
    """The sweeper runs on a timer; a root deleted by an operator must not kill the task."""
    scratch = Scratch(str(tmp_path / "never-created"))
    assert scratch.root.is_dir()
    scratch.discard()
    assert scratch.sweep_expired(1) == 0


def test_a_scratch_root_can_be_reused(tmp_path):
    first = Scratch(str(tmp_path / "shared"))
    job = new_job_id()
    first.ensure_job_dir(job)
    # A second Scratch over the same root is what the pool workers effectively are: same tree, and
    # ``discard`` on one of them must not be the thing that surprises the other.
    second = Scratch(str(tmp_path / "shared"))
    assert second.job_dir(job).is_dir()


# ------------------------------------------------------------------- symlink escape


def test_a_symlinked_job_directory_is_not_followed(scratch):
    """The allowlist cannot see this one, which is why ``_under`` exists."""
    outside = scratch.root.parent / "someone-elses-overlay"
    outside.mkdir()
    (outside / "ela.png").write_bytes(b"private")
    link = scratch.root / new_job_id()
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not let an unprivileged process create symlinks")
    if not link.is_symlink():
        pytest.skip("symlinks created but not honoured on this filesystem")
    assert scratch.overlay_path(link.name, "ela") is None


# ---------------------------------------------------------------- writing overlays


def _rgb(height: int, width: int) -> np.ndarray:
    """A gradient with one pure-red pixel, so a colour-space mistake is visible."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[..., 1] = 30
    image[0, 0] = (255, 0, 0)
    return image


def test_write_overlay_downscales_and_reports_dimensions(scratch):
    job = new_job_id()
    out = scratch.overlay_path(job, "ela")
    written = write_overlay(_rgb(900, 1200), out, max_dim=600)
    assert (written.width, written.height) == (600, 450)
    assert written.rel_path == f"{job}/ela.png"
    assert Path(scratch.abs_overlay_url_path(job, written.rel_path)).is_file()
    assert written.bytes_written == out.stat().st_size
    assert written.bytes_written > 0


def test_write_overlay_leaves_a_small_image_alone(scratch):
    out = scratch.overlay_path(new_job_id(), "noise")
    written = write_overlay(_rgb(37, 41), out, max_dim=2048)
    assert (written.width, written.height) == (41, 37)


def test_write_overlay_converts_rgb_to_bgr_once(scratch):
    """The contract is RGB in; PNG is BGR. If both sides did it, red faces come back cyan."""
    import cv2

    out = scratch.overlay_path(new_job_id(), "face")
    write_overlay(_rgb(8, 8), out, max_dim=0)
    read_back = cv2.imread(str(out), cv2.IMREAD_COLOR)  # cv2 gives BGR
    assert read_back[0, 0].tolist() == [0, 0, 255], "the source pixel was pure red"


def test_write_overlay_accepts_a_float_heatmap(scratch):
    out = scratch.overlay_path(new_job_id(), "frequency")
    heat = _rgb(16, 16).astype(np.float32) * 1.5
    written = write_overlay(heat, out, max_dim=0)
    assert written.bytes_written > 0


def test_write_overlay_creates_the_job_directory_itself(scratch):
    job = new_job_id()
    assert not scratch.job_dir(job).exists()
    write_overlay(_rgb(8, 8), scratch.overlay_path(job, "ela"), max_dim=0)
    assert scratch.job_dir(job).is_dir()
