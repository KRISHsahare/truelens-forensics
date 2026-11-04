"""Every dataclass default in this codebase has to satisfy CPython 3.11's rule.

`pyproject.toml` promises `requires-python = ">=3.11"` and the container image is built on
`python:3.11-slim`, but the development venv here is 3.12 — and this line imported fine locally
while killing the deployed service at import:

    kwargs: MappingProxyType[str, Arg] = MappingProxyType({})   # registry.TechniqueSpec

    ValueError: mutable default <class 'mappingproxy'> for field kwargs is not allowed

The dataclasses predicate did not change between the two versions; the *type* did. `mappingproxy`
is unhashable in 3.11 and gained `__hash__` in 3.12, and dataclasses uses unhashability as its proxy
for "mutable". So the identical class body is legal on the machine that wrote it and fatal on the
machine that runs it, which is why no test caught it: every test ran on the interpreter that
stopped complaining.

Hence the second clause below. Checking hashability alone would be a guard that silently does
nothing on 3.12+, so `MappingProxyType` is named explicitly — it is the one type known to differ,
and the point of this test is to fail on whichever interpreter it is executed on.

The definitive check remains running the suite on 3.11 itself:
`UV_PROJECT_ENVIRONMENT=.venv311 UV_NATIVE_TLS=1 uv sync --frozen --extra dev --python 3.11`.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import sys
import types
from pathlib import Path

import pytest

import app

ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

#: Packages whose dataclasses end up inside the 3.11 container.
_PACKAGES = ("app", "truelens_ml")

#: Defaults of these types are rejected by 3.11's dataclasses even where the running interpreter's
#: own hashability test has stopped objecting.
_VERSION_SPLIT_TYPES: tuple[type, ...] = (types.MappingProxyType,)


def _is_rejected_by_311(value: object) -> bool:
    return value.__class__.__hash__ is None or isinstance(value, _VERSION_SPLIT_TYPES)


def _importable(package_name: str):
    package = importlib.import_module(package_name)
    yield package
    for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        try:
            yield importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - optional extras are allowed to be absent
            # An import failure is its own bug, but not this test's subject: the optional `clip`
            # extra is genuinely not installed in most environments, and pretending otherwise would
            # make this suite depend on a 900 MB download.
            if "open_clip" not in str(exc) and "No module named 'torch'" not in str(exc):
                raise


def _dataclasses_in(module) -> list[type]:
    out = []
    for name in dir(module):
        obj = getattr(module, name, None)
        if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                and obj.__module__.startswith(_PACKAGES)):
            out.append(obj)
    return out


def test_no_dataclass_default_would_be_rejected_by_python_311() -> None:
    offenders: list[str] = []
    seen: set[str] = set()
    checked = 0
    for package_name in _PACKAGES:
        for module in _importable(package_name):
            for cls in _dataclasses_in(module):
                ident = f"{cls.__module__}.{cls.__qualname__}"
                if ident in seen:
                    continue
                seen.add(ident)
                for field in dataclasses.fields(cls):
                    if field.default is dataclasses.MISSING:
                        continue
                    checked += 1
                    if _is_rejected_by_311(field.default):
                        offenders.append(
                            f"{cls.__module__}.{cls.__qualname__}.{field.name} = "
                            f"{field.default!r} ({field.default.__class__.__name__})")

    assert offenders == [], (
        "dataclass default(s) that CPython 3.11 rejects at import — and 3.11 is the interpreter the "
        "Docker image runs. Use dataclasses.field(default_factory=...):\n  "
        + "\n  ".join(offenders))
    # A guard that inspected nothing is worse than no guard: it reports success forever. The
    # assertion is on the class that actually caused the outage, not on a count, because a count is
    # a number somebody guessed and every new dataclass silently erodes its meaning.
    assert "app.analysis.registry.TechniqueSpec" in seen, (
        f"the walk never reached TechniqueSpec ({len(seen)} dataclasses seen); it cannot guard what "
        "it does not look at")
    assert checked > 20, f"only {checked} defaults inspected"


@pytest.mark.parametrize("name", ["app.main", "app.analysis.registry", "app.analysis.features"])
def test_the_startup_import_chain_is_clean(name: str) -> None:
    """The exact import that failed in the container, asserted by name rather than implied.

    Render's log stopped at `app/main.py:36 -> registry.py:95`. If a future default breaks the chain
    again, this test says which module instead of a deploy that says only "Exited with status 1
    while running your code".
    """
    assert importlib.import_module(name) is not None
