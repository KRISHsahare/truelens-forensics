"""Settings behaviour that only matters in production, asserted in both directions.

These are the configuration rules whose failure mode is a service that *starts*. `main.py` calls
`validate_prod()` and refuses to boot on a configuration that cannot work, which is only worth
anything if the values it inspects are the values the process would actually have used — a
convenience default applied a few lines earlier can satisfy a guard that was written to catch it.

That is exactly what happened on the deployed API: with `ENVIRONMENT=prod` and no
`OVERLAY_SIGNING_KEY`, `get_settings()` substituted the development constant, `validate_prod()`
found a non-empty key, reported `problems: []`, and the service signed every overlay URL with a
value published in this repository. The health endpoint looked healthy. The guard looked like it
had run.
"""

from __future__ import annotations

import pytest

from app import config


@pytest.fixture
def fresh_settings(monkeypatch):
    """Call `get_settings()` with a clean cache and a controlled environment.

    Keys the test does not name are *removed* rather than left alone: the ambient environment of a
    developer's shell or a CI job can carry `ENVIRONMENT` or `OVERLAY_SIGNING_KEY`, and a settings
    test that silently inherits them proves nothing about the defaults it claims to check.
    """
    managed = ("ENVIRONMENT", "OVERLAY_SIGNING_KEY", "WEB_CONCURRENCY", "STORE_BACKEND")

    def build(**env: str):
        for key in managed:
            if key in env:
                monkeypatch.setenv(key, env[key])
            else:
                monkeypatch.delenv(key, raising=False)
        config.reset_settings_cache()
        try:
            return config.get_settings()
        finally:
            config.reset_settings_cache()

    return build


def test_prod_refuses_the_development_signing_key(fresh_settings) -> None:
    s = fresh_settings(ENVIRONMENT="prod", OVERLAY_SIGNING_KEY="")

    # The fallback must not fire in prod, or the check below can never see an empty key.
    assert s.overlay_signing_key == "", (
        "get_settings() substituted the dev constant in prod; validate_prod() then passes on a key "
        "that is published in this repository")
    problems = s.validate_prod()
    assert any("OVERLAY_SIGNING_KEY" in p for p in problems), problems
    assert config._DEV_OVERLAY_KEY not in s.overlay_signing_key


def test_dev_still_gets_a_usable_key(fresh_settings) -> None:
    s = fresh_settings(ENVIRONMENT="dev", OVERLAY_SIGNING_KEY="")

    # The other direction: local development must not need a secret to run at all, and the guard
    # that makes prod fatal has to stay silent here.
    assert s.overlay_signing_key == config._DEV_OVERLAY_KEY
    assert not any("OVERLAY_SIGNING_KEY" in p for p in s.validate_prod())


def test_a_configured_key_is_used_verbatim_in_prod(fresh_settings) -> None:
    s = fresh_settings(ENVIRONMENT="prod", OVERLAY_SIGNING_KEY="a-real-deployed-secret-value")
    assert s.overlay_signing_key == "a-real-deployed-secret-value"
    assert s.validate_prod() == []


def test_two_workers_on_the_memory_store_is_fatal(fresh_settings) -> None:
    """The sibling rule, because it fails the same way: intermittently, on one worker only."""
    s = fresh_settings(ENVIRONMENT="prod", OVERLAY_SIGNING_KEY="x" * 40,
                      WEB_CONCURRENCY="2", STORE_BACKEND="memory")
    assert any("WEB_CONCURRENCY" in p for p in s.validate_prod())
