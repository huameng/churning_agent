"""
Profile tests: the path-keyed load cache and update_profile's field handling.
update_profile is pointed at a temp YAML so the real config is never touched.
"""

import textwrap

import pytest
import yaml

from churning_agent.tools import profile


_BASE_YAML = textwrap.dedent("""
    personal:
      state: "FL"
      zip_code: "33613"
    banking:
      existing_accounts:
        - "chase"
      existing_credit_cards: []
      preferences:
        min_profit_threshold: 100
        willing_to_visit_branch: true
        excluded_banks: []
        additional_context: []
""")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "user_profile.yaml"
    path.write_text(_BASE_YAML, encoding="utf-8")
    monkeypatch.setattr(profile, "_CONFIG_PATH", path)
    profile._reset_cache()
    yield path
    profile._reset_cache()


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_load_profile_reads_fields(cfg):
    p = profile.load_profile(cfg)
    assert p.state == "FL"
    assert "chase" in p.existing_accounts
    assert p.preferences.min_profit_threshold == 100


def test_load_profile_caches_per_path(tmp_path, cfg):
    """A second path must not return the first path's cached profile (#2)."""
    other = tmp_path / "other.yaml"
    other.write_text(_BASE_YAML.replace('state: "FL"', 'state: "NY"'), encoding="utf-8")
    assert profile.load_profile(cfg).state == "FL"
    assert profile.load_profile(other).state == "NY"   # not the stale FL cache


def test_update_adds_list_value_and_dedupes(cfg):
    assert "Added" in profile.update_profile("existing_accounts", "Citi")
    assert "citi" in _read(cfg)["banking"]["existing_accounts"]      # normalized
    assert "already present" in profile.update_profile("existing_accounts", "citi")


def test_update_bool_field_validates(cfg):
    assert "Set willing_to_visit_branch to False" in profile.update_profile("willing_to_visit_branch", "false")
    assert _read(cfg)["banking"]["preferences"]["willing_to_visit_branch"] is False
    assert "must be 'true' or 'false'" in profile.update_profile("willing_to_visit_branch", "maybe")


def test_update_numeric_field_validates(cfg):
    assert "Set min_profit_threshold to 50.0" in profile.update_profile("min_profit_threshold", "50")
    assert _read(cfg)["banking"]["preferences"]["min_profit_threshold"] == 50.0
    assert "must be a number" in profile.update_profile("min_profit_threshold", "lots")


def test_update_unknown_field_rejected(cfg):
    assert "unknown field" in profile.update_profile("favorite_color", "blue")


def test_update_refreshes_cache(cfg):
    profile.load_profile(cfg)                       # prime the cache
    profile.update_profile("excluded_banks", "wells fargo")
    assert "wells fargo" in profile.load_profile(cfg).preferences.excluded_banks
