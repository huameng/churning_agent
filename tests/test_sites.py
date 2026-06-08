from churning_agent.tools import sites


def test_whitelisted_domain_allowed():
    assert sites.is_allowed("https://www.topcashback.com/onlinecashback/")
    assert sites.is_allowed("https://topcashback.com/some/path")


def test_unknown_domain_blocked():
    assert not sites.is_allowed("https://www.chase.com/login")
    assert not sites.is_allowed("https://evil.example.com")


def test_subdomain_not_silently_allowed():
    # Only exact hostnames in allowed_domains count; a lookalike host is blocked.
    assert not sites.is_allowed("https://www.topcashback.com.evil.com/")


def test_adapter_for_url():
    adapter = sites.adapter_for_url("https://www.topcashback.com/logon/")
    assert adapter is not None
    assert adapter.name == "topcashback"
    assert sites.adapter_for_url("https://example.com") is None


def test_garbage_url_blocked():
    assert not sites.is_allowed("not a url")
    assert not sites.is_allowed("")
