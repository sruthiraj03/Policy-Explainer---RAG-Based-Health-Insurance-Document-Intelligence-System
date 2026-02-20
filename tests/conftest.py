"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def project_root() -> str:
    """Path to PolicyExplainer project root for tests."""
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
