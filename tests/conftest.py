"""Keep GRID_SPACING_MM from leaking between tests.

coarse.use_spacing() rewrites GRID_SPACING_MM/GRID_SPACING_M across a dozen
modules. Several test files call it -- test_optimization_loop.py does so at
*import* time, which pytest runs during collection, so its 2.0 mm leaks into
every test in the session including ones collected earlier. That silently
rebuilt masks at the wrong resolution and turned 13 tests red only in a
full-suite run; each passed in isolation.

conftest is imported before any test module, so the snapshot below is the
pristine value. Restoring it after every test makes spacing test-local.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geometry_contract  # noqa: E402

_PRISTINE_MM = geometry_contract.GRID_SPACING_MM


@pytest.fixture(autouse=True)
def _restore_grid_spacing():
    yield
    _reset()


def pytest_collectstart(collector):
    """Reset the spacing before each test module is imported.

    Several test files do `from geometry_contract import GRID_SPACING_M` at
    module level, which freezes whatever value collection order happened to
    leave behind -- test_unified_phi sets 2.0 mm at import, so every file
    alphabetically after it froze 2.0 while its tests actually ran at 0.3.
    Resetting first means a file freezes either the pristine default or the
    spacing it sets for itself on the next line, never another file's.
    """
    if isinstance(collector, pytest.Module):
        _reset()


def pytest_collection_finish(session):
    """Undo spacing changes made at test-module import time."""
    _reset()


def _reset():
    if geometry_contract.GRID_SPACING_MM != _PRISTINE_MM:
        from sandbox import coarse
        coarse.use_spacing(_PRISTINE_MM)
