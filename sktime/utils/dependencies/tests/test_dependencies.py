# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for dependency checking utilities."""

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_for_class(_check_soft_dependencies),
    reason="run test incrementally (if requested)",
)
def test_check_soft_dependencies():
    """Test check_soft_dependencies."""
    ALWAYS_INSTALLED = "sktime"
    ALWAYS_INSTALLED2 = "numpy"
    ALWAYS_INSTALLED_W_V = "sktime>=0.5.0"
    ALWAYS_INSTALLED_W_V2 = "numpy>=0.1.0"
    NEVER_INSTALLED = "nonexistent__package_foo_bar"
    NEVER_INSTALLED_W_V = "sktime<0.1.0"

    # Test that the function does not raise an error when all dependencies are installed
    _check_soft_dependencies(ALWAYS_INSTALLED)
    _check_soft_dependencies(ALWAYS_INSTALLED, ALWAYS_INSTALLED2)
    _check_soft_dependencies(ALWAYS_INSTALLED_W_V)
    _check_soft_dependencies(ALWAYS_INSTALLED_W_V, ALWAYS_INSTALLED_W_V2)
    _check_soft_dependencies(ALWAYS_INSTALLED, ALWAYS_INSTALLED2, ALWAYS_INSTALLED_W_V2)
    _check_soft_dependencies([ALWAYS_INSTALLED, ALWAYS_INSTALLED2])

    # Test that error is raised when a dependency is not installed
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(NEVER_INSTALLED)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(NEVER_INSTALLED, ALWAYS_INSTALLED)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([ALWAYS_INSTALLED, NEVER_INSTALLED])
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(ALWAYS_INSTALLED, NEVER_INSTALLED_W_V)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([ALWAYS_INSTALLED, NEVER_INSTALLED_W_V])

    # disjunction cases, "or" - positive cases
    _check_soft_dependencies([[ALWAYS_INSTALLED, NEVER_INSTALLED]])
    _check_soft_dependencies(
        [
            [ALWAYS_INSTALLED, NEVER_INSTALLED],
            [ALWAYS_INSTALLED_W_V, NEVER_INSTALLED_W_V],
            ALWAYS_INSTALLED2,
        ]
    )

    # disjunction cases, "or" - negative cases
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([[NEVER_INSTALLED, NEVER_INSTALLED_W_V]])
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                [NEVER_INSTALLED, NEVER_INSTALLED_W_V],
                [ALWAYS_INSTALLED, NEVER_INSTALLED],
                ALWAYS_INSTALLED2,
            ]
        )
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                ALWAYS_INSTALLED2,
                [ALWAYS_INSTALLED, NEVER_INSTALLED],
                NEVER_INSTALLED_W_V,
            ]
        )
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                [ALWAYS_INSTALLED, ALWAYS_INSTALLED2],
                NEVER_INSTALLED,
                ALWAYS_INSTALLED2,
            ]
        )


@pytest.mark.skipif(
    not run_test_for_class(_check_soft_dependencies),
    reason="run test incrementally (if requested)",
)
def test_check_soft_dependencies_pep503_normalization(monkeypatch):
    """Test that package name comparison is normalized according to PEP 503.

    Regression test for the bug where distributions whose metadata name uses
    underscores, e.g., ``huggingface_hub``, were not found when checked with
    the hyphenated spelling, e.g., ``huggingface-hub``.
    """
    from sktime.utils.dependencies import _dependencies
    from sktime.utils.dependencies._dependencies import (
        _get_installed_packages,
        _normalize_pkg_name,
    )

    # _normalize_pkg_name implements PEP 503 normalization
    assert _normalize_pkg_name("huggingface_hub") == "huggingface-hub"
    assert _normalize_pkg_name("HuggingFace-Hub") == "huggingface-hub"
    assert _normalize_pkg_name("scikit.learn") == "scikit-learn"

    # keys of the installed packages dict are PEP 503 normalized
    pkgs = _get_installed_packages()
    assert all(name == _normalize_pkg_name(name) for name in pkgs)

    # a distribution registered with underscores in its metadata name, e.g.,
    # huggingface_hub, must be found under the hyphenated spelling and vice versa
    def _mock_installed_packages():
        return {"huggingface-hub": "0.36.2"}

    monkeypatch.setattr(
        _dependencies, "_get_installed_packages", _mock_installed_packages
    )
    assert _check_soft_dependencies("huggingface-hub", severity="none")
    assert _check_soft_dependencies("huggingface_hub", severity="none")
    assert _check_soft_dependencies("HuggingFace_Hub>=0.20", severity="none")
    assert not _check_soft_dependencies("huggingface-hub>1.0", severity="none")
    monkeypatch.undo()

    # spellings of the same installed hard dependency must agree, scikit-learn
    # is a hard dependency of sktime and therefore always present
    assert _check_soft_dependencies("scikit-learn", severity="none")
    assert _check_soft_dependencies("scikit_learn", severity="none")
    assert _check_soft_dependencies("Scikit-Learn", severity="none")
