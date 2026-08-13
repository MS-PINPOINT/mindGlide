"""Packaging contract tests: importability, version, python -m support."""
import subprocess
import sys


def test_regular_package_with_version():
    import mindglide

    # A regular package (not an accidental namespace package) with a version.
    assert mindglide.__file__ is not None
    assert mindglide.__version__ not in (None, "", "unknown")


def test_python_dash_m_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "mindglide", "--version"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "mindglide" in result.stdout


def test_console_script_matches_package_version():
    import mindglide

    result = subprocess.run(
        [sys.executable, "-m", "mindglide", "--version"],
        capture_output=True, text=True, timeout=120,
    )
    assert mindglide.__version__ in result.stdout


def test_importing_mindglide_does_not_silence_warnings():
    """The CLI quiets warnings inside main(); importing the library must not."""
    code = (
        "import warnings, mindglide.infer\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    warnings.warn('probe')\n"
        "assert len(caught) == 1, 'library import must not install warning filters'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
