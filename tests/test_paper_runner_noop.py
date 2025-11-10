import sys
import types


def test_import_run_paper_module():
    # Ensure the module file loads; we won't execute network calls in tests
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_paper", "run_paper.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert isinstance(mod, types.ModuleType)

