"""Tests for scripts/diagnose_torch_faiss.py -- the diagnostic-only tool
for the Windows torch/faiss DLL failure.

These tests do NOT depend on Windows, real torch/faiss DLL behavior, or
the real dataset: they exercise the script's own harness logic (test
classification, DLL-listing plumbing, model-name resolution) using tiny,
fast, OS-agnostic subprocess snippets (e.g. `sys.exit(0)` / `sys.exit(1)`)
rather than actually importing torch or faiss. Whether torch/faiss/pandas
actually conflict on any given machine is exactly what the script is for
-- these tests only prove the script correctly reports what a subprocess
did, which is testable everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import diagnose_torch_faiss as diag  # noqa: E402


# --- _configured_model ----------------------------------------------------------


def test_configured_model_reads_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "some/custom-model")
    assert diag._configured_model() == "some/custom-model"


def test_configured_model_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(diag, "PROJECT_ROOT", tmp_path)  # no .env here
    assert diag._configured_model() == diag.DEFAULT_MODEL


def test_configured_model_reads_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    (tmp_path / ".env").write_text("EMBEDDING_MODEL=from/dotenv\n", encoding="utf-8")
    monkeypatch.setattr(diag, "PROJECT_ROOT", tmp_path)
    assert diag._configured_model() == "from/dotenv"


# --- build_import_tests ----------------------------------------------------------


def test_build_import_tests_covers_required_ids():
    tests = diag.build_import_tests()
    ids = {t.id for t in tests}
    required = {"TEST 1", "TEST 2", "TEST 3", "TEST 4", "TEST 5 / A", "TEST 6 / B", "C", "D", "X1", "X2", "X3", "X4"}
    assert required <= ids


def test_build_import_tests_ids_are_unique():
    tests = diag.build_import_tests()
    ids = [t.id for t in tests]
    assert len(ids) == len(set(ids))


# --- build_narrowing_tests (round-2 A-K matrix) -----------------------------------


def test_build_narrowing_tests_covers_all_required_letters():
    tests = diag.build_narrowing_tests()
    ids = {t.id for t in tests}
    assert ids == set("ABCDEFGHIJK")


def test_build_narrowing_tests_every_test_imports_torch():
    # Every letter test must actually import torch somewhere -- it's the
    # thing that fails on the real machine, and the one invariant every
    # test in this matrix shares.
    for t in diag.build_narrowing_tests():
        assert "import torch" in t.code


def test_build_narrowing_tests_letters_isolate_expected_packages():
    tests = {t.id: t.code for t in diag.build_narrowing_tests()}
    assert "numpy" in tests["A"] and "pandas" not in tests["A"] and "pyarrow" not in tests["A"]
    assert "pyarrow" in tests["B"] and "pandas" not in tests["B"] and "numpy" not in tests["B"]
    assert "pandas" in tests["C"] and "pyarrow" not in tests["C"] and "numpy" not in tests["C"]
    assert "pandas" in tests["D"] and "pyarrow" in tests["D"]
    assert "numpy" in tests["E"] and "pyarrow" in tests["E"]
    assert "pandas" in tests["F"] and "numpy" in tests["F"]
    assert "pyarrow" in tests["G"] and "numpy" in tests["G"]
    assert "pandas" in tests["H"] and "numpy" in tests["H"] and "pyarrow" in tests["H"]
    for letter in ("I", "J", "K"):
        assert "read_parquet" in tests[letter]


def test_build_narrowing_tests_no_id_collision_with_round_one():
    round_one_ids = {t.id for t in diag.build_import_tests()}
    round_two_ids = {t.id for t in diag.build_narrowing_tests()}
    # Round-one "C" and "D" (faiss/sentence_transformers combos) are short
    # bare letters too -- confirm they're a DIFFERENT test (different code)
    # from round-two "C"/"D", so a reader can't confuse which is which by
    # id alone; both sets existing is fine (documented in the script's
    # module docstring) as long as their *codes* clearly differ.
    round_one_by_id = {t.id: t.code for t in diag.build_import_tests()}
    round_two_by_id = {t.id: t.code for t in diag.build_narrowing_tests()}
    shared = round_one_ids & round_two_ids
    for id_ in shared:
        assert round_one_by_id[id_] != round_two_by_id[id_]


# --- run_isolated: PASS / FAIL / ERROR classification -----------------------------


def test_run_isolated_pass_on_successful_code():
    test = diag.ImportTest("T", "trivial success", "print('ok')")
    result = diag.run_isolated(test)
    assert result.status == diag.PASS
    assert result.detail == "ok"


def test_run_isolated_fail_on_nonzero_exit():
    test = diag.ImportTest("T", "trivial failure", "import sys; sys.exit(1)")
    result = diag.run_isolated(test)
    assert result.status == diag.FAIL


def test_run_isolated_fail_on_raised_exception():
    test = diag.ImportTest("T", "raises", "raise RuntimeError('boom')")
    result = diag.run_isolated(test)
    assert result.status == diag.FAIL
    assert "boom" in result.detail


def test_run_isolated_error_on_timeout():
    test = diag.ImportTest("T", "hangs", "import time; time.sleep(5)", timeout=0.2)
    result = diag.run_isolated(test)
    assert result.status == diag.ERROR
    assert "TIMEOUT" in result.detail


def test_run_isolated_captures_full_traceback_not_truncated():
    # A traceback with well more than 8 lines (the old truncation limit
    # was `stderr.splitlines()[-8:]`) -- every frame, including the
    # EARLIEST one (f_outermost), must survive into `detail`, since the
    # whole point of this round is inspecting the complete failure, not
    # just whatever fit in a short tail.
    depth = 15
    lines = [f"def f0(): raise RuntimeError('deep failure marker')"]
    for i in range(1, depth):
        lines.append(f"def f{i}(): f{i - 1}()")
    lines.append(f"f{depth - 1}()")
    code = "\n".join(lines)

    test = diag.ImportTest("T", "deep traceback", code)
    result = diag.run_isolated(test)
    assert result.status == diag.FAIL
    assert "deep failure marker" in result.detail
    assert f"f{depth - 1}" in result.detail  # outermost frame (would be cut by an 8-line tail)
    assert "f0" in result.detail  # innermost frame
    assert len(result.detail.splitlines()) > 8  # proves it wasn't cut to a short tail


# --- find_package_dir / list_dlls -------------------------------------------------


def test_find_package_dir_finds_installed_package():
    path = diag.find_package_dir("pytest")
    assert path is not None
    assert path.exists()


def test_find_package_dir_returns_none_for_missing_package():
    assert diag.find_package_dir("this_package_does_not_exist_xyz") is None


def test_list_dlls_returns_empty_for_missing_directory(tmp_path):
    assert diag.list_dlls(tmp_path / "nonexistent") == []


def test_list_dlls_finds_dll_files(tmp_path):
    (tmp_path / "fake.dll").write_bytes(b"")
    (tmp_path / "not_a_dll.txt").write_bytes(b"")
    assert diag.list_dlls(tmp_path) == ["fake.dll"]


# --- inspect_dlls: must never raise, must report a consistent shape -------------------


def test_inspect_dlls_returns_expected_keys():
    info = diag.inspect_dlls()
    assert set(info.keys()) == {
        "platform",
        "package_dll_lists",
        "overlapping_dll_filenames_between_packages",
        "c10_dll_path",
        "c10_dll_exists",
        "vcruntime_status_system32",
    }
    assert set(info["package_dll_lists"].keys()) == {"torch", "faiss", "numpy", "pandas", "pyarrow"}


def test_inspect_dlls_c10_exists_is_none_off_windows():
    info = diag.inspect_dlls()
    if info["platform"] != "nt":
        assert info["c10_dll_exists"] is None
        assert info["vcruntime_status_system32"] == {}


# --- Result / print_result --------------------------------------------------------


def test_print_result_does_not_raise(capsys):
    r = diag.Result("T", "desc", diag.PASS, "detail text")
    diag.print_result(r)
    captured = capsys.readouterr()
    assert "[PASS] T: desc" in captured.out
    assert "detail text" in captured.out


def test_run_project_embedder_test_returns_result_object():
    # Will most likely FAIL in this environment (no network / no Windows
    # DLL issue to reproduce), but must return a well-formed Result, not
    # raise -- that's what's actually being tested here.
    result = diag.run_project_embedder_test("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(result, diag.Result)
    assert result.status in (diag.PASS, diag.FAIL)


def test_run_parquet_smoke_test_skips_when_file_missing(monkeypatch, tmp_path):
    import types

    fake_cfg = types.SimpleNamespace(
        PROCESSED_DIR=tmp_path,
        CONFIG=types.SimpleNamespace(brand=types.SimpleNamespace(author_id="AmazonHelp")),
    )
    monkeypatch.setitem(sys.modules, "src.config", fake_cfg)
    result = diag.run_parquet_smoke_test("sentence-transformers/all-MiniLM-L6-v2")
    assert result.status == diag.SKIP
    assert "not found" in result.detail
