#!/usr/bin/env python3
"""Diagnostic-only script for the Windows `OSError: [WinError 1114] ...
c10.dll` failure seen in `scripts/build_index.py`. NOT part of the
production pipeline -- never imported by src/ or other scripts.

Context: the initial hypothesis (faiss's bundled MKL/OpenMP runtime
conflicting with torch's, triggered by faiss loading first) was
implemented as an import-order fix in Retriever.build()/.load(), but the
failure recurred with the exact same traceback even though, with that
fix in place, torch is the FIRST native extension entering the process at
the point of failure (faiss hasn't been imported yet). That result does
NOT confirm the original hypothesis -- it argues against it being
*sufficient* on its own.

Round 2 (real Windows results): `import faiss; import torch` and
`import torch; import faiss` BOTH pass -- faiss is not implicated.
`import torch; <then> pandas+pyarrow` passes, but `import pandas;
import torch`, `import pandas; import pyarrow; <then> sentence_
transformers`, and `pandas.read_parquet(...); <then> sentence_
transformers` all FAIL with the same WinError 1114. pandas and/or
pyarrow (tested only together so far) are implicated; numpy alone is
untested. PART A2 below (tests A-K) isolates numpy, pandas, and pyarrow
individually and in every pairwise/triple combination before `import
torch`, to identify the minimal trigger from evidence rather than guess.
This script exists to gather that evidence before any further code
change is made.

Every import-order test runs in its own SEPARATE subprocess (not just a
separate function call in this process), because the entire point is to
isolate which combination of already-loaded native libraries is
necessary to reproduce the failure -- running tests in a shared process
would contaminate every later test with whatever an earlier one already
loaded.

Each test reports exactly one of:
    PASS  - the test's code ran and completed successfully
    FAIL  - the test's code ran and raised/exited non-zero (a real signal)
    ERROR - the test harness itself couldn't get a clean result (e.g. a
            subprocess timeout) -- inconclusive, not necessarily a failure
            of the thing being tested
    SKIP  - deliberately not run (e.g. an opt-in flag wasn't passed, a
            required file doesn't exist, or a check is Windows-only and
            this isn't Windows)

Usage (Windows, from the activated .venv):
    python scripts\\diagnose_torch_faiss.py
    python scripts\\diagnose_torch_faiss.py --model-load-test    (adds TEST 7: loads the real model; downloads it if not cached)
    python scripts\\diagnose_torch_faiss.py --parquet-smoke-test (adds a real 1-10 row encode against data/processed/*.parquet)

Usage (any OS): same, with scripts/diagnose_torch_faiss.py.

Nothing here modifies files, installs/uninstalls packages, or touches the
retrieval index. It only imports things, in controlled combinations, and
reports what happened.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PASS, FAIL, ERROR, SKIP = "PASS", "FAIL", "ERROR", "SKIP"


@dataclass
class Result:
    id: str
    description: str
    status: str  # one of PASS, FAIL, ERROR, SKIP
    detail: str = ""


def _configured_model() -> str:
    """Reads EMBEDDING_MODEL from the real environment or .env, WITHOUT
    importing src.config into this (parent) diagnostic process -- keeps
    this process itself free of any heavy import, so it can't accidentally
    become "process state" that affects the subprocess tests below."""
    model = os.environ.get("EMBEDDING_MODEL")
    if model:
        return model
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EMBEDDING_MODEL="):
                return line.split("=", 1)[1].strip()
    return DEFAULT_MODEL


# --------------------------------------------------------------------------
# PART A: subprocess-isolated import-order tests
# --------------------------------------------------------------------------


@dataclass
class ImportTest:
    id: str
    description: str
    code: str
    timeout: float = 60.0


def build_import_tests() -> list[ImportTest]:
    return [
        ImportTest("TEST 1", "import torch", "import torch; print(torch.__version__)"),
        ImportTest("TEST 2", "import sentence_transformers", "import sentence_transformers; print('ST import OK')"),
        ImportTest(
            "TEST 3",
            "from sentence_transformers import SentenceTransformer",
            "from sentence_transformers import SentenceTransformer; print('ST class import OK')",
        ),
        ImportTest("TEST 4", "import faiss", "import faiss; print(faiss.__version__)"),
        ImportTest(
            "TEST 5 / A",
            "import faiss; import torch  (faiss THEN torch)",
            "import faiss; import torch; print('FAISS -> TORCH OK')",
        ),
        ImportTest(
            "TEST 6 / B",
            "import torch; import faiss  (torch THEN faiss)",
            "import torch; import faiss; print('TORCH -> FAISS OK')",
        ),
        ImportTest(
            "C",
            "sentence_transformers alone, faiss never imported",
            "from sentence_transformers import SentenceTransformer; print('ST alone OK, no faiss')",
        ),
        ImportTest(
            "D",
            "import faiss; from sentence_transformers import SentenceTransformer",
            "import faiss; from sentence_transformers import SentenceTransformer; print('FAISS -> ST OK')",
        ),
        # --- Extra tests beyond the torch/faiss axis: build_index.py's REAL
        # import chain loads pandas (and pyarrow, via pd.read_parquet) BEFORE
        # sentence-transformers/torch ever get touched. Since the traceback
        # shows torch failing even when faiss hasn't been imported yet at
        # all, pandas/pyarrow being the "other native library already in the
        # process" is at least as plausible as faiss was, and untested until
        # now. ---
        ImportTest(
            "X1",
            "import pandas; import torch  (mirrors build_index.py's real order)",
            "import pandas; import torch; print('PANDAS -> TORCH OK')",
        ),
        ImportTest(
            "X2",
            "import pandas; import pyarrow; from sentence_transformers import SentenceTransformer",
            "import pandas as pd; import pyarrow; from sentence_transformers import SentenceTransformer; "
            "print('PANDAS+PYARROW -> ST OK')",
        ),
        ImportTest(
            "X3",
            "import torch FIRST, then pandas+pyarrow (reversed order control)",
            "import torch; import pandas as pd; import pyarrow; print('TORCH -> PANDAS+PYARROW OK')",
        ),
        ImportTest(
            "X4",
            "pandas.read_parquet a tiny real file, THEN sentence_transformers "
            "(closest simulation of build_index.py's actual sequence, minus faiss)",
            (
                "import pandas as pd; "
                "df = pd.DataFrame({'a':[1,2,3]}); "
                "import tempfile, os as _os; "
                "p = tempfile.mktemp(suffix='.parquet'); "
                "df.to_parquet(p); "
                "pd.read_parquet(p); "
                "_os.remove(p); "
                "from sentence_transformers import SentenceTransformer; "
                "print('PANDAS(read_parquet)+PYARROW -> ST OK')"
            ),
        ),
    ]


def build_narrowing_tests() -> list[ImportTest]:
    """Round 2, added after real Windows evidence showed:
      - PASS: import faiss; import torch  AND  import torch; import faiss
        (faiss is not the trigger either order)
      - PASS: import torch; then pandas + pyarrow (torch first is fine)
      - FAIL: import pandas; import torch
      - FAIL: import pandas; import pyarrow; from sentence_transformers import SentenceTransformer
      - FAIL: pandas.read_parquet(tiny_file); then sentence_transformers
      - FAIL: the project's own SentenceTransformerEmbedder + Retriever construction

    That evidence implicates "something pandas and/or pyarrow load"
    conflicting with torch's later init when pandas/pyarrow go first --
    but pandas and pyarrow were always tested together, so which one (or
    whether it's numpy underneath either of them) is still unknown. This
    matrix isolates each of numpy/pandas/pyarrow individually and in every
    pairwise/triple combination, always ending in `import torch` last, so
    the single result that changes from PASS to FAIL identifies the
    minimal trigger.
    """
    parquet_prelude = (
        "import pandas as pd, tempfile, os as _os; "
        "df = pd.DataFrame({'a':[1,2,3]}); "
        "p = tempfile.mktemp(suffix='.parquet'); "
        "df.to_parquet(p); "
        "pd.read_parquet(p); "
        "_os.remove(p); "
    )
    return [
        ImportTest("A", "import numpy; import torch", "import numpy; import torch; print('A OK')"),
        ImportTest("B", "import pyarrow; import torch", "import pyarrow; import torch; print('B OK')"),
        ImportTest("C", "import pandas; import torch", "import pandas; import torch; print('C OK')"),
        ImportTest(
            "D", "import pandas; import pyarrow; import torch",
            "import pandas; import pyarrow; import torch; print('D OK')",
        ),
        ImportTest(
            "E", "import numpy; import pyarrow; import torch",
            "import numpy; import pyarrow; import torch; print('E OK')",
        ),
        ImportTest(
            "F", "import pandas; import numpy; import torch",
            "import pandas; import numpy; import torch; print('F OK')",
        ),
        ImportTest(
            "G", "import pyarrow; import numpy; import torch",
            "import pyarrow; import numpy; import torch; print('G OK')",
        ),
        ImportTest(
            "H", "import pandas; import numpy; import pyarrow; import torch",
            "import pandas; import numpy; import pyarrow; import torch; print('H OK')",
        ),
        ImportTest(
            "I", "pandas.read_parquet(tiny_file); import torch",
            parquet_prelude + "import torch; print('I OK')",
        ),
        ImportTest(
            "J", "import numpy; pandas.read_parquet(tiny_file); import torch",
            "import numpy; " + parquet_prelude + "import torch; print('J OK')",
        ),
        ImportTest(
            "K", "import pyarrow; pandas.read_parquet(tiny_file); import torch",
            "import pyarrow; " + parquet_prelude + "import torch; print('K OK')",
        ),
    ]


def run_isolated(test: ImportTest) -> Result:
    """Runs `test.code` in a brand-new `python -c` subprocess (a clean
    process with nothing pre-loaded except the interpreter itself).

    PASS: exit code 0.
    FAIL: the subprocess ran and exited non-zero (a real import/runtime
          error -- the exact signal this script exists to capture).
    ERROR: the subprocess could not be timed/collected cleanly (timeout,
           or the harness failed to even launch it) -- inconclusive.
    """
    env = os.environ.copy()
    # Preserve KMP_DUPLICATE_LIB_OK if the project's config.py normally sets
    # it (see src/config.py) so this test reflects the real, currently
    # configured runtime, not a stripped-down one.
    if os.name == "nt":
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", test.code],
            capture_output=True,
            text=True,
            timeout=test.timeout,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return Result(test.id, test.description, ERROR, f"TIMEOUT after {test.timeout}s: {exc}")
    except OSError as exc:
        return Result(test.id, test.description, ERROR, f"Could not launch subprocess: {exc}")

    if proc.returncode == 0:
        return Result(test.id, test.description, PASS, proc.stdout.strip())
    # Full traceback captured, not truncated -- the whole point of this
    # narrowing round is to see exactly what each failure says.
    full_stderr = proc.stderr.strip() or "(no stderr captured)"
    return Result(test.id, test.description, FAIL, full_stderr)


# --------------------------------------------------------------------------
# PART B: static DLL inspection (no imports that could crash the process)
# --------------------------------------------------------------------------


def find_package_dir(name: str) -> Path | None:
    """Locates a package's install directory WITHOUT importing it (uses
    importlib's module finder only), so this can't itself trigger the
    crash we're diagnosing."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def list_dlls(directory: Path | None, subdir: str = "") -> list[str]:
    if directory is None:
        return []
    target = directory / subdir if subdir else directory
    if not target.exists():
        return []
    return sorted(p.name for p in target.glob("*.dll"))


def inspect_dlls() -> dict:
    """Reports, per package, its bundled .dll filenames (Windows only --
    on other platforms this reports empty lists, which is expected and
    not a failure). Cross-references filenames that appear in MORE THAN
    ONE package's directory: a same-named DLL bundled by two packages
    (e.g. an MKL or OpenMP runtime shipped by both faiss and numpy, or
    numpy and torch) is the concrete, checkable signature of the
    "duplicate/conflicting native runtime" hypothesis -- distinct from
    merely asserting it.
    """
    packages = {
        "torch": find_package_dir("torch"),
        "faiss": find_package_dir("faiss"),
        "numpy": find_package_dir("numpy"),
        "pandas": find_package_dir("pandas"),
        "pyarrow": find_package_dir("pyarrow"),
    }
    dlls: dict[str, list[str]] = {}
    for name, pkg_dir in packages.items():
        if pkg_dir is None:
            dlls[name] = []
            continue
        # torch's DLLs live in torch/lib/, most other packages ship them
        # directly alongside the package's .pyd files or in a `.libs`/
        # `<pkg>.libs` sibling directory depending on how the wheel was
        # built -- check the common locations rather than assuming one.
        found: set[str] = set()
        found.update(list_dlls(pkg_dir, "lib"))
        found.update(list_dlls(pkg_dir))
        found.update(list_dlls(pkg_dir.parent, f"{name}.libs"))
        dlls[name] = sorted(found)

    overlaps: dict[str, list[str]] = {}
    for name_a, dlls_a in dlls.items():
        for name_b, dlls_b in dlls.items():
            if name_a >= name_b:
                continue
            shared = sorted(set(dlls_a) & set(dlls_b))
            if shared:
                overlaps[f"{name_a} & {name_b}"] = shared

    torch_dir = find_package_dir("torch")
    c10_path = (torch_dir / "lib" / "c10.dll") if torch_dir is not None else None

    vcruntime_status: dict[str, bool] = {}
    if os.name == "nt":
        system32 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"
        for dll_name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
            vcruntime_status[dll_name] = (system32 / dll_name).exists()

    return {
        "platform": os.name,
        "package_dll_lists": dlls,
        "overlapping_dll_filenames_between_packages": overlaps,
        "c10_dll_path": str(c10_path) if c10_path else None,
        "c10_dll_exists": (c10_path.exists() if (c10_path is not None and os.name == "nt") else None),
        "vcruntime_status_system32": vcruntime_status,
    }


# --------------------------------------------------------------------------
# PART C: the project's own embedder wrapper (in-process, run last so any
# crash here doesn't prevent the isolated subprocess tests from completing)
# --------------------------------------------------------------------------


def run_project_embedder_test(configured_model: str) -> Result:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import numpy as np

        from src.retrieval.embeddings import SentenceTransformerEmbedder
        from src.retrieval.retriever import Retriever

        embedder = SentenceTransformerEmbedder(configured_model)
        retriever = Retriever(embedder)  # construct only -- do NOT call .build()
        assert retriever.index is None
        vectors = embedder.embed(["I cannot access my Amazon account", "My package has not arrived"])

        assert vectors.shape[0] == 2, f"expected 2 rows, got {vectors.shape}"
        assert vectors.dtype == np.float32, f"expected float32, got {vectors.dtype}"
        assert np.isfinite(vectors).all(), "embeddings contain non-finite values"
        detail = f"OK -- shape={vectors.shape}, dtype={vectors.dtype}, dim={embedder.dimension}"
        return Result("TEST 8/9", "project embedder + retriever construction", PASS, detail)
    except Exception as exc:  # noqa: BLE001 -- diagnostic script, report everything
        return Result(
            "TEST 8/9", "project embedder + retriever construction", FAIL, f"{type(exc).__name__}: {exc}"
        )


# --------------------------------------------------------------------------
# PART D (opt-in): real smoke test against actual processed data
# --------------------------------------------------------------------------


def run_parquet_smoke_test(configured_model: str, n_rows: int = 10) -> Result:
    desc = "real embed of 1-10 actual train_retrieval customer_text rows"
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import numpy as np
        import pandas as pd

        from src import config as cfg
        from src.retrieval.embeddings import SentenceTransformerEmbedder

        pairs_path = cfg.PROCESSED_DIR / f"{cfg.CONFIG.brand.author_id.lower()}_resolution_pairs.parquet"
        if not pairs_path.exists():
            return Result(
                "PARQUET SMOKE TEST", desc, SKIP, f"{pairs_path} not found -- run scripts/preprocess.py first"
            )

        pairs = pd.read_parquet(pairs_path)
        train = pairs[pairs["split"] == "train_retrieval"]
        train = train[train["customer_text"].fillna("").str.strip() != ""]
        sample = train["customer_text"].head(n_rows).tolist()
        if not sample:
            return Result("PARQUET SMOKE TEST", desc, FAIL, "No non-empty train_retrieval customer_text rows found")

        embedder = SentenceTransformerEmbedder(configured_model)
        vectors = embedder.embed(sample)

        assert vectors.shape == (len(sample), embedder.dimension)
        assert vectors.dtype == np.float32
        assert np.isfinite(vectors).all(), "embeddings contain NaN/inf"
        detail = f"OK -- embedded {len(sample)} real rows, shape={vectors.shape}, dtype={vectors.dtype}, all finite"
        return Result("PARQUET SMOKE TEST", desc, PASS, detail)
    except Exception as exc:  # noqa: BLE001
        return Result("PARQUET SMOKE TEST", desc, FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_result(r: Result) -> None:
    print(f"[{r.status}] {r.id}: {r.description}")
    if r.detail:
        lines = r.detail.splitlines() or [r.detail]
        print(f"        -> {lines[0]}")
        for line in lines[1:]:
            print(f"           {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-load-test",
        action="store_true",
        help="Adds TEST 7: loads the real configured SentenceTransformer model "
        "(downloads it if not cached). Off by default since it's slow/network-dependent.",
    )
    parser.add_argument(
        "--parquet-smoke-test",
        action="store_true",
        help="Adds a real 1-10 row smoke test against data/processed/*.parquet "
        "(train_retrieval split only). Off by default; requires preprocess.py to have run.",
    )
    args = parser.parse_args()

    configured_model = _configured_model()
    results: list[Result] = []

    print("=" * 78)
    print("PART A: isolated import-order tests (each in its own subprocess)")
    print("=" * 78)
    for test in build_import_tests():
        r = run_isolated(test)
        print_result(r)
        results.append(r)

    print()
    print("=" * 78)
    print("PART A2: numpy/pandas/pyarrow narrowing matrix (tests A-K)")
    print("=" * 78)
    for test in build_narrowing_tests():
        r = run_isolated(test)
        print_result(r)
        results.append(r)

    if args.model_load_test:
        model_test = ImportTest(
            "TEST 7",
            f"load real model {configured_model!r} (downloads if not cached)",
            (
                "from sentence_transformers import SentenceTransformer; "
                f"m = SentenceTransformer({configured_model!r}); "
                "print('MODEL LOAD OK, dim=', m.get_sentence_embedding_dimension())"
            ),
            timeout=600.0,
        )
        r = run_isolated(model_test)
        print_result(r)
        results.append(r)
    else:
        r = Result("TEST 7", "load real configured model (opt-in via --model-load-test)", SKIP, "not requested")
        print_result(r)
        results.append(r)

    print()
    print("=" * 78)
    print("PART B: static DLL inspection (no imports -- cannot itself crash)")
    print("=" * 78)
    dll_info = inspect_dlls()
    is_windows = dll_info["platform"] == "nt"
    if not is_windows:
        print("Not running on Windows (os.name != 'nt') -- DLL listings below will be")
        print("empty/not applicable here. Run this script on the Windows machine that")
        print("produced the failure to get a meaningful DLL report.")
    for pkg, dlls in dll_info["package_dll_lists"].items():
        print(f"  {pkg}: {len(dlls)} dll(s){' -> ' + ', '.join(dlls[:8]) if dlls else ''}")
    results.append(
        Result(
            "DLL LISTING",
            "enumerate bundled DLLs per package (torch/faiss/numpy/pandas/pyarrow)",
            PASS if is_windows else SKIP,
            f"{sum(len(v) for v in dll_info['package_dll_lists'].values())} dll(s) found"
            if is_windows
            else "not Windows",
        )
    )

    print(f"  c10.dll path: {dll_info['c10_dll_path']}")
    if is_windows:
        c10_status = PASS if dll_info["c10_dll_exists"] else FAIL
    else:
        c10_status = SKIP
    print(f"  c10.dll exists: {dll_info['c10_dll_exists']}")
    results.append(Result("C10.DLL EXISTS", "c10.dll present on disk", c10_status, str(dll_info["c10_dll_exists"])))

    if dll_info["overlapping_dll_filenames_between_packages"]:
        print("  OVERLAPPING DLL FILENAMES (candidate duplicate-runtime conflict):")
        for pair, shared in dll_info["overlapping_dll_filenames_between_packages"].items():
            print(f"    {pair}: {shared}")
        overlap_result = Result(
            "DLL OVERLAP CHECK",
            "same-named DLL bundled by more than one package",
            FAIL,
            str(dll_info["overlapping_dll_filenames_between_packages"]),
        )
    else:
        print("  No overlapping DLL filenames found between the inspected packages.")
        overlap_result = Result(
            "DLL OVERLAP CHECK",
            "same-named DLL bundled by more than one package",
            PASS if is_windows else SKIP,
            "none found" if is_windows else "not Windows",
        )
    results.append(overlap_result)

    print(f"  VC++ runtime DLLs in System32: {dll_info['vcruntime_status_system32'] or '(not Windows)'}")
    if is_windows:
        missing = [k for k, v in dll_info["vcruntime_status_system32"].items() if not v]
        vcruntime_result = Result(
            "VCRUNTIME CHECK",
            "required MSVC runtime DLLs present in System32",
            PASS if not missing else FAIL,
            "all present" if not missing else f"missing: {missing}",
        )
    else:
        vcruntime_result = Result("VCRUNTIME CHECK", "required MSVC runtime DLLs present in System32", SKIP, "not Windows")
    results.append(vcruntime_result)

    print()
    print("=" * 78)
    print("PART C: project's own SentenceTransformerEmbedder + Retriever (construct + 2-sentence embed)")
    print("=" * 78)
    r = run_project_embedder_test(configured_model)
    print_result(r)
    results.append(r)

    print()
    print("=" * 78)
    print("PART D: real smoke test against data/processed/*.parquet (train_retrieval only)")
    print("=" * 78)
    if args.parquet_smoke_test:
        r = run_parquet_smoke_test(configured_model)
    else:
        r = Result("PARQUET SMOKE TEST", "real train_retrieval rows (opt-in via --parquet-smoke-test)", SKIP, "not requested")
    print_result(r)
    results.append(r)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    counts = {PASS: 0, FAIL: 0, ERROR: 0, SKIP: 0}
    for r in results:
        counts[r.status] += 1
        print(f"  [{r.status}] {r.id}: {r.description}")
    print(
        f"\n{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[ERROR]} ERROR, "
        f"{counts[SKIP]} SKIP (of {len(results)} total)."
    )
    failed = [r for r in results if r.status == FAIL]
    if failed:
        print("\nFAILED:")
        for r in failed:
            print(f"  - {r.id}: {r.description}")
            print(f"    {r.detail}")
    print(
        "\nReport this FULL output back verbatim -- especially which of TEST 5/6/C/D, "
        "X1-X4, and the PART A2 narrowing tests A-K are PASS vs FAIL (with their full "
        "tracebacks), plus the PART B DLL overlap / c10.dll / vcruntime results -- so "
        "the next step is chosen from evidence, not another guess."
    )


if __name__ == "__main__":
    main()
