"""Regression tests for the Windows UnicodeEncodeError bug.

`pathlib.Path.write_text()` without an explicit `encoding=` argument uses
`locale.getpreferredencoding()`, which is cp1252 on most Windows setups --
not UTF-8. The Customer Support on Twitter dataset contains legitimate
multilingual/emoji customer text, so any output-writing code path that
doesn't pass `encoding="utf-8"` explicitly can crash (or silently mangle
Unicode) on Windows even though it works fine on Linux/macOS, where the
default encoding is already UTF-8. These tests build a tiny CSV containing
non-Latin1 Unicode (emoji, CJK, accented Latin) and drive the real
`scripts/analyze_dataset.py` pipeline end-to-end, asserting the output
files -- explicitly read back as UTF-8 -- contain those exact characters,
unmodified and unstripped.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UNICODE_SAMPLES = [
    "My café order never arrived \U0001f622",  # emoji + accented Latin
    "配送が遅れています、助けてください",  # Japanese: "delivery is late, please help"
    "Mi pedido llegó dañado, necesito ayuda por favor ñ",  # Spanish + ñ
]

_TWITTER_DATE = "Mon Jan 02 10:00:00 +0000 2017"


def _write_unicode_fixture_csv(path: Path) -> None:
    rows = [
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id"
    ]
    tid = 1000
    for i, text in enumerate(UNICODE_SAMPLES):
        cust_id, brand_id = tid, tid + 1
        rows.append(f'{cust_id},cust{i},True,{_TWITTER_DATE},"{text}",{brand_id},')
        rows.append(
            f'{brand_id},AmazonHelp,False,{_TWITTER_DATE},"Thanks for reaching out, we are on it ❤",,{cust_id}'
        )
        tid += 10
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def unicode_csv(tmp_path) -> Path:
    csv_path = tmp_path / "unicode_twcs.csv"
    _write_unicode_fixture_csv(csv_path)
    return csv_path


def test_analyze_dataset_preserves_unicode_end_to_end(tmp_path, unicode_csv, monkeypatch):
    """Drives the exact code path that crashed on Windows
    (scripts/analyze_dataset.py's candidate_brands.md write) and confirms
    every output file round-trips the non-ASCII customer/brand text
    byte-for-byte as UTF-8.
    """
    import scripts.analyze_dataset as analyze_mod

    importlib.reload(analyze_mod)  # ensure a clean module state between test runs
    monkeypatch.setattr(analyze_mod, "MIN_OUTBOUND_FOR_CANDIDATE", 1)
    monkeypatch.setattr(analyze_mod, "TOP_N_WITH_EXAMPLES", 1)
    monkeypatch.setattr(analyze_mod, "N_EXAMPLES_PER_BRAND", 3)

    out_dir = tmp_path / "analysis_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dataset.py",
            "--csv-path",
            str(unicode_csv),
            "--out-dir",
            str(out_dir),
            "--chunksize",
            "5",
            "--focus-brand",
            "AmazonHelp",
        ],
    )

    analyze_mod.main()  # this is the exact call that raised UnicodeEncodeError on Windows

    md_path = out_dir / "candidate_brands.md"
    summary_path = out_dir / "dataset_summary.json"
    assert md_path.exists()
    assert summary_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    for sample in UNICODE_SAMPLES:
        assert sample in md_text, f"Unicode customer text was mangled or lost: {sample!r}"
    assert "❤" in md_text  # the brand reply's heart emoji

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_texts = [row["text"] for row in summary["schema"]["sample_rows"]]
    assert any(UNICODE_SAMPLES[0] in (t or "") for t in sample_texts)

    # Prove the bytes on disk are valid UTF-8 (not cp1252 mojibake) by
    # decoding raw bytes directly, independent of Path.read_text's own
    # platform-default behavior.
    raw_bytes = md_path.read_bytes()
    decoded = raw_bytes.decode("utf-8")  # raises UnicodeDecodeError if not real UTF-8
    assert UNICODE_SAMPLES[1] in decoded


def test_write_text_without_encoding_would_have_failed_on_cp1252():
    """Sanity-checks the premise of the bug report: encoding the emoji
    sample as cp1252 (Windows' typical default) is impossible, which is
    exactly the UnicodeEncodeError the user hit. This documents *why*
    `encoding="utf-8"` is required, rather than merely asserting the fix
    exists.
    """
    with pytest.raises(UnicodeEncodeError):
        UNICODE_SAMPLES[0].encode("cp1252")
