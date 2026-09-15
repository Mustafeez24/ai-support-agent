#!/usr/bin/env python3
"""Phase 1 dataset analysis for the Customer Support on Twitter CSV.

Produces, under outputs/analysis/:
    dataset_summary.json      - schema, row/dup/missing stats, inbound/outbound,
                                 conversation-structure overview, assumptions
    brand_statistics.csv      - per-candidate-brand volume/structure stats + score
    conversation_statistics.csv - conversation-length distribution
    candidate_brands.md       - narrative ranking, recommendation, real examples

Design notes (see DECISIONS.md for the full rationale):
    - The raw CSV's `text` column is never held in memory for the whole
      file. A first streaming pass builds a structural index (ids, author,
      inbound flag, reply pointers, timestamp) for every row; a second,
      much smaller pass pulls `text` only for the handful of candidate
      brands' example messages.
    - "Candidate brand" = any author_id that sends outbound (inbound=False)
      messages, i.e. a support account, above a minimum volume floor.

Usage (Windows / PowerShell or cmd):
    python scripts\\analyze_dataset.py
    python scripts\\analyze_dataset.py --sample-size 200000   (fast dev run)

Usage (macOS/Linux):
    python scripts/analyze_dataset.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src.data import clean, load, threads  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_dataset")

MIN_OUTBOUND_FOR_CANDIDATE = 500
TOP_N_CANDIDATES = 20
TOP_N_WITH_EXAMPLES = 5
N_EXAMPLES_PER_BRAND = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", type=Path, default=cfg.CONFIG.data.csv_path)
    p.add_argument(
        "--sample-size",
        type=int,
        default=cfg.CONFIG.data.sample_size,
        help="Cap rows read from the index pass, for fast local development. "
        "Omit (or 0) to process the full file.",
    )
    p.add_argument("--chunksize", type=int, default=cfg.CONFIG.data.chunksize)
    p.add_argument("--out-dir", type=Path, default=cfg.ANALYSIS_DIR)
    p.add_argument(
        "--focus-brand",
        type=str,
        default=cfg.CONFIG.brand.author_id,
        help="Brand to always include in scored candidates/examples even if "
        "outside the automatic top-N, so it's directly comparable.",
    )
    return p.parse_args()


def compute_missing_text_stats(csv_path: Path, chunksize: int, max_rows: int | None) -> dict:
    """Second lightweight pass reading only the `text` column, to report its
    missingness without holding all text in memory."""
    n_missing = 0
    n_total = 0
    for chunk in pd.read_csv(
        csv_path, usecols=["text"], dtype={"text": "string"}, chunksize=chunksize
    ):
        if max_rows is not None:
            remaining = max_rows - n_total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        n_missing += int(chunk["text"].isna().sum() + (chunk["text"].fillna("").str.strip() == "").sum() - int(chunk["text"].isna().sum()))
        n_total += len(chunk)
        if max_rows is not None and n_total >= max_rows:
            break
    return {
        "column": "text",
        "missing_count": n_missing,
        "missing_pct": round(100 * n_missing / n_total, 4) if n_total else 0.0,
        "rows_scanned": n_total,
    }


def build_candidate_brand_stats(index_df: pd.DataFrame) -> pd.DataFrame:
    """Per-support-account (candidate brand) volume and conversation stats,
    computed entirely from the structural index (no `text` needed)."""
    outbound = index_df[index_df["inbound"] == False]  # noqa: E712
    volume = outbound.groupby("author_id").size().rename("outbound_messages")
    candidates = volume[volume >= MIN_OUTBOUND_FOR_CANDIDATE].sort_values(ascending=False)

    rows = []
    for author_id, outbound_messages in candidates.items():
        conv_ids = set(outbound.loc[outbound["author_id"] == author_id, "conversation_id"])
        brand_conv_df = index_df[index_df["conversation_id"].isin(conv_ids)]
        conv_sizes = brand_conv_df.groupby("conversation_id").size()
        multi_turn = int((conv_sizes >= 3).sum())

        # Direct customer->brand resolution pairs, computed from the index
        # alone: a brand outbound row whose in_response_to_tweet_id points
        # at a row that is itself inbound (a genuine customer message).
        brand_rows = outbound[outbound["author_id"] == author_id]
        by_id = index_df.set_index("tweet_id")
        parent_ids = brand_rows["in_response_to_tweet_id"].dropna().astype(str)
        valid_parents = parent_ids[parent_ids.isin(by_id.index)]
        parent_inbound = by_id.loc[valid_parents.values, "inbound"]
        resolution_pairs = int((parent_inbound == True).sum())  # noqa: E712

        rows.append(
            {
                "brand": author_id,
                "outbound_messages": int(outbound_messages),
                "conversations": len(conv_ids),
                "resolution_pairs": resolution_pairs,
                "multi_turn_conversations": multi_turn,
                "multi_turn_fraction": round(multi_turn / len(conv_ids), 4) if conv_ids else 0.0,
                "avg_conversation_size": round(float(conv_sizes.mean()), 3) if len(conv_sizes) else 0.0,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("outbound_messages", ascending=False)
    return stats_df.head(TOP_N_CANDIDATES).reset_index(drop=True)


def score_candidates(stats_df: pd.DataFrame, golden_target: int) -> pd.DataFrame:
    """Transparent, documented scoring (not a black box):

    - volume_score: log-scaled outbound message count (diminishing returns
      past "plenty"; we don't need the single biggest brand, just "enough").
    - pairs_score: resolution_pairs relative to what's needed for a healthy
      retrieval corpus + a `golden_target`-example held-out set without
      starving either side (need roughly >= 5x the golden target in pairs
      so an 80/20 thread split still leaves a usable retrieval corpus).
    - structure_score: multi_turn_fraction, rewarding brands whose
      conversations have real back-and-forth (more representative of a
      support agent that needs conversational context, not just one-shot
      Q&A).

    Final score = mean of the three sub-scores, each clipped to [0, 1].
    """
    df = stats_df.copy()
    df["volume_score"] = np.clip(np.log10(df["outbound_messages"].clip(lower=1)) / 5.0, 0, 1)
    needed_pairs = golden_target * 5
    df["pairs_score"] = np.clip(df["resolution_pairs"] / needed_pairs, 0, 1)
    df["structure_score"] = np.clip(df["multi_turn_fraction"] / 0.5, 0, 1)
    df["score"] = (df["volume_score"] + df["pairs_score"] + df["structure_score"]) / 3.0
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def collect_examples(
    csv_path: Path,
    index_df: pd.DataFrame,
    brand_author_id: str,
    chunksize: int,
    n_examples: int,
) -> list[dict]:
    """Pull a handful of real (customer_text -> brand_reply_text) examples
    for one brand via a small, targeted second pass over the CSV."""
    outbound = index_df[
        (index_df["inbound"] == False) & (index_df["author_id"] == brand_author_id)  # noqa: E712
    ]
    by_id = index_df.set_index("tweet_id")
    sample_rows = outbound.sample(
        n=min(n_examples * 3, len(outbound)), random_state=cfg.CONFIG.data.random_seed
    )
    wanted_ids: set[str] = set()
    pairs_meta = []
    for _, row in sample_rows.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if parent_id is None or pd.isna(parent_id):
            continue
        parent_id = str(parent_id)
        if parent_id not in by_id.index:
            continue
        parent_inbound = by_id.loc[parent_id, "inbound"]
        if isinstance(parent_inbound, pd.Series):
            parent_inbound = parent_inbound.iloc[0]
        if not bool(parent_inbound):
            continue
        wanted_ids.add(row["tweet_id"])
        wanted_ids.add(parent_id)
        pairs_meta.append((parent_id, row["tweet_id"]))
        if len(pairs_meta) >= n_examples:
            break

    if not wanted_ids:
        return []

    text_by_id: dict[str, str] = {}
    for chunk in load.iter_full_chunks(csv_path, chunksize=chunksize, tweet_ids=wanted_ids):
        for _, r in chunk.iterrows():
            text_by_id[r["tweet_id"]] = clean.clean_text(r["text"])
        if len(text_by_id) >= len(wanted_ids):
            break

    examples = []
    for customer_id, brand_id in pairs_meta:
        if customer_id in text_by_id and brand_id in text_by_id:
            examples.append(
                {
                    "customer_tweet_id": customer_id,
                    "customer_text": text_by_id[customer_id],
                    "brand_tweet_id": brand_id,
                    "brand_text": text_by_id[brand_id],
                }
            )
    return examples


def main() -> None:
    args = parse_args()
    cfg.ensure_output_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_rows = args.sample_size or None

    logger.info("Inspecting schema at %s", args.csv_path)
    schema = load.inspect_schema(args.csv_path)

    logger.info("Pass 1/2: building structural index (chunksize=%d, max_rows=%s)", args.chunksize, max_rows)
    index_df = load.build_full_index(args.csv_path, chunksize=args.chunksize, max_rows=max_rows)
    raw_row_count = len(index_df)
    dup_report = clean.duplicate_report(index_df)
    index_df = clean.drop_duplicate_tweet_ids(index_df)

    missing_report = clean.missing_value_report(index_df)
    text_missing = compute_missing_text_stats(args.csv_path, args.chunksize, max_rows)

    inbound_counts = index_df["inbound"].value_counts(dropna=False).to_dict()

    logger.info("Assigning conversation ids (union-find over reply graph)")
    index_df["conversation_id"] = threads.assign_conversation_ids(index_df)
    conv_sizes = index_df.groupby("conversation_id").size()

    conversation_stats = pd.DataFrame(
        {
            "conversation_size_messages": conv_sizes.values,
        }
    )
    conv_stat_summary = {
        "total_conversations": int(conv_sizes.shape[0]),
        "avg_messages_per_conversation": round(float(conv_sizes.mean()), 3),
        "median_messages_per_conversation": float(conv_sizes.median()),
        "max_messages_per_conversation": int(conv_sizes.max()),
        "multi_turn_conversations_ge3": int((conv_sizes >= 3).sum()),
        "multi_turn_fraction_ge3": round(float((conv_sizes >= 3).mean()), 4),
        "single_message_conversations": int((conv_sizes == 1).sum()),
    }

    logger.info("Scoring candidate brands")
    brand_stats = build_candidate_brand_stats(index_df)
    if args.focus_brand not in set(brand_stats["brand"]):
        focus_stats = build_candidate_brand_stats(
            index_df[
                index_df["author_id"].isin(
                    set(brand_stats["brand"]).union({args.focus_brand})
                )
                | (index_df["author_id"] == args.focus_brand)
            ]
        )
        brand_stats = pd.concat([brand_stats, focus_stats]).drop_duplicates("brand")
    scored = score_candidates(brand_stats, cfg.CONFIG.golden.target_size)

    logger.info("Pass 2/2: collecting real examples for top candidates")
    top_for_examples = list(dict.fromkeys([args.focus_brand] + scored["brand"].head(TOP_N_WITH_EXAMPLES).tolist()))
    examples_by_brand = {}
    for brand in top_for_examples:
        examples_by_brand[brand] = collect_examples(
            args.csv_path, index_df, brand, args.chunksize, N_EXAMPLES_PER_BRAND
        )

    # --- write artifacts ---
    dataset_summary = {
        "schema": schema,
        "row_counts": {
            "raw_rows_read": raw_row_count,
            "unique_tweet_id_rows": len(index_df),
            "sample_capped": max_rows is not None,
            "sample_size_used": max_rows,
        },
        "duplicates": dup_report,
        "missing_values": missing_report.to_dict(orient="records") + [text_missing],
        "inbound_outbound_counts": {str(k): int(v) for k, v in inbound_counts.items()},
        "conversation_structure": conv_stat_summary,
        "candidate_brand_count": int(len(scored)),
        "assumptions": [
            "A 'conversation' is a connected component of the reply graph "
            "formed by in_response_to_tweet_id / response_tweet_id, not a "
            "fixed time window.",
            "A 'candidate brand' is any author_id with >= "
            f"{MIN_OUTBOUND_FOR_CANDIDATE} outbound (inbound=False) messages.",
            "'resolution_pairs' counts only DIRECT reply pairs (brand tweet's "
            "in_response_to_tweet_id points straight at a customer inbound "
            "tweet), not transitive multi-hop pairs within a longer thread.",
            "Duplicate tweet_id rows (if any) are dropped, keeping the first "
            "occurrence, before all downstream statistics.",
            "created_at is parsed with Twitter's standard format "
            "('%a %b %d %H:%M:%S %z %Y'); unparseable timestamps fall back "
            "to pandas' generic parser and are otherwise left NaT.",
        ],
    }
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, default=str), encoding="utf-8"
    )

    scored.to_csv(args.out_dir / "brand_statistics.csv", index=False, encoding="utf-8")
    conversation_stats.to_csv(args.out_dir / "conversation_statistics.csv", index=False, encoding="utf-8")

    md_lines = [
        "# Candidate Brand Analysis",
        "",
        f"Rows scanned: {raw_row_count:,} | Unique tweet_id rows: {len(index_df):,} | "
        f"Total conversations: {conv_stat_summary['total_conversations']:,}",
        "",
        "## Top candidate brands (scored)",
        "",
        "| Rank | Brand | Outbound msgs | Conversations | Resolution pairs | "
        "Multi-turn % | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in scored.head(10).iterrows():
        md_lines.append(
            f"| {i + 1} | {row['brand']} | {row['outbound_messages']:,} | "
            f"{row['conversations']:,} | {row['resolution_pairs']:,} | "
            f"{row['multi_turn_fraction'] * 100:.1f}% | {row['score']:.3f} |"
        )

    md_lines += ["", "## Real examples (customer message -> brand reply)", ""]
    for brand, examples in examples_by_brand.items():
        md_lines.append(f"### {brand}")
        if not examples:
            md_lines.append("_No direct reply-pair examples found in this pass._")
            continue
        for ex in examples:
            md_lines.append(f"- **Customer:** {ex['customer_text']}")
            md_lines.append(f"  **{brand}:** {ex['brand_text']}")
        md_lines.append("")

    md_lines += [
        "## Scoring methodology",
        "",
        "`score = mean(volume_score, pairs_score, structure_score)`, each in [0, 1]:",
        "- `volume_score`: log10(outbound_messages) / 5, clipped — rewards having",
        "  enough data, with diminishing returns past 'plenty'.",
        f"- `pairs_score`: resolution_pairs / (golden_target * 5) = resolution_pairs / "
        f"{cfg.CONFIG.golden.target_size * 5}, clipped — a brand needs materially more",
        "  resolution pairs than the golden-set target so an 80/20 thread-level split",
        "  still leaves a healthy retrieval corpus after the golden set is carved out.",
        "- `structure_score`: multi_turn_fraction / 0.5, clipped — rewards brands whose",
        "  conversations have real back-and-forth rather than one-shot Q&A.",
        "",
        "See dataset_summary.json for full schema/missing/duplicate stats and",
        "DECISIONS.md for the rationale behind these thresholds.",
    ]
    (args.out_dir / "candidate_brands.md").write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Wrote analysis artifacts to %s", args.out_dir)
    logger.info("Top candidate: %s (score=%.3f)", scored.iloc[0]["brand"], scored.iloc[0]["score"])


if __name__ == "__main__":
    main()
