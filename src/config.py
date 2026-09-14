"""Central, typed configuration for the whole project.

Every tunable constant used by more than one module lives here so it is not
scattered across scripts. Values are sourced from environment variables
(loaded from a local `.env` file if present) with documented defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present. Never overrides variables
# already set in the real environment (e.g. CI secrets).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CACHE_DIR = OUTPUTS_DIR / "cache"
CONFIG_DIR = PROJECT_ROOT / "config"

RAW_CSV_PATH = Path(_env_str("RAW_CSV_PATH", str(RAW_DIR / "twcs.csv")))


@dataclass(frozen=True)
class DataConfig:
    """Dataset processing configuration."""

    csv_path: Path = RAW_CSV_PATH
    chunksize: int = _env_int("DATA_CHUNKSIZE", 50_000)
    # Optional cap on rows read, for fast local development. None = full file.
    sample_size: int | None = (
        _env_int("DATA_SAMPLE_SIZE", 0) or None
    )
    random_seed: int = _env_int("RANDOM_SEED", 42)
    encoding: str = "utf-8"


@dataclass(frozen=True)
class BrandConfig:
    """The single brand this project targets."""

    name: str = _env_str("BRAND_NAME", "AmazonHelp")
    # Support-account author_id in the dataset that represents the brand's
    # outbound (agent) side of the conversation.
    author_id: str = _env_str("BRAND_AUTHOR_ID", "AmazonHelp")


@dataclass(frozen=True)
class SplitConfig:
    """Train / retrieval-corpus / golden(held-out) split configuration.

    Splitting is done at the THREAD level (conversation_id), never at the
    individual tweet level, to avoid leaking a conversation's resolution
    into both the retrieval corpus and the evaluation set.
    """

    train_frac: float = _env_float("SPLIT_TRAIN_FRAC", 0.80)
    golden_frac: float = _env_float("SPLIT_GOLDEN_FRAC", 0.20)
    random_seed: int = _env_int("RANDOM_SEED", 42)


@dataclass(frozen=True)
class GoldenSetConfig:
    target_size: int = _env_int("GOLDEN_SET_SIZE", 200)
    min_size: int = 150
    max_size: int = 250
    random_seed: int = _env_int("RANDOM_SEED", 42)


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = _env_str(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = _env_int("RETRIEVAL_TOP_K", 5)
    index_path: Path = CACHE_DIR / "faiss_index" / "amazonhelp.index"
    metadata_path: Path = CACHE_DIR / "faiss_index" / "amazonhelp_meta.parquet"
    batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 256)


@dataclass(frozen=True)
class LLMConfig:
    provider: str = _env_str("LLM_PROVIDER", "nvidia")
    api_key: str | None = os.environ.get("NVIDIA_API_KEY") or None
    base_url: str = _env_str(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    model: str = _env_str("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    judge_model: str = _env_str("NVIDIA_JUDGE_MODEL", _env_str("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"))
    timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 30.0)
    max_retries: int = _env_int("LLM_MAX_RETRIES", 3)
    temperature: float = _env_float("LLM_TEMPERATURE", 0.2)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 400)

    def is_configured(self) -> bool:
        placeholder_values = {"", "YOUR_NVIDIA_API_KEY_HERE"}
        return bool(self.api_key) and self.api_key not in placeholder_values


@dataclass(frozen=True)
class EscalationConfig:
    """Thresholds for the AUTO_HANDLE vs ESCALATE_TO_HUMAN decision.

    All thresholds are configurable so the policy can be tuned per the
    evaluation results without touching agent code.
    """

    min_intent_confidence: float = _env_float("ESCALATION_MIN_INTENT_CONFIDENCE", 0.55)
    min_evidence_similarity: float = _env_float("ESCALATION_MIN_EVIDENCE_SIMILARITY", 0.35)
    min_evidence_count: int = _env_int("ESCALATION_MIN_EVIDENCE_COUNT", 2)
    # Intents that always require a human regardless of confidence/evidence.
    # Must match intent names in config/intents.yaml (see each intent's
    # `always_escalate` field, which is the source of truth this list
    # mirrors for fast lookup without re-parsing YAML on every request).
    always_escalate_intents: tuple[str, ...] = (
        "account_access_issue",
        "payment_or_billing_issue",
        "general_complaint_or_feedback",
        "other_unclear",
    )


@dataclass(frozen=True)
class EvaluationConfig:
    random_seed: int = _env_int("RANDOM_SEED", 42)
    judge_human_review_sample_size: int = _env_int("JUDGE_HUMAN_REVIEW_SAMPLE_SIZE", 40)
    retrieval_recall_k_values: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    golden: GoldenSetConfig = field(default_factory=GoldenSetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


CONFIG = AppConfig()


def ensure_output_dirs() -> None:
    """Create all output directories the pipeline writes to, if missing."""
    for d in (PROCESSED_DIR, GOLDEN_DIR, ANALYSIS_DIR, METRICS_DIR, FIGURES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
