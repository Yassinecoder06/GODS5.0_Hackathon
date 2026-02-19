import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_page_artifacts: bool = True
    normalize_esg_aliases: bool = True


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"\s+")


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_page_artifacts(text: str) -> str:
    text = text.replace("\u2022", " ")
    text = text.replace("|", " ")
    text = re.sub(r"[\-_=]{3,}", " ", text)
    text = re.sub(r"\bpage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    return text


def _normalize_esg_aliases(text: str) -> str:
    replacements = {
        r"\be\.s\.g\b": "esg",
        r"\benvironmental\s*,?\s*social\s*(and|&)\s*governance\b": "esg",
        r"\benvironmental\s+social\s+governance\b": "esg",
    }
    out = text
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def clean_text(text: str, cfg: PreprocessConfig | None = None) -> str:
    if cfg is None:
        cfg = PreprocessConfig()

    if text is None:
        text = ""
    text = str(text)

    if cfg.normalize_unicode:
        text = _normalize_unicode(text)

    if cfg.remove_urls:
        text = URL_PATTERN.sub(" ", text)

    if cfg.remove_emails:
        text = EMAIL_PATTERN.sub(" ", text)

    if cfg.remove_page_artifacts:
        text = _remove_page_artifacts(text)

    if cfg.normalize_esg_aliases:
        text = _normalize_esg_aliases(text)

    if cfg.lowercase:
        text = text.lower()

    if cfg.collapse_whitespace:
        text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text


def build_ablation_configs() -> Dict[str, PreprocessConfig]:
    return {
        "raw": PreprocessConfig(
            lowercase=False,
            normalize_unicode=False,
            collapse_whitespace=False,
            remove_urls=False,
            remove_emails=False,
            remove_page_artifacts=False,
            normalize_esg_aliases=False,
        ),
        "basic_clean": PreprocessConfig(
            lowercase=True,
            normalize_unicode=True,
            collapse_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            remove_page_artifacts=False,
            normalize_esg_aliases=False,
        ),
        "esg_normalized": PreprocessConfig(
            lowercase=True,
            normalize_unicode=True,
            collapse_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            remove_page_artifacts=True,
            normalize_esg_aliases=True,
        ),
    }


def apply_pipeline(texts: List[str], fn: Callable[[str], str]) -> List[str]:
    return [fn(t) for t in texts]
