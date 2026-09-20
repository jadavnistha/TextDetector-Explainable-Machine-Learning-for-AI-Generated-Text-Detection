"""
features.py — Extract linguistic features from text using spaCy and textdescriptives.

Each text is converted to a fixed-length numeric vector covering readability,
descriptive statistics, POS-tag ratios, syntax depth, lexical diversity,
and punctuation rates.  These features feed the downstream classifiers.
"""

import numpy as np
import spacy
import textdescriptives as td
import warnings

warnings.filterwarnings("ignore")

# ── Load spaCy model once ──────────────────────────────────────────────
# textdescriptives hooks into the pipeline automatically via add_pipe
nlp = spacy.load("en_core_web_sm")

# Add textdescriptives components (readability, descriptive_stats, etc.)
# Only add if not already present (avoids errors on reimport)
for component in ["textdescriptives/readability",
                   "textdescriptives/descriptive_stats"]:
    if component not in nlp.pipe_names:
        nlp.add_pipe(component)

# ── Feature name list (fixed order) ────────────────────────────────────
FEATURE_NAMES = [
    # Readability (5)
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    "coleman_liau", "automated_readability_index",
    # Descriptive stats (5)
    "mean_sentence_length", "median_sentence_length", "std_sentence_length",
    "mean_token_length", "std_token_length",
    # Counts (2)
    "n_tokens", "n_sentences",
    # POS-tag ratios (9)
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio",
    "adp_ratio", "det_ratio", "cconj_ratio", "punct_ratio",
    # Syntax (3)
    "mean_dep_distance", "std_dep_distance", "mean_parse_tree_depth",
    # Lexical diversity (3)
    "type_token_ratio", "unique_word_ratio", "hapax_legomenon_ratio",
    # Punctuation rates (5)
    "comma_rate", "period_rate", "question_mark_rate",
    "exclamation_rate", "semicolon_rate",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 32


def _safe(value, default=0.0):
    """Return default if value is None, NaN, or inf."""
    if value is None:
        return default
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def extract_features(text: str) -> np.ndarray:
    """
    Convert a single text string into a fixed-length numpy feature vector.

    Returns a 1-D array of shape (NUM_FEATURES,).
    If the text is empty or too short, returns a zero vector.
    """
    # Handle empty / very short text
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return np.zeros(NUM_FEATURES, dtype=np.float64)

    # Truncate very long texts to keep processing fast on CPU
    text = text[:10_000]

    try:
        doc = nlp(text)
    except Exception:
        return np.zeros(NUM_FEATURES, dtype=np.float64)

    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    sentences = list(doc.sents)
    n_sentences = len(sentences)

    if n_tokens == 0:
        return np.zeros(NUM_FEATURES, dtype=np.float64)

    # ── 1. Readability (from textdescriptives) ─────────────────────────
    rd = doc._.readability
    flesch_reading_ease       = _safe(rd.get("flesch_reading_ease"))
    flesch_kincaid_grade      = _safe(rd.get("flesch_kincaid_grade"))
    gunning_fog               = _safe(rd.get("gunning_fog"))
    coleman_liau              = _safe(rd.get("coleman_liau_index"))
    automated_readability_idx = _safe(rd.get("automated_readability_index"))

    # ── 2. Descriptive statistics ──────────────────────────────────────
    sent_lens = [len([t for t in s if not t.is_space]) for s in sentences]
    mean_sent_len   = float(np.mean(sent_lens))   if sent_lens else 0.0
    median_sent_len = float(np.median(sent_lens))  if sent_lens else 0.0
    std_sent_len    = float(np.std(sent_lens))     if len(sent_lens) > 1 else 0.0

    tok_lens = [len(t.text) for t in tokens if not t.is_punct]
    mean_tok_len = float(np.mean(tok_lens)) if tok_lens else 0.0
    std_tok_len  = float(np.std(tok_lens))  if len(tok_lens) > 1 else 0.0

    # ── 3. POS-tag ratios ──────────────────────────────────────────────
    pos_counts = {}
    for t in tokens:
        pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1

    pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "CCONJ", "PUNCT"]
    pos_ratios = [pos_counts.get(tag, 0) / n_tokens for tag in pos_tags]

    # ── 4. Syntax features ─────────────────────────────────────────────
    # Dependency distance: |token_index - head_index|
    dep_distances = [abs(t.i - t.head.i) for t in tokens if t.dep_ != "ROOT"]
    mean_dep_dist = float(np.mean(dep_distances)) if dep_distances else 0.0
    std_dep_dist  = float(np.std(dep_distances))  if len(dep_distances) > 1 else 0.0

    # Parse tree depth per sentence
    def _tree_depth(token, depth=0):
        children = list(token.children)
        if not children:
            return depth
        return max(_tree_depth(c, depth + 1) for c in children)

    tree_depths = []
    for sent in sentences:
        root = [t for t in sent if t.dep_ == "ROOT"]
        if root:
            tree_depths.append(_tree_depth(root[0]))
    mean_tree_depth = float(np.mean(tree_depths)) if tree_depths else 0.0

    # ── 5. Lexical diversity ───────────────────────────────────────────
    words = [t.text.lower() for t in tokens if t.is_alpha]
    n_words = len(words)

    if n_words > 0:
        unique_words = set(words)
        n_unique = len(unique_words)
        type_token_ratio  = n_unique / n_words
        unique_word_ratio = n_unique / n_words  # same formula, kept for clarity

        # Hapax legomenon ratio: words appearing exactly once / total words
        from collections import Counter
        word_freq = Counter(words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / n_words
    else:
        type_token_ratio = 0.0
        unique_word_ratio = 0.0
        hapax_ratio = 0.0

    # ── 6. Punctuation rates (per token) ───────────────────────────────
    raw_text = text
    comma_rate      = raw_text.count(",")  / n_tokens
    period_rate     = raw_text.count(".")  / n_tokens
    question_rate   = raw_text.count("?")  / n_tokens
    exclamation_rate = raw_text.count("!") / n_tokens
    semicolon_rate  = raw_text.count(";")  / n_tokens

    # ── Assemble the feature vector ────────────────────────────────────
    features = np.array([
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog,
        coleman_liau, automated_readability_idx,
        mean_sent_len, median_sent_len, std_sent_len,
        mean_tok_len, std_tok_len,
        n_tokens, n_sentences,
        *pos_ratios,
        mean_dep_dist, std_dep_dist, mean_tree_depth,
        type_token_ratio, unique_word_ratio, hapax_ratio,
        comma_rate, period_rate, question_rate,
        exclamation_rate, semicolon_rate,
    ], dtype=np.float64)

    # Final safety: replace any remaining NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features_batch(texts: list) -> np.ndarray:
    """
    Extract features for a list of texts.  Prints progress every 500 rows.
    Returns a 2-D array of shape (len(texts), NUM_FEATURES).
    """
    n = len(texts)
    result = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    for i, text in enumerate(texts):
        result[i] = extract_features(text)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  Extracted features: {i + 1}/{n}")

    return result
