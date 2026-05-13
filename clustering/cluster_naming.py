"""
cluster_naming.py

Heuristic cluster-name generator. TF-IDF + n-gram extraction.

Used by Phase 2.1 to produce a short human-readable label for each cluster
(e.g., "authentication & session security: OAuth, session, validation").
The label is for audit logs and dashboard cluster-browser only — the
canonical competency title is produced by Phase 2.2's LLM.

The Padovano & Cardamone (2024) curriculum-design paper uses TF-IDF +
most-frequent bigrams/trigrams to name education-region clusters; we
borrow that approach for cluster labeling here.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


# Tokens that almost always appear in skill/knowledge phrases but carry
# little semantic load for naming purposes. Lowercased.
_STOPWORDS_EXTRA = {
    "skill",
    "skills",
    "knowledge",
    "ability",
    "abilities",
    "experience",
    "experienced",
    "familiarity",
    "familiar",
    "using",
    "use",
    "used",
    "work",
    "working",
    "strong",
    "good",
    "excellent",
    "proficient",
    "proficiency",
    "understanding",
    "area",
}


def _tokenize_for_naming(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Keeps acronyms intact (e.g. "AWS", "CI/CD" -> "ci/cd"). Doesn't lemmatize —
    we want labels to keep the form they appear in the corpus.
    """
    text = text.lower()
    # Replace punctuation (except `/`, `-`, `+` which are common in tech terms) with space
    text = re.sub(r"[^\w/\-\+\s]", " ", text)
    return [t for t in text.split() if t]


def _build_label_from_terms(unigrams: Sequence[str], bigrams: Sequence[str], trigrams: Sequence[str]) -> str:
    """Compose the human-readable label.

    Format:
        "<top trigram or bigram>: <unigram1>, <unigram2>"

    If no n-grams of length >=2 survive filtering, fall back to top unigrams.
    """
    head = ""
    if trigrams:
        head = trigrams[0]
    elif bigrams:
        head = bigrams[0]

    # Build the unigram tail. Strip any token already present in `head`.
    head_tokens = set(_tokenize_for_naming(head))
    tail_terms = [u for u in unigrams if u not in head_tokens][:2]
    tail = ", ".join(tail_terms)

    if head and tail:
        return f"{head}: {tail}"
    if head:
        return head
    if tail:
        return tail
    return "(unlabeled cluster)"


def _extract_top_ngrams(
    cluster_texts: Sequence[str],
    background_texts: Sequence[str],
    ngram_range: Tuple[int, int],
    top_k: int,
) -> List[str]:
    """Return the top-k n-grams that distinguish the cluster from the background.

    Implementation: build a TF-IDF vectorizer fit on cluster_texts (the IDF
    component captures how distinctive a term is *within* this cluster's
    documents). Then re-score using term frequencies in cluster_texts relative
    to combined corpus document frequency to bias toward distinctive terms.

    Simpler approach used here: vectorize the joined cluster text against the
    full background corpus and pick the highest-scoring n-grams.
    """
    # Need at least 2 documents for TF-IDF idf to make sense. If cluster has
    # only one item or all items are duplicates, fall back to plain top-k by frequency.
    docs = list(background_texts) + [" ".join(cluster_texts)]
    cluster_idx = len(docs) - 1

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r"(?u)\b[\w/\-\+]+\b",
            stop_words="english",
            min_df=1,
            max_df=1.0,
        )
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        # Empty vocabulary after stop-word + min_df filtering. Bail out.
        return []

    feature_names = vectorizer.get_feature_names_out()
    cluster_row = tfidf[cluster_idx].toarray().ravel()
    if cluster_row.sum() == 0:
        return []

    # Filter terms containing only stopwords-extra
    def _is_keepable(term: str) -> bool:
        tokens = term.split()
        # For n-grams, require at least one non-stop token
        if all(tok in _STOPWORDS_EXTRA for tok in tokens):
            return False
        # Drop very short terms (single letters, digits)
        if len(term) <= 1:
            return False
        return True

    scored = [
        (feature_names[i], cluster_row[i])
        for i in range(len(feature_names))
        if cluster_row[i] > 0 and _is_keepable(feature_names[i])
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _score in scored[:top_k]]


def label_cluster(
    cluster_texts: Sequence[str],
    background_texts: Sequence[str],
    max_unigrams: int = 5,
    max_bigrams: int = 3,
    max_trigrams: int = 2,
) -> Tuple[str, List[str]]:
    """Produce (summary_label, top_terms) for a single cluster.

    Args:
        cluster_texts: item texts belonging to this cluster.
        background_texts: item texts across the WHOLE corpus (all clusters).
            Used as TF-IDF background to surface distinctive terms.
        max_unigrams, max_bigrams, max_trigrams: how many of each to extract.

    Returns:
        (summary_label, top_terms)
            summary_label: short human-readable string
            top_terms: combined top-k list (unigrams ++ bigrams ++ trigrams),
                       useful for cluster-browser UI tooltips
    """
    if not cluster_texts:
        return "(empty cluster)", []

    # Single-item cluster: just use the item text itself as the label.
    if len(cluster_texts) == 1:
        text = cluster_texts[0].strip()
        return text[:80], [text]

    unigrams = _extract_top_ngrams(cluster_texts, background_texts, (1, 1), max_unigrams)
    bigrams = _extract_top_ngrams(cluster_texts, background_texts, (2, 2), max_bigrams)
    trigrams = _extract_top_ngrams(cluster_texts, background_texts, (3, 3), max_trigrams)

    label = _build_label_from_terms(unigrams, bigrams, trigrams)
    top_terms = list(unigrams) + list(bigrams) + list(trigrams)

    # Fallback: TF-IDF produced nothing usable — use the most common item text.
    if label == "(unlabeled cluster)" or not top_terms:
        # Use the shortest item text as a heuristic label (often the canonical form).
        shortest = min(cluster_texts, key=len)
        label = shortest[:80]
        top_terms = [shortest]

    return label, top_terms
