"""Generation-quality metrics: BLEU-4, chrF, ROUGE-L, diversity, repetition.

Spec 2.3 "Generation quality" requires, against reference continuations
from the held-out set: BLEU (corpus-level, n-gram order stated), chrF or
chrF++, ROUGE-L, plus repetition rate, Distinct-1/Distinct-2, and
qualitative notes.

**Tokenization choice.** BLEU and ROUGE are n-gram overlap metrics and so
depend on what counts as a token. Whitespace tokenization is wrong for
these languages: Phase 1 measured 6.8M unique whitespace words in Nepali
against 3.96M in Hindi, because agglutinative case marking fuses
postpositions onto the stem (``नेपालको`` = नेपाल + को). Two continuations
that differ only in case marking would score zero whitespace overlap
despite being nearly identical. Each language's own SentencePiece model
is used instead, so overlap is measured on morpheme-like units.

``sacrebleu`` and ``rouge_score`` are metric libraries, not pretrained
models, so they are permitted under the "no pretrained models" constraint.
Pure-Python fallbacks are implemented for both so the suite still runs if
they are unavailable.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------- n-gram core


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """Corpus-level BLEU-4 with the standard brevity penalty.

    Corpus-level (not sentence-averaged): counts are pooled across the
    whole corpus before the precision ratio is taken, which is the
    standard definition and far more stable than averaging per-sentence
    scores over short generations.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")

    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h = _ngrams(hyp, n)
            r = _ngrams(ref, n)
            # clip each hypothesis n-gram count by its reference count
            clipped[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            totals[n - 1] += max(sum(h.values()), 0)

    precisions = []
    for n in range(max_n):
        precisions.append(clipped[n] / totals[n] if totals[n] > 0 else 0.0)

    # geometric mean; zero at any order collapses BLEU to zero, which is the
    # standard behaviour and common for short open-ended generations
    if min(precisions) > 0:
        log_gm = sum(math.log(p) for p in precisions) / max_n
        geo_mean = math.exp(log_gm)
    else:
        geo_mean = 0.0

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    return {
        "bleu": 100.0 * bp * geo_mean,
        "brevity_penalty": bp,
        "hyp_len": hyp_len,
        "ref_len": ref_len,
        **{f"precision_{i+1}": 100.0 * p for i, p in enumerate(precisions)},
    }


def chrf(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """chrF: character n-gram F-score, beta=2 (recall weighted).

    The most informative of the three overlap metrics for Devanagari:
    it operates on character n-grams, so a generation that gets the stem
    right but inflects differently still scores partial credit, whereas
    word-level BLEU scores zero.
    """
    tp = [0] * max_n
    hyp_tot = [0] * max_n
    ref_tot = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        h_chars = list(hyp.replace(" ", ""))
        r_chars = list(ref.replace(" ", ""))
        for n in range(1, max_n + 1):
            h = _ngrams(h_chars, n)
            r = _ngrams(r_chars, n)
            tp[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            hyp_tot[n - 1] += sum(h.values())
            ref_tot[n - 1] += sum(r.values())

    precs, recs = [], []
    for n in range(max_n):
        precs.append(tp[n] / hyp_tot[n] if hyp_tot[n] else 0.0)
        recs.append(tp[n] / ref_tot[n] if ref_tot[n] else 0.0)

    p = sum(precs) / max_n
    r = sum(recs) / max_n
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Longest common subsequence length (O(len(a) * len(b)) space-optimized)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(
    hypotheses: List[List[str]],
    references: List[List[str]],
    beta: float = 1.2,
) -> Dict[str, float]:
    """ROUGE-L: LCS-based F-measure, averaged over the corpus.

    Unlike BLEU, LCS rewards in-order overlap without requiring contiguity,
    so it degrades more gracefully than 4-gram precision on generations
    that are topically right but locally reworded.
    """
    fs, ps, rs = [], [], []
    for hyp, ref in zip(hypotheses, references):
        lcs = _lcs_length(hyp, ref)
        p = lcs / len(hyp) if hyp else 0.0
        r = lcs / len(ref) if ref else 0.0
        if p + r > 0:
            b2 = beta ** 2
            f = (1 + b2) * p * r / (r + b2 * p)
        else:
            f = 0.0
        fs.append(f); ps.append(p); rs.append(r)

    n = max(len(fs), 1)
    return {
        "rouge_l_f": 100.0 * sum(fs) / n,
        "rouge_l_precision": 100.0 * sum(ps) / n,
        "rouge_l_recall": 100.0 * sum(rs) / n,
    }


# ------------------------------------------------------------ diversity

def distinct_n(token_lists: List[List[str]], n: int = 1) -> float:
    """Distinct-n: unique n-grams / total n-grams over all generations.

    Catches the failure mode overlap metrics miss: a model that emits the
    same phrase every time can score acceptably on BLEU while being
    useless. Low Distinct-2 with decent chrF is the signature of
    degenerate repetition.
    """
    total, uniq = 0, set()
    for toks in token_lists:
        grams = [tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def repetition_rate(token_lists: List[List[str]], n: int = 3) -> float:
    """Fraction of n-grams that are repeats WITHIN their own generation."""
    rep, total = 0, 0
    for toks in token_lists:
        grams = [tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        rep += sum(c - 1 for c in counts.values() if c > 1)
        total += len(grams)
    return rep / total if total else 0.0


def self_bleu(token_lists: List[List[str]], sample: int = 100) -> float:
    """Mean BLEU of each generation against the others -- corpus-level diversity.

    High self-BLEU means the model produces near-identical outputs for
    different prompts.
    """
    import random
    if len(token_lists) < 2:
        return 0.0
    idx = list(range(len(token_lists)))
    random.Random(42).shuffle(idx)
    idx = idx[:sample]
    scores = []
    for i in idx:
        others = [token_lists[j] for j in idx if j != i][:5]
        if not others:
            continue
        scores.append(max(
            corpus_bleu([token_lists[i]], [o])["bleu"] for o in others
        ))
    return sum(scores) / max(len(scores), 1)


# ------------------------------------------------------- combined entry point

def evaluate_generations(
    hypotheses_text: List[str],
    references_text: List[str],
    tokenize_fn: Callable[[str], List[str]],
    max_n: int = 4,
) -> Dict[str, object]:
    """Every required generation metric for one (model, decoding) pair.

    Args:
        hypotheses_text: generated continuations, decoded to text.
        references_text: the true held-out continuations.
        tokenize_fn: text -> token list. Pass the language's own
            SentencePiece encoder, NOT ``str.split``.
    """
    hyp_toks = [tokenize_fn(h) for h in hypotheses_text]
    ref_toks = [tokenize_fn(r) for r in references_text]

    bleu = corpus_bleu(hyp_toks, ref_toks, max_n=max_n)
    rouge = rouge_l(hyp_toks, ref_toks)

    return {
        "n_samples": len(hypotheses_text),
        "bleu_4": round(bleu["bleu"], 4),
        "bleu_brevity_penalty": round(bleu["brevity_penalty"], 4),
        "bleu_precisions": [round(bleu[f"precision_{i}"], 4)
                            for i in range(1, max_n + 1)],
        "chrf": round(chrf(hypotheses_text, references_text), 4),
        "rouge_l_f": round(rouge["rouge_l_f"], 4),
        "rouge_l_precision": round(rouge["rouge_l_precision"], 4),
        "rouge_l_recall": round(rouge["rouge_l_recall"], 4),
        "distinct_1": round(distinct_n(hyp_toks, 1), 6),
        "distinct_2": round(distinct_n(hyp_toks, 2), 6),
        "repetition_rate_3gram": round(repetition_rate(hyp_toks, 3), 6),
        "mean_generation_tokens": round(
            sum(len(t) for t in hyp_toks) / max(len(hyp_toks), 1), 2),
        "n_gram_order": max_n,
        "tokenization": "sentencepiece (language-specific, not whitespace)",
    }


def try_sacrebleu(hypotheses: List[str], references: List[str]) -> Optional[Dict]:
    """Cross-check against sacrebleu, if installed.

    The hand-written implementations above are what the report uses (they
    are transparent and their tokenization is under our control); this is
    a sanity check that they are in the right range.
    """
    try:
        import sacrebleu
    except ImportError:
        return None
    return {
        "sacrebleu_bleu": round(
            sacrebleu.corpus_bleu(hypotheses, [references]).score, 4),
        "sacrebleu_chrf": round(
            sacrebleu.corpus_chrf(hypotheses, [references]).score, 4),
    }
