# yaha metrics likhe jaayenge 
import os, statistics
from dataclasses import dataclass
from typing import Tuple, List
from src.context_autocorrect import ContextAwareAutocorrect, load_whitelist

BASE = os.path.dirname(os.path.dirname(__file__))
WL = os.path.join(BASE, 'whitelist.txt')

@dataclass
class Counts:
    det_tp: int = 0
    det_fp: int = 0
    det_fn: int = 0
    cor_tp: int = 0
    cor_fp: int = 0
    cor_fn: int = 0

def load_pairs(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            src, tgt = line.rstrip().split('\t')
            pairs.append((src, tgt))
    return pairs

def token_changes(a: str, b: str):
    a_toks = a.split()
    b_toks = b.split()
    changes = []
    for i, (x, y) in enumerate(zip(a_toks, b_toks)):
        if x != y:
            changes.append((i, x, y))
    return changes

def evaluate():
    wl = load_whitelist(WL)
    corrector = ContextAwareAutocorrect(whitelist=wl)

    nonword = load_pairs(os.path.join(BASE, 'data', 'mini', 'nonword.tsv'))
    realword = load_pairs(os.path.join(BASE, 'data', 'mini', 'realword.tsv'))
    clean = [l.strip() for l in open(os.path.join(BASE, 'data', 'mini', 'clean.txt'), encoding='utf-8').read().splitlines() if l.strip()]

    counts = Counts()
    mlm = corrector.mlm
    deltas_corrected = []
    deltas_clean = []

    # Non-word
    for src, tgt in nonword:
        out, edits = corrector.correct_sentence(src)
        changes = token_changes(src, out)
        gold_changes = token_changes(src, tgt)

        counts.det_tp += min(len(changes), len(gold_changes))
        if len(changes) > len(gold_changes):
            counts.det_fp += len(changes) - len(gold_changes)
        else:
            counts.det_fn += len(gold_changes) - len(changes)

        if out == tgt:
            counts.cor_tp += len(gold_changes)
        else:
            match = sum(1 for c in changes if c in gold_changes)
            counts.cor_tp += match
            counts.cor_fp += max(0, len(changes) - match)
            counts.cor_fn += max(0, len(gold_changes) - match)

        if changes:
            ppl_b = mlm.pseudo_perplexity(src)
            ppl_a = mlm.pseudo_perplexity(out)
            deltas_corrected.append((ppl_b - ppl_a) / max(ppl_b, 1e-9))

    # Real-word
    for src, tgt in realword:
        out, edits = corrector.correct_sentence(src)
        changes = token_changes(src, out)
        gold_changes = token_changes(src, tgt)

        counts.det_tp += min(len(changes), len(gold_changes))
        if len(changes) > len(gold_changes):
            counts.det_fp += len(changes) - len(gold_changes)
        else:
            counts.det_fn += len(gold_changes) - len(changes)

        if out == tgt:
            counts.cor_tp += len(gold_changes)
        else:
            match = sum(1 for c in changes if c in gold_changes)
            counts.cor_tp += match
            counts.cor_fp += max(0, len(changes) - match)
            counts.cor_fn += max(0, len(gold_changes) - match)

        if changes:
            ppl_b = mlm.pseudo_perplexity(src)
            ppl_a = mlm.pseudo_perplexity(out)
            deltas_corrected.append((ppl_b - ppl_a) / max(ppl_b, 1e-9))

    # Clean set: CTPR and ΔPPL drift
    total_clean = len(clean)
    unchanged = 0
    for s in clean:
        out, _ = corrector.correct_sentence(s)
        if out == s:
            unchanged += 1
        ppl_b = mlm.pseudo_perplexity(s)
        ppl_a = mlm.pseudo_perplexity(out)
        deltas_clean.append((ppl_a - ppl_b) / max(ppl_b, 1e-9))

    # Metrics
    det_prec = counts.det_tp / max(counts.det_tp + counts.det_fp, 1)
    det_rec  = counts.det_tp / max(counts.det_tp + counts.det_fn, 1)
    cor_prec = counts.cor_tp / max(counts.cor_tp + counts.cor_fp, 1)
    cor_rec  = counts.cor_tp / max(counts.cor_tp + counts.cor_fn, 1)
    ctpr = unchanged / max(total_clean, 1)
    ucr  = 1 - ctpr
    ppl_improve_corrected = statistics.mean(deltas_corrected) if deltas_corrected else 0.0
    ppl_delta_clean = statistics.mean(deltas_clean) if deltas_clean else 0.0

    print("== Evaluation (Mini) ==")
    print(f"Detection Precision: {det_prec:.3f}")
    print(f"Detection Recall   : {det_rec:.3f}")
    print(f"Correction Precision: {cor_prec:.3f}")
    print(f"Correction Recall   : {cor_rec:.3f}")
    print(f"Clean Text Preservation Rate (CTPR): {ctpr:.3f}")
    print(f"Unnecessary Change Rate (UCR): {ucr:.3f}")
    print(f"ΔPPL (corrected sentences avg improvement): {ppl_improve_corrected*100:.2f}% (target ≥ 5%)")
    print(f"ΔPPL (clean sentences avg drift): {ppl_delta_clean*100:.2f}% (target ≤ +1%)")

if __name__ == "__main__":
    evaluate()
