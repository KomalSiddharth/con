import argparse, math
from typing import List, Tuple
from dataclasses import dataclass
import textdistance
from wordfreq import zipf_frequency
from .scoring import MLMHelper
from .accept_rules import AcceptanceConfig
from .utils import simple_tokenize, is_word, same_casing, is_potential_proper_noun

def edit_distance(a: str, b: str) -> int:
    return textdistance.Levenshtein().distance(a, b)


@dataclass
class EditDecision:
    index: int
    original: str
    candidate: str
    reason: str
    cand_prob: float
    logprob_impr: float
    ppl_before: float
    ppl_after: float

class ContextAwareAutocorrect:
    def __init__(self, model_name: str = "bert-base-uncased", whitelist: set = None, cfg: AcceptanceConfig = None):
        self.mlm = MLMHelper(model_name=model_name)
        self.mask_tok = self.mlm.mask_token
        self.whitelist = set(w.lower() for w in (whitelist or []))
        self.cfg = cfg or AcceptanceConfig()

    def _word_freq_ok(self, w: str) -> bool:
        return zipf_frequency(w, 'en') >= self.cfg.min_zipf or w.lower() in self.whitelist

    def _mask_sentence(self, tokens: List[str], idx: int) -> str:
        masked = tokens.copy()
        masked[idx] = self.mask_tok
        out = []
        for i, t in enumerate(masked):
            if i > 0 and t not in ",.!?;:'\")]}" and masked[i-1] not in "([\"'" :
                out.append(" ")
            out.append(t)
        return "".join(out)

    def correct_sentence(self, sentence: str) -> Tuple[str, List[EditDecision]]:
        tokens = simple_tokenize(sentence)
        edits: List[EditDecision] = []
        applied = 0
        ppl_before_all = self.mlm.pseudo_perplexity(" ".join(tokens))

        for i, tok in enumerate(tokens):
            if applied >= self.cfg.max_edits_per_sentence:
                break
            if not is_word(tok):
                continue
            low = tok.lower()
            if low in self.whitelist:
                continue
            if is_potential_proper_noun(tok, i):
                continue

            masked = self._mask_sentence(tokens, i)

            # original token prob at masked position
            try:
                p_orig = self.mlm.masked_probs(masked, [low]).get(low, 0.0)
            except Exception:
                p_orig = 0.0

            nonword = zipf_frequency(low, 'en') == 0.0
            suspicious_realword = (not nonword) and (p_orig < self.cfg.realword_original_prob_threshold)
            if not (nonword or suspicious_realword):
                continue

            cands = self.mlm.topk_candidates(masked, top_k=15)
            best_decision = None
            best_score = -1e9

            for cand_tok, cand_prob in cands:
                cand = cand_tok
                if cand.lower() == low:
                    continue
                ed = edit_distance(low, cand.lower())
                if (len(low) <= 4 and ed > self.cfg.max_edit_distance_short) or                    (len(low) > 4 and ed > self.cfg.max_edit_distance):
                    continue
                if not self._word_freq_ok(cand):
                    continue

                cand_cased = same_casing(tok, cand)
                cand_tokens = tokens.copy()
                cand_tokens[i] = cand_cased
                cand_text = " ".join(cand_tokens)

                ppl_before = ppl_before_all
                ppl_after = self.mlm.pseudo_perplexity(cand_text)
                ppl_improve = (ppl_before - ppl_after) / max(ppl_before, 1e-9)

                log_impr = 0.0
                if p_orig > 0:
                    log_impr = math.log(max(cand_prob, 1e-12)) - math.log(max(p_orig, 1e-12))

                cond1 = (log_impr >= self.cfg.min_logprob_impr) or (cand_prob >= self.cfg.min_candidate_prob)
                cond5 = ppl_improve >= self.cfg.min_ppl_improve

                if cond1 and cond5:
                    score = ppl_improve * 5.0 + cand_prob + log_impr * 0.1
                    if score > best_score:
                        best_score = score
                        best_decision = EditDecision(
                            index=i, original=tok, candidate=cand_cased,
                            reason=f"ppl_improve={ppl_improve:.3f}", cand_prob=cand_prob,
                            logprob_impr=log_impr, ppl_before=ppl_before, ppl_after=ppl_after
                        )

            if best_decision:
                tokens[i] = best_decision.candidate
                edits.append(best_decision)
                applied += 1
                ppl_before_all = best_decision.ppl_after

        corrected = " ".join(tokens)
        return corrected, edits

def load_whitelist(path: str) -> set:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip().lower() for line in f if line.strip()])
    except FileNotFoundError:
        return set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single sentence to autocorrect.")
    parser.add_argument("--infile", type=str, help="Input file: one sentence per line.")
    parser.add_argument("--outfile", type=str, default="corrected.txt", help="Where to write corrected lines.")
    parser.add_argument("--whitelist", type=str, default="whitelist.txt", help="Whitelist file.")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="MLM model name.")
    args = parser.parse_args()

    wl = load_whitelist(args.whitelist)
    corrector = ContextAwareAutocorrect(model_name=args.model, whitelist=wl)

    if args.text:
        out, edits = corrector.correct_sentence(args.text)
        print(out)
        if edits:
            print("\n[Edits Applied]")
            for e in edits:
                print(f"- {e.original} -> {e.candidate} | prob={e.cand_prob:.3f} logΔ={e.logprob_impr:.2f} ppl: {e.ppl_before:.2f}→{e.ppl_after:.2f} ({e.reason})")
        else:
            print("\n(No edits applied)")
    elif args.infile:
        with open(args.infile, 'r', encoding='utf-8') as fin, open(args.outfile, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue
                out, _ = corrector.correct_sentence(line)
                fout.write(out + "\n")
        print(f"Wrote corrected lines to {args.outfile}")
    else:
        parser.error("Provide either --text or --infile.")

if __name__ == "__main__":
    main()
