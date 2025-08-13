# yaha BERT helper aur pseaudo perplexity ka logic likha jaayega 
from typing import List, Tuple
import math
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class MLMHelper:
    def __init__(self, model_name: str = "bert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    @torch.no_grad()
    def masked_probs(self, text_with_mask: str, target_tokens: List[str]):
        enc = self.tokenizer(text_with_mask, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            raise ValueError("No [MASK] token found in input.")
        pos = mask_positions[0, 1].item()
        logits = self.model(**enc).logits[0, pos]
        probs = torch.softmax(logits, dim=-1)
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in target_tokens]
        out = {}
        for t, tid in zip(target_tokens, ids):
            out[t] = float(probs[tid].item()) if tid is not None and tid >= 0 else 0.0
        return out

    @torch.no_grad()
    def topk_candidates(self, text_with_mask: str, top_k: int = 15) -> List[Tuple[str, float]]:
        enc = self.tokenizer(text_with_mask, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            raise ValueError("No [MASK] token found in input.")
        pos = mask_positions[0, 1].item()
        logits = self.model(**enc).logits[0, pos]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=top_k)
        ids = topk.indices.tolist()
        scores = topk.values.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        outs = []
        for t, s in zip(tokens, scores):
            if t.startswith("##") or t in {"[CLS]","[SEP]","[PAD]"}:
                continue
            outs.append((t, float(s)))
        return outs

    @torch.no_grad()
    def pseudo_perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        n_tokens = 0
        nll = 0.0
        for i in range(1, input_ids.shape[0]-1):  # skip [CLS]/[SEP]
            if attn[i].item() == 0:
                continue
            orig_id = input_ids[i].item()
            masked = input_ids.clone()
            masked[i] = self.mask_token_id
            logits = self.model(input_ids=masked.unsqueeze(0), attention_mask=attn.unsqueeze(0)).logits[0, i]
            probs = torch.softmax(logits, dim=-1)
            p = float(probs[orig_id].item())
            if p <= 1e-12:
                p = 1e-12
            nll += -math.log(p)
            n_tokens += 1
        if n_tokens == 0:
            return float('inf')
        return math.exp(nll / n_tokens)
