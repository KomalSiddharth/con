import re

def simple_tokenize(text: str):
    # Split into words and punctuation while keeping order
    return re.findall(r"\w+|[^\w\s]", text)

def is_word(token: str) -> bool:
    return token.isalpha()

def same_casing(as_ref: str, cand: str) -> str:
    if as_ref.istitle():
        return cand.title()
    if as_ref.isupper():
        return cand.upper()
    if as_ref.islower():
        return cand.lower()
    return cand  # mixed: keep as is

def is_potential_proper_noun(token: str, idx: int):
    # Capitalized and not at sentence start -> likely proper noun
    return len(token) > 1 and token[0].isupper() and idx != 0
