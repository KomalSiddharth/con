from dataclasses import dataclass

@dataclass
class AcceptanceConfig:
    min_logprob_impr:float=2.0
    min_candidate_prob:float=0.6
    max_edit_distance:int=2
    max_edit_distance_short:int=1 #if len<=4 
    min_zipf: float=1.5 # frequency threshold via wordfreq.zip
    min_ppl_improve:float=0.03
    max_edits_per_sentence:int=2
    realword_original_prob_threshold: float=0.15
