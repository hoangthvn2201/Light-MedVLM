from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score


@dataclass
class ReportMetrics:
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    meteor: float
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bertscore_f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "BLEU-1": self.bleu1,
            "BLEU-2": self.bleu2,
            "BLEU-3": self.bleu3,
            "BLEU-4": self.bleu4,
            "METEOR": self.meteor,
            "ROUGE-1 F": self.rouge1_f,
            "ROUGE-2 F": self.rouge2_f,
            "ROUGE-L F": self.rougeL_f,
            "BERTScore-F1": self.bertscore_f1,
        }


def compute_report_metrics(
    gt_list: Sequence[str],
    pred_list: Sequence[str],
    bertscore_lang: str = "en",
    bertscore_rescale_with_baseline: bool = True,
) -> ReportMetrics:
    """Compute BLEU(1-4), METEOR, ROUGE(1/2/L F1) and BERTScore-F1.

    This follows the original notebook logic (tokenization = simple `.split()`).
    """
    assert len(gt_list) == len(pred_list), "gt_list and pred_list must have the same length"

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu1_scores: List[float] = []
    bleu2_scores: List[float] = []
    bleu3_scores: List[float] = []
    bleu4_scores: List[float] = []
    meteor_scores: List[float] = []
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougel_scores: List[float] = []

    for g, p in zip(gt_list, pred_list):
        g_tok = g.split()
        p_tok = p.split()

        bleu1_scores.append(sentence_bleu([g_tok], p_tok, weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu2_scores.append(sentence_bleu([g_tok], p_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu3_scores.append(sentence_bleu([g_tok], p_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu([g_tok], p_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        meteor_scores.append(meteor_score([g_tok], p_tok))

        r = scorer.score(g, p)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougel_scores.append(r["rougeL"].fmeasure)

    # BERTScore
    P, R, F1 = bert_score(
        list(pred_list),
        list(gt_list),
        lang=bertscore_lang,
        rescale_with_baseline=bertscore_rescale_with_baseline,
    )

    mean = lambda xs: float(sum(xs) / max(1, len(xs)))

    return ReportMetrics(
        bleu1=mean(bleu1_scores),
        bleu2=mean(bleu2_scores),
        bleu3=mean(bleu3_scores),
        bleu4=mean(bleu4_scores),
        meteor=mean(meteor_scores),
        rouge1_f=mean(rouge1_scores),
        rouge2_f=mean(rouge2_scores),
        rougeL_f=mean(rougel_scores),
        bertscore_f1=float(F1.mean()),
    )
