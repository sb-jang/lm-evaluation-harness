from typing import Dict, List, Tuple

import numpy as np
from evaluate import load
from rouge_score import rouge_scorer, scoring
from torchmetrics.text import SacreBLEUScore


BLEU_1_SCORER = None
BLEU_2_SCORER = None
BLEU_4_SCORER = None
ROUGE_SCORER = None
METEOR_SCORER = None


def process_results_gen(doc, results):
    completion = results[0]
    true_refs, false_refs = [doc["timely_response"]], [doc["untimely_response"]]
    all_refs = true_refs + false_refs

    # Process the sentence-level BLEU, ROUGE, METEOR for similarity measures.

    # BLEU
    bleu_scores = [bleu([[ref]], [completion]) for ref in all_refs]
    bleu_correct = bleu_scores[0]
    bleu_incorrect = bleu_scores[1]
    # BLEU calculated with timely response as reference := BLEU(timely)
    bleu_1_correct = bleu_correct[0]
    bleu_2_correct = bleu_correct[1]
    bleu_4_correct = bleu_correct[2]
    # diff(BLEU) := BLEU(timely) - BLEU(untimely)
    bleu_diff = bleu_correct - bleu_incorrect
    bleu_1_diff = bleu_diff[0]
    bleu_2_diff = bleu_diff[1]
    bleu_4_diff = bleu_diff[2]
    # acc(BLEU) := ratio of BLEU(timely) > BLEU(untimely)
    bleu_acc = bleu_correct > bleu_incorrect
    bleu_1_acc = int(bleu_acc[0])
    bleu_2_acc = int(bleu_acc[1])
    bleu_4_acc = int(bleu_acc[2])

    # ROUGE
    rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
    # ROUGE-1
    rouge1_scores = [score["rouge1"] for score in rouge_scores]
    rouge1_correct = rouge1_scores[0]
    rouge1_incorrect = rouge1_scores[1]
    rouge1_diff = rouge1_correct - rouge1_incorrect
    rouge1_acc = int(rouge1_correct > rouge1_incorrect)
    # ROUGE-2
    rouge2_scores = [score["rouge2"] for score in rouge_scores]
    rouge2_correct = rouge2_scores[0]
    rouge2_incorrect = rouge2_scores[1]
    rouge2_diff = rouge2_correct - rouge2_incorrect
    rouge2_acc = int(rouge2_correct > rouge2_incorrect)
    # ROUGE-L
    rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
    rougeL_correct = rougeL_scores[0]
    rougeL_incorrect = rougeL_scores[1]
    rougeL_diff = rougeL_correct - rougeL_incorrect
    rougeL_acc = int(rougeL_correct > rougeL_incorrect)

    # METEOR
    meteor_scores = [meteor([ref], [completion]) for ref in all_refs]
    meteor_correct = meteor_scores[0]
    meteor_incorrect = meteor_scores[1]
    meteor_diff = meteor_correct - meteor_incorrect
    meteor_acc = int(meteor_correct > meteor_incorrect)

    return {
        "bleu_1_correct": bleu_1_correct,
        "bleu_1_diff": bleu_1_diff,
        "bleu_1_acc": bleu_1_acc,
        "bleu_2_correct": bleu_2_correct,
        "bleu_2_diff": bleu_2_diff,
        "bleu_2_acc": bleu_2_acc,
        "bleu_4_correct": bleu_4_correct,
        "bleu_4_diff": bleu_4_diff,
        "bleu_4_acc": bleu_4_acc,
        "rouge1_correct": rouge1_correct,
        "rouge1_diff": rouge1_diff,
        "rouge1_acc": rouge1_acc,
        "rouge2_correct": rouge2_correct,
        "rouge2_diff": rouge2_diff,
        "rouge2_acc": rouge2_acc,
        "rougeL_correct": rougeL_correct,
        "rougeL_diff": rougeL_diff,
        "rougeL_acc": rougeL_acc,
        "meteor_correct": meteor_correct,
        "meteor_diff": meteor_diff,
        "meteor_acc": meteor_acc,
    }


def bleu(refs: List[List[str]], preds: List[str]) -> np.ndarray:
    """
    :param refs: list of list of str
    :param preds: list of str
    :return: (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    """
    global BLEU_1_SCORER, BLEU_2_SCORER, BLEU_4_SCORER
    if BLEU_1_SCORER is None:
        BLEU_1_SCORER = SacreBLEUScore(n_gram=1, lowercase=True)
    bleu_1_scorer = BLEU_1_SCORER
    if BLEU_2_SCORER is None:
        BLEU_2_SCORER = SacreBLEUScore(n_gram=2, lowercase=True)
    bleu_2_scorer = BLEU_2_SCORER
    if BLEU_4_SCORER is None:
        BLEU_4_SCORER = SacreBLEUScore(n_gram=4, lowercase=True)
    bleu_4_scorer = BLEU_4_SCORER

    bleu_1_score = bleu_1_scorer(preds, refs).item()
    bleu_2_score = bleu_2_scorer(preds, refs).item()
    bleu_4_score = bleu_4_scorer(preds, refs).item()
    return np.array([bleu_1_score * 100, bleu_2_score * 100, bleu_4_score * 100])


def rouge(refs: List[str], preds: List[str]) -> Dict[str, float]:
    """
    :param refs: list of str
    :param preds: list of str
    :return: (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER
    
    # Add newlines between setences to correctly compute `rougeLsum`.
    def _prepare_summary(summary: str) -> str:
        summary = summary.replace(" . ", ".\n").lower()
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type_: result[type_].mid.fmeasure * 100 for type_ in rouge_types}


def meteor(refs: List[str], preds: List[str]) -> float:
    """
    :param refs: list of str
    :param preds: list of str
    :return: METEOR
    """
    global METEOR_SCORER
    if METEOR_SCORER is None:
        METEOR_SCORER = load("meteor")
    scorer = METEOR_SCORER

    preds = [pred.lower() for pred in preds]
    refs = [[ref.lower()] for ref in refs]

    meteor_score = scorer.compute(predictions=preds, references=refs)
    return meteor_score["meteor"] * 100