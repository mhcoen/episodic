import torch
import torch.nn as nn
import numpy as np
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

from sklearn.metrics import precision_recall_fscore_support
import statistics

def convert_seg_hypo(pred, diag):
    """
    input : pred (str) (e.g. '3,5')
    output : list of int e.g. [0,0,0,1,1,2,2,2]
    """
    topic_shift = list(map(int, pred.split(',')))
    target_seq, topic_cnt = [], 0
    for i in range(len(diag)):
        if i in topic_shift:
            topic_cnt += 1
        target_seq.append(topic_cnt)
        
    return target_seq 

def convert_seg_ref(diag):
    target_seq, topic_cnt = [], 0
    for i in range(len(diag)):
        if i in _sample['topic_shift']:
            topic_cnt += 1
        target_seq.append(topic_cnt)
        
    return target_seq

def _calculate_accuracy(_ref, _hypo):
    correct = sum(p == t for p, t in zip(_hypo, _ref))
    total = len(_hypo)
    accuracy = correct / total
    return accuracy

def _calculate_f1_score(_ref, _hypo):
    precision, recall, f1_score, _ = precision_recall_fscore_support(_ref, _hypo, average='weighted', zero_division=1)
    return precision, recall, f1_score

def calc_classification(ref, hypo):
    """
    input : ref, dictionary of reference sentences (id, sentence)
            hypo, dictionary of hypothesis sentences (id, sentence)
    output : score, dictionary of scores
    """
    em, p, r, f1 = 0, [],[],[]
    for i in range(len(ref)):
        if all(isinstance(key, str) for key in ref.keys()): 
            i = str(i)
        if ref[i] == hypo[i]:
            em+=1
        _p, _r, _f1 = _calculate_f1_score(ref[i], hypo[i])
        p.append(_p)
        r.append(_r)
        f1.append(_f1)
    
    em /= len(ref)
    p, r, f1 = np.mean(p),  np.mean(r),  np.mean(f1)
    
    return em, p, r, f1

def calculate_nlg_metrics(ref, hypo):
    """
    input : ref, dictionary of reference sentences (id, sentence)
            hypo, dictionary of hypothesis sentences (id, sentence)
    output : score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores 

if __name__ == "__main__":
    ref = {0: ['a'], 1: ['b']}
    hypo = {0: ['a b c'], 1: ['a b c']}
    metrics = calculate_nlg_metrics(ref, hypo)
    print(metrics)