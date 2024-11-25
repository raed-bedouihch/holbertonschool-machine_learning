#!/usr/bin/env python3
"""a function that calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """Returns: the unigram BLEU score"""
    sentence = np.array(sentence)
    unique_sentence, sentence_count = np.unique(sentence, return_counts=True)
    max_ref_counts = np.zeros_like(unique_sentence, dtype=int)
    for ref in references:
        ref = np.array(ref)
        unique_ref, ref_count = np.unique(ref, return_counts=True)
        for i, word in enumerate(unique_sentence):
            if word in unique_ref:
                max_ref_counts[i] = max(
                    max_ref_counts[i], ref_count[unique_ref == word][0])
    clipped_count = np.minimum(sentence_count, max_ref_counts)
    precision = np.sum(clipped_count) / len(sentence)
    ref_length = np.array([len(ref) for ref in references])
    close_ref_len = ref_length[np.argmin(np.abs(ref_length - len(sentence)))]
    if len(sentence) > close_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - close_ref_len / len(sentence))
    return precision * bp
