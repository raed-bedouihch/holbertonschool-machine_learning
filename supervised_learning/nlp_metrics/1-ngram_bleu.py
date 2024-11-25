#!/usr/bin/env python3
"""a function that calculates the n-gram BLEU score for a sentence"""
import numpy as np


def calc_clipped_count(sentence_ngrams, ref_ngrams):
    """calculate clipped counts of n grams"""
    clip_count = 0
    for ngram in sentence_ngrams:
        if ngram in ref_ngrams:
            clip_count += 1
    return clip_count


def ngram_bleu(references, sentence, n):
    """Returns: the n-gram BLEU score"""
    short_ref_len = len(min(references, key=len))
    len_sentence = len(sentence)
    bp = min(1, np.exp(1 - short_ref_len / len_sentence))
    sentence_n_grams = []
    for i in range(len_sentence - n + 1):
        sentence_n_grams.append(tuple(sentence[i:i + n]))
    ref_n_grams = []
    for ref in references:
        for i in range(len(ref) - n + 1):
            ref_n_grams.append(tuple(ref[i:i + n]))
    clipped_count = calc_clipped_count(sentence_n_grams, ref_n_grams)
    total_sentence_count = len(sentence_n_grams)
    if total_sentence_count == 0:
        return 0
    precision_score = clipped_count / total_sentence_count
    return bp * np.exp(np.log(precision_score))
