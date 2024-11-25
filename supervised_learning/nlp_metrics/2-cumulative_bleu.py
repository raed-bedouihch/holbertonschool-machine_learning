#!/usr/bin/env python3
"""a function that calculates the cumulative
n-gram BLEU score for a sentence"""
import numpy as np


def calc_clipped_count(sentence_ngrams, ref_ngrams):
    """calculate clipped counts of n grams"""
    clip_count = 0
    for ngram in sentence_ngrams:
        if ngram in ref_ngrams:
            clip_count += 1
    return clip_count


def cumulative_bleu(references, sentence, n):
    """Returns: the cumulative n-gram BLEU score"""
    total_count = 0
    total_bleu = 0
    for i in range(1, n + 1):
        short_ref_len = len(min(references, key=len))
        len_sentence = len(sentence)
        if len_sentence == 0:
            return 0
        bp = min(1, np.exp(1 - short_ref_len / len_sentence))

        sentence_n_grams = []
        for j in range(len_sentence - i + 1):
            sentence_n_grams.append(tuple(sentence[j:j + i]))

        ref_n_grams = []
        for ref in references:
            for j in range(len(ref) - i + 1):
                ref_n_grams.append(tuple(ref[j:j + i]))

        clipped_count = calc_clipped_count(sentence_n_grams, ref_n_grams)
        total_sentence_count = len(sentence_n_grams)
        if total_sentence_count == 0:
            precision = 0
        else:
            precision = clipped_count / total_sentence_count
        if precision > 0:
            total_bleu += np.log(precision)
            total_count += 1
    if total_count == 0:
        return 0
    return bp * np.exp(total_bleu / total_count)
