#!/usr/bin/env python3
"""a function that creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Returns: embeddings, features"""
    tokens = []
    for sentence in sentences:
        cleaned_sentence = re.sub(r"'s\b", '', sentence.lower())
        words = re.findall(r'\b\w+\b', cleaned_sentence)
        tokens.append(words)
    if vocab is None:
        vocab = set()
        for sentence in tokens:
            for word in sentence:
                vocab.add(word)
        vocab = sorted(vocab)
    vocab = np.array(vocab)
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(tokens):
        for word in sentence:
            if word in vocab:
                embeddings[i][np.where(vocab == word)[0][0]] += 1
    return embeddings, vocab
