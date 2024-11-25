#!/usr/bin/env python3
"""a function that creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis"""
    tf_idf_vect = TfidfVectorizer(vocabulary=vocab)
    tf_idf = tf_idf_vect.fit_transform(sentences).toarray()
    features = tf_idf_vect.get_feature_names_out()
    return tf_idf, features
