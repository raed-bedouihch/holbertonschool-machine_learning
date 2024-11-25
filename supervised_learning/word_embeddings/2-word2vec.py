#!/usr/bin/env python3
"""a function that creates , builds and trains a gensim word2vec model"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Returns: the trained model"""
    sg = 1 if not cbow else 0
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers)
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs)
    return model
