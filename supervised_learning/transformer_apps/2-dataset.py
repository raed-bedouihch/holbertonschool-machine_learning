#!/usr/bin/env python3
"""a class Dataset that loads and preps a dataset for machine translation"""
import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """Dataset class"""

    def __init__(self):
        """constructor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split="train",
            as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split="validation",
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for the dataset"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        def iterate_pt():
            """generate potuguese sentences one at a time from the dataset"""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def iterate_en():
            """generate english sentences one at a time from the dataset"""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate_en(), vocab_size=2**13)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens"""
        pt_tokens = self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        en_tokens = self.tokenizer_en.encode(en.numpy().decode('utf-8'))
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            en_tokens + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """acts as a tensorflow wrapper for the encode instance method"""
        encoded_pt, encoded_en = tf.py_function(
            func=self.encode, inp=[
                pt, en], Tout=[
                tf.int64, tf.int64])
        encoded_pt.set_shape([None])
        encoded_en.set_shape([None])
        return encoded_pt, encoded_en
