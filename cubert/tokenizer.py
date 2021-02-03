import collections
import torch
from tensor2tensor.data_generators import text_encoder
from cubert import java_tokenizer, code_to_subtokenized_sentences
import os

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token = token[1:-1]
            vocab[token] = index
            index += 1
    return vocab

class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""
    def __init__(self, vocab_file):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)
        self.tokenizer = java_tokenizer.JavaTokenizer()

    def tokenize(self, code):
        return code_to_subtokenized_sentences.code_to_cubert_sentences(
          code=code,
          initial_tokenizer=self.tokenizer,
          subword_tokenizer=self.subword_tokenizer)[0]


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


