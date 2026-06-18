import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx,token in enumerate(special_tokens):
            self.word_to_id[token] = idx        
            self.id_to_word[idx] = token 

        vocab = set()
        for text in texts:
            for word in text.split():
                vocab.add(word)
        sorted_vocab = sorted(vocab)
        for idx,word in enumerate(sorted_vocab):
            targt = idx  + 4 
            self.word_to_id[word] = targt
            self.id_to_word[targt] = word 
        self.vocab_size = len(sorted_vocab) + 4 
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if len(text) == 0:
            return []
        text = text.lower().split(' ')
        encoded_list = []
        for t in text:
            encoded_list.append(self.word_to_id.get(t,self.word_to_id[self.unk_token]))
        return encoded_list 
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded_list = []
        for id in ids:
            decoded_list.append(self.id_to_word.get(id, self.unk_token))
        return " ".join(decoded_list)