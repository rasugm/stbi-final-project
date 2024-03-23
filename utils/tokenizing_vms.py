import re
from nltk.tokenize import word_tokenize

def tokenizer_vms(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if re.match(r'\b\w+\b', token)]
    return tokens