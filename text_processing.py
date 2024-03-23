import re
from nltk.tokenize import word_tokenize

def tokenizer(text):
    tokens = word_tokenize(text)
    # Optionally, you can apply further preprocessing steps such as removing punctuation or lowercasing
    tokens = [token.lower() for token in tokens if re.match(r'\b\w+\b', token)]
    return tokens