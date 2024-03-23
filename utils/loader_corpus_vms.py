import pickle

def load_corpus():
    with open('corpus_data.pkl', 'rb') as f:
        corpus_data = pickle.load(f)
    return corpus_data
