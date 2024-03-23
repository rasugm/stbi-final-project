from flask import Blueprint, jsonify, request
from config.database.mongo import Mongo
from bson import json_util
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils.search_engine_vms import searching_query
from utils.search_engine_vms import load_model
from utils.loader_corpus_vms import load_corpus
from utils.search_engine_bert import search_engine_bert

from pymongo import MongoClient
from transformers import BertTokenizer

import time


# LOAD VMS MODEL
corpus_data = load_corpus()
corpus_texts = [entry['text'] for entry in corpus_data]
loaded_vsm_model, corpus_vectors = load_model(corpus_texts)

# LOAD BERT MODEL
# Connect to MongoDB
client = MongoClient('mongodb://root:admin123%23@localhost:27017/?authMechanism=SCRAM-SHA-1&authSource=admin')
db = client['kpu']
collection_dataset_caleg_training = db['col_dataset_caleg_text']

# Retrieve data from MongoDB collection
data_collection = list(collection_dataset_caleg_training.find())

# Convert ObjectId to string
for data in data_collection:
    data['_id'] = str(data['_id'])
    data['original_id'] = str(data['original_id'])

# Print the modified data_collection
documents = json.loads(json.dumps(data_collection))

print("Start loading BERT tokenizer...")
start_time = time.time()
# Load pre-trained BERT tokenizer for Indonesian
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
end_time = time.time()
load_model_time = end_time - start_time
print("Load Model Time: ", load_model_time)

# Function to tokenize documents
def tokenize_documents(documents):
    tokenized_docs = [tokenizer.tokenize(doc['_id'] + ' ' + doc['text'].lower()) for doc in documents]
    return tokenized_docs

print("Start loading tokenized docs...")
# Tokenize documents
start_time_tokenized_docs = time.time()
tokenized_docs = tokenize_documents(documents)
end_time_tokenized_docs = time.time()
load_time_tokenized_docs = end_time_tokenized_docs - start_time_tokenized_docs
print("Load Tokenized Docs: ", load_time_tokenized_docs)


search = Blueprint('search', __name__)

@search.route('/search/vms')
def searchFunctionVsm():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter "query" is required.'}), 400


    # Process the query and retrieve similar documents
    similar_documents = searching_query(query, loaded_vsm_model, corpus_data, corpus_vectors)
    
    response = json.loads(json_util.dumps(similar_documents))
    return jsonify(response)
    # Return the similar documents as JSON response
    # return jsonify(similar_documents)
    
@search.route('/search/bert')
def searchFunctionBert():
    query = request.args.get('query')
    similar_documents = search_engine_bert(query, tokenized_docs, documents, tokenizer)
    
    response = json.loads(json_util.dumps(similar_documents))
    return jsonify(response)
    # Return the similar documents as JSON response
    # return jsonify(similar_documents)

@search.route('/train')
def train():
    args = request.args
    cursor = Mongo().db["doc_caleg_cleaned"].find({})
    result_json = json.loads(json_util.dumps(list(cursor)))

    return result_json

@search.route('/testing')
def testing():
    
    # Example document collection
    documents = {
        'Document 1': "Introduction to Artificial Intelligence",
        'Document 2': "Applications of Machine Learning in Healthcare",
        'Document 3': "Overview of Natural Language Processing",
        'Document 4': "Deep Learning Techniques for Image Recognition",
        'Document 5': "Advancements in Reinforcement Learning"
    }

    # Example relevance judgments
    relevance_judgments = {
        'Machine Learning algorithms': ['Document 2', 'Document 4'],
        'Neural networks': ['Document 4'],
        'Natural language processing applications': ['Document 3']
    }

    # Example test queries
    test_queries = [
        "Machine Learning algorithms",
        "Neural networks",
        "Deep learning applications",
        "Reinforcement learning techniques"
    ]
    
    def load_model(filename):
        with open(filename, 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer
    
    def retrieve_documents(query, vectorizer, document_vectors, k=3):
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        top_indices = similarities.argsort()[-k:][::-1]  # Top k most similar documents
        return [list(documents.keys())[i] for i in top_indices]
    
    def evaluate_search_engine_with_metrics(vectorizer, document_vectors):
        all_expected_relevant_documents = []
        all_retrieved_documents = []
        
        for query in test_queries:
            expected_relevant_documents = relevance_judgments.get(query, [])
            retrieved_documents = retrieve_documents(query, vectorizer, document_vectors, k=3)
            
            # Extend the lists with the same length
            all_expected_relevant_documents.extend([1 if doc in expected_relevant_documents else 0 for doc in documents])
            all_retrieved_documents.extend([1 if doc in retrieved_documents else 0 for doc in documents])
        
        # Calculate precision, recall, F1 score, and accuracy
        precision = precision_score(all_expected_relevant_documents, all_retrieved_documents)
        recall = recall_score(all_expected_relevant_documents, all_retrieved_documents)
        f1 = f1_score(all_expected_relevant_documents, all_retrieved_documents)
        accuracy = accuracy_score(all_expected_relevant_documents, all_retrieved_documents)
        
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1)
        # print("Accuracy:", accuracy)

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(all_expected_relevant_documents, all_retrieved_documents)
        print("Confusion Matrix:")
        print(confusion_mat)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
    
    loaded_vectorizer = load_model("vsm_search_engine_model.pkl")
    loaded_document_vectors = loaded_vectorizer.transform(documents.values())

    evaluate = evaluate_search_engine_with_metrics(loaded_vectorizer, loaded_document_vectors)
    
    print(evaluate)
    
    return {
        "status": "success",
        "data": {
            "precision": float(evaluate["precision"]),
            "recall": float(evaluate["recall"]),
            "f1": float(evaluate["f1"]),
            "accuracy": float(evaluate["accuracy"]),
            # "confusion_matrix": confusion_mat
        }
    }
