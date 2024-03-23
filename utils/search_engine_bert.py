import time
from pymongo import MongoClient
import json
from rank_bm25 import BM25Okapi
from bson import ObjectId

def get_data_from_mongodb_with_similarity(documents):
	start_time = time.time()
	client = MongoClient('mongodb://root:admin123%23@localhost:27017/?authMechanism=SCRAM-SHA-1&authSource=admin')
	db = client['kpu']
	collection_dataset_caleg = db['col_dataset_caleg']

	documents_with_similarity = []

	for doc_obj in documents:
		print(doc_obj)
		document_id = ObjectId(doc_obj['document_id'])
		score = doc_obj['score']

		query = {"_id": document_id}
		result = collection_dataset_caleg.find_one(query)

		if result:
			result['score'] = score
			documents_with_similarity.append(result)

	client.close()
	
	end_time = time.time()
	load_mongodb_processing_time = end_time - start_time
	print("Load MongoDB Processing Time: ", load_mongodb_processing_time)

	return documents_with_similarity

def search_engine_bert(query, tokenized_docs, documents, tokenizer):

	# Function to create BM25 index
	def create_bm25_index(tokenized_docs):
		return BM25Okapi(tokenized_docs)

	# Create BM25 index
	bm25 = create_bm25_index(tokenized_docs)

	# Function to preprocess query
	def preprocess_query(query):
		return tokenizer.tokenize(query.lower())

	# Function to perform search using BM25
	def bm25_search(query, bm25_index, documents, top_n=20):
		tokenized_query = preprocess_query(query)
		doc_scores = bm25_index.get_scores(tokenized_query)
		
		# Normalize scores using min-max scaling
		min_score = min(doc_scores)
		max_score = max(doc_scores)
		normalized_scores = [(score - min_score) / (max_score - min_score) for score in doc_scores]
		
		top_doc_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:top_n]
		return [(documents[i]['original_id'], normalized_scores[i]) for i in top_doc_indices]


	# Perform BM25 search
	start_time = time.time()
	bm25_search_results = bm25_search(query, bm25, documents)
	search_time = time.time() - start_time

	# Display BM25 search results
 	# Modified to Output JSON
  
	list_doc_id = []
 
	if bm25_search_results:
		print("BM25 Search Results:")
		for i, (doc_id, score) in enumerate(bm25_search_results, start=1):
			print(i)
			documents_data = {
				"document_id": doc_id,
				"score": score
			}
   
			list_doc_id.append(documents_data)
   
	else:
		print("No relevant documents found.")
  
	print("Search Time:", search_time)
 
	documents_data = get_data_from_mongodb_with_similarity(list_doc_id)
  
	metadata = {
        "processing_time": search_time,
        "query": query
    }

	return {
        "metadata": metadata,
        "documents": documents_data
    }

	
