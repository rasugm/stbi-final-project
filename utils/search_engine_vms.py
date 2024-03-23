import time
from pymongo import MongoClient
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.tokenizing_vms import tokenizer_vms
# from utils.loader_corpus_vms import load_corpus

def load_model(corpus_texts):
    print("Start loading VMS Model...")
    start_time = time.time()
    loaded_vsm_model = TfidfVectorizer(tokenizer=tokenizer_vms)
    corpus_vectors = loaded_vsm_model.fit_transform(corpus_texts)
    end_time = time.time()
    load_model_time = end_time - start_time
    print("Load Model Time: ", load_model_time)
    return loaded_vsm_model, corpus_vectors

def searching_query(query, loaded_vsm_model, corpus_data, corpus_vectors):
    start_time = time.time()

    query_lower = query.lower()
    query_vector = loaded_vsm_model.transform([query_lower])

    similarities = cosine_similarity(query_vector, corpus_vectors)

    k = 20
    top_indices = similarities.argsort()[0][-k:][::-1]

    similar_documents = [(i, corpus_data[i]['original_id'], corpus_data[i]['text']) for i in top_indices]

    list_doc_id = []

    for index, doc_id, text in similar_documents:
        object_data = {
            'document_id': doc_id,
            'score': similarities[0][index]
        }

        list_doc_id.append(object_data)

    def get_data_from_mongodb_with_similarity(documents):
        start_time = time.time()
        client = MongoClient('mongodb://root:admin123%23@localhost:27017/?authMechanism=SCRAM-SHA-1&authSource=admin')
        db = client['kpu']
        collection_dataset_caleg = db['col_dataset_caleg']

        documents_with_similarity = []

        for doc_obj in documents:
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

    documents_data = get_data_from_mongodb_with_similarity(list_doc_id)

    end_time = time.time()
    processing_time = end_time - start_time

    metadata = {
        "processing_time": processing_time,
        "query": query
    }

    return {
        "metadata": metadata,
        "documents": documents_data
    }
