import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

# Load and preprocess corpus
def load_corpus(file_path):
    corpus = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc['text']
    return corpus

# Load test queries
def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']  # Corrected key
    return queries


# Load relevance judgments
def load_relevance(file_path):
    relevance = defaultdict(set)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'doc_id', 'score'])
    for _, row in df.iterrows():
        relevance[str(row['query_id'])].add(str(row['doc_id']))  
    return relevance

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def build_tfidf_index(corpus):
    vectorizer = TfidfVectorizer(preprocessor=preprocess, stop_words='english')
    doc_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in doc_ids]
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    return vectorizer, tfidf_matrix, doc_ids

# Retrieve and Rank Documents
def retrieve_rank(queries, vectorizer, tfidf_matrix, doc_ids):
    results = []
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_vectors = vectorizer.transform(query_texts)
    
    for idx, query_id in enumerate(query_ids):
        print(f"Processing Query ID: {query_id}")  # Debug print for query processing
        similarities = []
        for doc_idx, doc_id in enumerate(doc_ids):
            query_vector = query_vectors[idx].toarray().flatten()
            doc_vector = tfidf_matrix[doc_idx].toarray().flatten()
            
            if np.any(query_vector) and np.any(doc_vector):
                sim = 1 - cosine(query_vector, doc_vector)
                print(f"Similarity between Query {query_id} and Doc {doc_id}: {sim}")  # Debug similarity score
            else:
                sim = 0  # Assign zero similarity if one of the vectors is empty
            
            if sim > 0:  # Only add non-zero similarities
                similarities.append((doc_id, sim))
        
        if not similarities:
            print(f"No similar documents found for Query {query_id}.")  # Debug for no results
        
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:100]
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_name")
    
    return results


# Main execution
def main():
    base_path = "/Users/saramahendran/Desktop/1"
    corpus_path = f"{base_path}/scifact/corpus.jsonl"
    queries_path = f"{base_path}/scifact/queries.jsonl"
    relevance_path = f"{base_path}/scifact/qrels/test.tsv"
    results_path = f"{base_path}/Results.txt"

    # Load dataset
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    relevance = load_relevance(relevance_path)

    # Build index and retrieve rankings
    vectorizer, tfidf_matrix, doc_ids = build_tfidf_index(corpus)
    results = retrieve_rank(queries, vectorizer, tfidf_matrix, doc_ids)

    # Debug: Check how many results are generated
    print(f"Total results generated: {len(results)}")

    # Save results
    if results:
        with open(results_path, "w") as f:
            f.write("\n".join(results))
        print(f"Results file generated successfully: {results_path}")
    else:
        print("No results to write to the file.")
if __name__ == "__main__":
    main()
