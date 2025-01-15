import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import math
from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def load_documents(filepath):
    """
    Loads a collection of documents from a JSON Lines (jsonl) file.
    
    Args:
        filepath: The path to the JSONL file containing the documents.
        
    Returns:
        A list of dictionaries, where each dictionary represents a document.
    """
    documents = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            documents.append(json.loads(line.strip()))
    return documents

corpus_path = "scifact/corpus.jsonl"
queries_path = "scifact/queries.jsonl"

corpus = load_documents(corpus_path)
queries = load_documents(queries_path)

# print()
# print("CORPUS")
# print()
# print(corpus)
# print()
# print("QUERIES")
# print()
# print(queries)
# print()

def preprocess_documents(documents):
    """
    Preprocesses the documents by removing stopwords, tokenizing, 
    and applying stemming or lemmatization.
    
    Args:
        documents (list of str or list of dict): A list of text documents (or dictionaries with a 'text' field) to preprocess.
        
    Returns:
        list of list of str: A list of tokenized and preprocessed words for each document.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            doc = doc.get('text', '') 
        
        tokens = word_tokenize(doc.lower())
        
        processed_tokens = [
            lemmatizer.lemmatize(stemmer.stem(token))
            for token in tokens if token.isalpha() and token not in stop_words
        ]
        processed_docs.append(processed_tokens)
    
    return processed_docs

# print("PREPROCESSED CORPUS")
# print()
preprocessed_corpus = preprocess_documents(corpus)
# print(preprocessed_corpus)
# print()

def build_inverted_index(documents):
    """
    Constructs an inverted index from the preprocessed documents.
    
    Args:
        documents (list of list of str): A list where each entry is a list of preprocessed tokens 
        from a document.
        
    Returns:
        dict: An inverted index where keys are terms and values are lists of document IDs (0-based indices).
    """
    inverted_index = {}
    
    for doc_id, tokens in enumerate(documents): 
        for token in tokens:
            if token in inverted_index:
                if doc_id not in inverted_index[token]: 
                    inverted_index[token].append(doc_id)
            else:
                inverted_index[token] = [doc_id]
                
    return inverted_index

# print("INVERTED INDEX")
# print()
inverted_index = build_inverted_index(preprocessed_corpus)
# print(inverted_index)
# print()

def calculate_tf_idf(inverted_index, documents):
    """
    Calculates the TF-IDF scores for terms in the inverted index.
    
    Args:
        inverted_index (dict): An inverted index where keys are terms and values are lists of document IDs 
                               (or sets of document IDs).
        documents (list of list of str): A list of preprocessed documents where each document is a list of tokens.
        
    Returns:
        dict: A nested dictionary where keys are terms, and values are dictionaries with document IDs as keys 
              and TF-IDF scores as values.
    """
    total_documents = len(documents)
    tf_idf_scores = defaultdict(dict)
    
    for term, doc_ids in inverted_index.items():
        doc_freq = len(doc_ids)
        idf = math.log(total_documents / doc_freq)
        
        for doc_id in doc_ids:
            term_count_in_doc = inverted_index[term].count(doc_id) 
            tf = term_count_in_doc / len(documents[doc_id])
            tf_idf_scores[term][doc_id] = tf * idf
    
    return dict(tf_idf_scores)

# print("TF-IDF SCORES")
# print()
tf_idf = calculate_tf_idf(inverted_index, corpus)
# print(tf_idf)
# print()

def retrieve_documents(query, tf_idf_scores, scoring_method="tf_idf"):
    """
    Retrieves documents relevant to a query using precomputed TF-IDF scores.
    
    Args:
        query (str or dict): The query string or a dictionary containing the query.
        tf_idf_scores (dict): A nested dictionary where keys are terms, and values are dictionaries
                              mapping document IDs to TF-IDF scores.
        scoring_method (str): The scoring method to use (default is "tf_idf").
        
    Returns:
        list: A sorted list of relevant document IDs and scores.
    """
    
    # If the query is a dictionary, extract the 'text' field
    if isinstance(query, dict):
        query = query.get('text', '').lower()
    else:
        query = query.lower()

    query_terms = query.split()
    doc_scores = defaultdict(float)

    for term in query_terms:
        if term in tf_idf_scores:
            for doc_id, tf_idf in tf_idf_scores[term].items():
                if scoring_method == "tf_idf":
                    doc_scores[doc_id] += tf_idf
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring_method}")

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_docs

# print("RETRIEVING DOCUMENTS FOR QUERY X")
# print()
query = queries[0]
documents = retrieve_documents(query, tf_idf)
# print(documents)
# print()

def evaluate_retrieval(retrieved_documents, relevant_documents):
    """
    Evaluates the retrieval system using metrics such as precision, recall, and F1-score.
    
    Args:
        retrieved_documents: 
        relevant_documents: 
        
    Returns:
    """
    pass

def main():
    """
    Main function to execute the workflow: load documents, preprocess, build index, and evaluate retrieval.
    
    Args:
        
    Returns:
    """
    pass
