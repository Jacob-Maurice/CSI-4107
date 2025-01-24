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

print()
print("CORPUS")
print()
print(corpus)
print()
print("QUERIES")
print()
print(queries)
print()

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

print("PREPROCESSED CORPUS")
print()
preprocessed_corpus = preprocess_documents(corpus)
print(preprocessed_corpus)
print()
print("PREPROCESSED QUERIES")
print()
preprocessed_queries = preprocess_documents(queries)
print(preprocessed_queries)
print()

def build_inverted_index(documents):
    """
    Constructs an inverted index from the preprocessed documents and computes TF and DF.

    Args:
        documents (list of list of str): A list where each entry is a list of preprocessed tokens 
        from a document.

    Returns:
        dict: An inverted index where keys are terms and values are dictionaries with:
              - "doc_ids": a list of document IDs (0-based indices) where the term appears.
              - "tf": a list of term frequencies corresponding to each document.
              - "df": the document frequency of the term (total number of documents it appears in).
    """
    inverted_index = {}

    for doc_id, tokens in enumerate(documents):
        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        for token, count in term_counts.items():
            if token not in inverted_index:
                inverted_index[token] = {"doc_ids": [], "tf": [], "df": 0}

            inverted_index[token]["doc_ids"].append(doc_id)
            inverted_index[token]["tf"].append(count)
            inverted_index[token]["df"] += 1

    return inverted_index

print("INVERTED INDEX")
print()
combined_documents = preprocessed_corpus + preprocessed_queries
inverted_index = build_inverted_index(combined_documents)
print(inverted_index)
print()

def calculate_tf_idf(inverted_index, total_documents):
    """
    Calculates the TF-IDF scores for terms in the inverted index.
    
    Args:
        inverted_index (dict): An inverted index where keys are terms and values are dictionaries with:
                               - "doc_ids": List of document IDs where the term appears.
                               - "tf": List of term frequencies for each corresponding document.
                               - "df": Document frequency of the term.
        total_documents (int): Total number of documents in the corpus.
        
    Returns:
        dict: A nested dictionary where keys are terms, and values are dictionaries with document IDs as keys 
              and TF-IDF scores as values.
    """
    tf_idf_scores = defaultdict(dict)

    for term, data in inverted_index.items():
        doc_ids = data["doc_ids"]
        term_tfs = data["tf"]
        df = data["df"]
        
        idf = math.log2(total_documents / df)
        
        for doc_id, tf in zip(doc_ids, term_tfs):
            tf_idf_scores[term][doc_id] = tf * idf

    return dict(tf_idf_scores)

print("TF-IDF SCORES")
print()
tf_idf = calculate_tf_idf(inverted_index, len(combined_documents))
print(tf_idf)
print()

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
