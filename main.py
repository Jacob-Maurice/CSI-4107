import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json

def load_corpus(file_path, num_lines=5):
    """
    Load and inspect the corpus file.

    Args:
        file_path (str): Path to the `corpus.jsonl` file.
        num_lines (int): Number of lines to display for inspection.

    Returns:
        list: A list of parsed JSON objects from the corpus.
    """
    corpus = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            document = json.loads(line)
            corpus.append(document)
    return corpus

file_path = "scifact/corpus.jsonl"
sample_corpus = load_corpus(file_path, num_lines=5)

for i, doc in enumerate(sample_corpus):
    print(f"Document {i + 1}:")
    print(json.dumps(doc, indent=4)) 
    print()

# Download required NLTK data files (uncomment if running for the first time)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_document(text, use_stemming=False):
    """
    Preprocesses a document by tokenizing, removing stopwords, and optionally stemming.

    Args:
        text (str): The input text to preprocess.
        use_stemming (bool): Whether to apply stemming using PorterStemmer.

    Returns:
        list: A list of preprocessed tokens.
    """
    # Initialize stopwords and stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer() if use_stemming else None

    # Tokenize text (split into words)
    tokens = word_tokenize(text.lower())  # Convert to lowercase for uniformity

    # Remove punctuation, numbers, and stopwords
    tokens = [
        re.sub(r'\W+', '', token)  # Remove non-alphanumeric characters
        for token in tokens if token.isalpha() and token not in stop_words
    ]

    # Apply stemming if enabled
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def preprocess_corpus(file_path, use_stemming=False):
    """
    Preprocesses all documents in a corpus file.

    Args:
        file_path (str): Path to the `corpus.jsonl` file.
        use_stemming (bool): Whether to apply stemming using PorterStemmer.

    Returns:
        dict: A dictionary with document IDs as keys and preprocessed tokens as values.
    """
    processed_corpus = {}

    # Read JSONL file line by line
    with open(file_path, 'r') as f:
        for line in f:
            document = json.loads(line)
            doc_id = document['doc_id']  # Assuming `doc_id` is the identifier
            text = document['text']      # Assuming `text` contains the content

            # Preprocess the text
            tokens = preprocess_document(text, use_stemming)
            processed_corpus[doc_id] = tokens

    return processed_corpus
