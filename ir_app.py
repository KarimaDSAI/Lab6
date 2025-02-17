import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
nltk.download("reuters")
nltk.download("punkt")
nltk.download("stopwords")

# Load Stopwords
stop_words = set(stopwords.words("english"))

# Function to preprocess text
def preprocess_text(text):
    """Tokenizes and removes stop words."""
    return [
        word.lower()
        for word in nltk.word_tokenize(text)
        if word.isalnum() and word.lower() not in stop_words
    ]

# Load and preprocess Reuters dataset
st.write("Loading Reuters dataset...")
corpus_sentences = [
    preprocess_text(reuters.raw(fileid)) for fileid in reuters.fileids()
]
st.write(f"Loaded {len(corpus_sentences)} documents from Reuters.")

# Train Word2Vec Model
st.write("Training Word2Vec model...")
model = Word2Vec(
    sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4
)
st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")

# Function to compute document embeddings
def compute_average_embedding(words, model):
    """Computes the average Word2Vec embedding for a given document or query."""
    embeddings = [model.wv[word] for word in words if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Compute embeddings for each document
document_embeddings = np.array(
    [compute_average_embedding(doc, model) for doc in corpus_sentences]
)

# Function to retrieve top relevant documents
def retrieve_documents(query, top_n=5):
    """Finds top-N most relevant documents for a given query using cosine similarity."""
    query_tokens = preprocess_text(query)
    query_embedding = compute_average_embedding(query_tokens, model)

    # Compute cosine similarity with all documents
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # Rank documents by similarity score
    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return ranked_docs[:top_n]

# Streamlit App UI
st.title("🔍 Information Retrieval System")
st.subheader("Search Reuters News Articles Using Word Embeddings")

# User input query
query = st.text_input("Enter your search query:")

if query:
    st.write(f"Searching for: **{query}**...")
    top_documents = retrieve_documents(query, top_n=5)

    st.write("### 📄 Top Relevant Documents:")
    for doc_id, score in top_documents:
        st.write(f"**Document {doc_id}** (Score: {score:.4f})")
        st.write(" ".join(corpus_sentences[doc_id][:50]) + "...")
        st.write("---")
