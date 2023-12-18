from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')  # Download the NLTK punkt tokenizer
nltk.download('stopwords')  # Download the NLTK stopwords

# Specify the directory containing preprocessed text files
preprocessed_directory = "chapters"

# Read preprocessed text from each file in the directory
preprocessed_documents = []
for filename in os.listdir(preprocessed_directory):
    file_path = os.path.join(preprocessed_directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            preprocessed_documents.append(text)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

# Remove stopwords from each document
stop_words = set(stopwords.words('english'))
preprocessed_documents_filtered = []
for document in preprocessed_documents:
    tokens = word_tokenize(document)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    preprocessed_documents_filtered.append(" ".join(filtered_tokens))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed documents
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_filtered)

# Compute cosine similarity between each pair of documents
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a DataFrame for the similarity matrix
columns = [f"Chapter {i + 1}" for i in range(len(preprocessed_documents))]
similarity_df = pd.DataFrame(similarity_matrix, columns=columns, index=columns)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw a heatmap with the numeric values in each cell
sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="viridis")

plt.title("Cosine Similarity Between Chapters")
plt.show()

