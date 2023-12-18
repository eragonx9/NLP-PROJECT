import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import sys

nltk.download('punkt')  # Download the NLTK punkt tokenizer
nltk.download('stopwords')  # Download the NLTK stopwords

# Specify the directory containing text files
input_directory = "chapters"  # Replace with the actual path to your directory

# Get a list of all text files in the directory
text_files = [f for f in os.listdir(input_directory) if f.endswith(".txt")]

if not text_files:
    print(f"Error: No text files found in the directory '{input_directory}'.")
    sys.exit(1)

# Process each text file
for file_name in text_files:
    # Construct the full path to the text file
    input_file_path = os.path.join(input_directory, file_name)

    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        continue

    # Tokenize the text into chapters using NLTK sentence tokenizer
    chapters = nltk.sent_tokenize(text)

    # Remove stopwords from each chapter
    stop_words = set(stopwords.words('english'))
    preprocessed_chapters = []
    for chapter in chapters:
        tokens = word_tokenize(chapter)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        preprocessed_chapters.append(" ".join(filtered_tokens))

    # Create a TF-IDF vectorizer for each chapter
    vectorizers = [TfidfVectorizer() for _ in preprocessed_chapters]

    # Fit and transform each chapter separately
    tfidf_matrices = [vectorizer.fit_transform([preprocessed_chapter]) for vectorizer, preprocessed_chapter in zip(vectorizers, preprocessed_chapters)]

    # Get the feature names (terms) for the first chapter
    feature_names = vectorizers[0].get_feature_names_out()

    # Print the TF-IDF matrix for each chapter
    for i, (chapter_tfidf, vectorizer) in enumerate(zip(tfidf_matrices, vectorizers)):
        print(f"\nFile: {file_name} - Chapter {i + 1} - TF-IDF Vectors:")
        dense_array = chapter_tfidf.toarray()

        # Print each word and its TF-IDF value on a separate line
        for term, tfidf_value in zip(feature_names, dense_array[0]):
            print(f"{term}: {tfidf_value}")

        # Optional: Save TF-IDF matrix for each chapter to a file
        output_tfidf_file_path = f"output_tfidf_matrix_{file_name}_chapter_{i + 1}.txt"  # Replace with the desired output file path
        with open(output_tfidf_file_path, 'w', encoding='utf-8') as output_tfidf_file:
            output_tfidf_file.write(f"File: {file_name} - Chapter {i + 1} - TF-IDF Vectors:\n")
            dense_array = chapter_tfidf.toarray()
            for j, (term, tfidf_value) in enumerate(zip(feature_names, dense_array[0])):
                output_tfidf_file.write(f"{term}: {tfidf_value}\n")

    print(f"\nTF-IDF matrix for each chapter in file '{file_name}' saved to individual files.")

