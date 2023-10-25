

import pandas
import nltk
import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

 #To open the file

file = open("Harry_Potter_and_the_Sorcerers_Stone.txt", encoding='utf-8')

listofwords=file.read().splitlines()

listofwords =[i for i in listofwords if i!=''] 
text = " "

text = text.join(listofwords)

#PREPROCESSING THE TEXT FILE 

# Define a function to preprocess the text
def preprocess_text(input_file, output_file):
    # Define a set of punctuation characters to be filtered out
    punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    try:
        # Open the input file for reading
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        # Remove punctuation and convert to lowercase
        text = ''.join([char.lower() for char in text if char not in punctuations])

        # Open the output file for writing
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)

        print("Text preprocessing complete. Filtered punctuation and converted to lowercase.")
    except FileNotFoundError:
        print(f"File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


input_file = "Harry_Potter_and_the_Sorcerers_Stone.txt"
output_file = "preprocessed_Harry_Potter_and_the_Sorcerers_Stone.txt"

# Preprocess the text
preprocess_text(input_file, output_file)


nltk.download('punkt')

# TOKENIZATION 

def tokenize(input_file, output_file):
    # reading the preprocessed text
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # tokenizing the text into words
    words = word_tokenize(text)
    tokenized_text = ' '.join(words)

    # writing the tokenized text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(tokenized_text)
    
    print(words)

    print("Tokenization completed.")

# Input and output file paths
input_file = "without_stopwords.txt"
output_file = "tokenized_without_stopwords.txt"

# Tokenize and remove stop words
tokenize(input_file, output_file)





import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# Ensure that NLTK's Punkt tokenizer models are downloaded
nltk.download('punkt')

# Read the text from a file
file_path = "tokenized_without_stopwords.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into words
words = word_tokenize(text)

# Calculate the frequency distribution of tokens
freq_dist = FreqDist(words)

# Get the most common tokens and their frequencies
most_common = freq_dist.most_common(20)  # Change 20 to the desired number of tokens to display

# Plot the frequency distribution
freq_dist.plot(30, cumulative=False)  # Change 30 to the desired number of tokens to display

# Print the most common tokens and their frequencies
for token, frequency in most_common:
    print(f"{token}: {frequency}")
    

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from collections import Counter

nltk.download('averaged_perceptron_tagger')

# Perform PoS tagging using the Penn Treebank tag set
tags = nltk.pos_tag(words)

tags[:10]







