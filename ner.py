import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

file_path = "tokenized_without_stopwords.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()


words = word_tokenize(text)

pos_tags = pos_tag(words)

# Perform NER
ner_result = ne_chunk(pos_tags)

#extract named entities
conll_tags = tree2conlltags(ner_result)


for i in range(len(conll_tags)):
    word, pos, tag = conll_tags[i]
    if tag != 'O':
        print(f"[{word}/{tag}]", end=' ')
    else:
        print(f"[{word}/{pos}]", end=' ')
print()  





from spacy import displacy

# Render the spaCy visualization
displacy.render(doc, style="ent", jupyter=True)
