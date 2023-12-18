import spacy
import sys


nlp = spacy.load("en_core_web_sm")


if len(sys.argv) > 1:
    file_path = sys.argv[1]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tokenized_text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
else:
    print("Usage: python entity.py /path/to/your/tokenized_file.txt")
    sys.exit(1)

doc = nlp(tokenized_text)

entity_categories = {
    'PER': 'People',
    'ORG': 'Organization',
    'LOC': 'Location',
    'GPE': 'Geo-Political Entity',
    'FAC': 'Facility',
    'VEH': 'Vehicles'
}

# Extracting entities
entities_with_types = [(ent.text, ent.label_) for ent in doc.ents]

# Print the identified entities and their types
print("\nIdentified Entities and Types:")
for entity, entity_type in entities_with_types:
    category = entity_categories.get(entity_type, 'Unknown')
    print(f"Entity: {entity}, Type: {entity_type} ({category})")

# Extracting unique entity types
unique_entity_types = set(entity_type for _, entity_type in entities_with_types)

# Print unique entity 
print("\nUnique Entity Types:")
for entity_type in unique_entity_types:
    print(entity_type)
