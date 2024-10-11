import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple Inc. was founded by Steve Jobs in California in 1976."

# Process the text
doc = nlp(text)

# Print the recognized entities
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
