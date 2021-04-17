import spacy
from pathlib import Path

output_dir=Path.cwd()

nlp = spacy.load(output_dir)


text = """For the spice mix I have eight guajillo chelates for flavor and four chiles de arbol to bring some heat from which I'm both going to remove the stems and seeds. I'm also gonna add about two teaspoons of whole cumin seeds"""
print()
print(text)
print()


doc = nlp(text)
for ent in doc.ents:
  print(ent.text)

