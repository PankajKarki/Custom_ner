import spacy
import config
from pathlib import Path

def predict(model_dir, text):
  nlp = spacy.load(model_dir)
  doc = nlp(text)
  list_ingredients = []
  for ent in doc.ents:
    list_ingredients.append(ent.text)
  return list_ingredients

#if __name__ == "__main__":
#  text = """For the spice mix I have eight guajillo chelates for flavor and four chiles de arbol to bring some heat from which #I'm both going to remove the stems and seeds. I'm also gonna add about two teaspoons of whole cumin seeds"""
#  print(predict(config.OUTPUT_DIR, text))




