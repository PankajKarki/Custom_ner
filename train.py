import spacy
import random
from tqdm import tqdm
from pathlib import Path

import config
from dataset import load_data


def clean(data):
  """ This function removes the lines with empty entities
  Args:
      data: a dictionary in spacy training format

  Returns:
      train_data: a clean dictionary
  """
  train_data = []
  for i in data['annotations']:
    k = i[1]['entities']
    if len(k) != 0:
      train_data.append(i)
  return train_data


def ner(train_data, model, n_iter, output_dir):
  """ This function is for training and saving model
  Args:
      train_data: a dictionary in spacy training format
      model: spacy pretrained model
      n_iter: number of iteration for training
      output_dir: path for saving traind model

  Returns:
      trained model
  """

  if model is not None:
      nlp = spacy.load(model)  
      print("Loaded model '%s'" % model)
  else:
      nlp = spacy.blank('en')  
      print("Created blank 'en' model")

  if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
  else:
    ner = nlp.get_pipe('ner')

  for _, annotations in train_data:
      for ent in annotations.get('entities'):
          ner.add_label(ent[2])

  other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
  with nlp.disable_pipes(*other_pipes):  # only train NER
      optimizer = nlp.begin_training()
      for itn in range(n_iter):
          random.shuffle(train_data)
          losses = {}
          for text, annotations in tqdm(train_data):
              nlp.update(
                  [text],  
                  [annotations],  
                  drop=0.2,   
                  sgd=optimizer,
                  losses=losses)
          print(losses)
          
  if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

if __name__ == "__main__":
    data = load_data(config.TRAINING_DATA)
    print(".....loading dataset")
    train_data = clean(data)
    print(".....training")
    ner(train_data, config.MODEL, config.N_ITER, config.OUTPUT_DIR)

