import json
import config

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

