# Extract the list of ingredients

![Page](assets/predict.png?raw=true)


# Inference:

## Sample Text for prediction:

```sh
"For the spice mix I have eight guajillo chelates for flavor and four chiles de arbol to bring some heat from which I'm both going to remove the stems and seeds. I'm also gonna add about two teaspoons of whole cumin seeds"
```

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run stream.py
```

# Training:

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python train.py
```
