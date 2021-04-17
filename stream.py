import predict
import config
import streamlit as st

text = st.text_area("Enter Your text", "Type Here ...")

if(st.button('Submit')):
    result = predict.predict(config.OUTPUT_DIR, text)
    for ele in result:
      st.success(ele)
