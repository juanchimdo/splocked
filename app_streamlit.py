import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd


#'https://s3-us-west-2.amazonaws.com/flx-editorial-wordpress/wp-content/uploads/2018/03/13153742/RT_300EssentialMovies_700X250.jpg'



CSS = """

body {
  background-size: cover;
    color: #783252;
    background-color: #EDE9E4;
}
    """

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("# **SPLOCKED !**")
st.markdown("### *Check if your friend's comment contain spoiler!\
 Or, supply an URL and find out if the comments contain any spoilers!*")


def main():

  #model = load_model('\models\saved_model.pb')
  text = st.text_input("Type the comment here: ", "OMG, I can't believe McDreamy died!")

  url = st.text_input("Or...type the URL here: ",\
   "https://www.imdb.com/title/tt8134470/reviews?ref_=tt_urv")


  choice = st.radio('Are you sure you would like to know?', ('Yes', 'No'))
  if choice == 'Yes':
    pred = model.predict(X)
    if pred[1] > 0.5:
      st.write("SPOILER ALLERT!!!!")
    else:
      st.write("Nhá, you can read it!")
  else:
    st.write("Why are you here then?!")

if __name__ == "__main__":
    #df = read_data()
    main()
