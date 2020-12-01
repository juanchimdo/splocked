import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup as bsp
from splocked.predict import preprocess_review
from splocked.utils import get_reviews, imdb_api


def predict(df, model, word_to_id):
  """
  predicts the probability of being a spoiler and
  returns a new df column with the results per comment
  """

  preprocessed_reviews = [preprocess_review(review, word_to_id)\
  for review in df['title'] + ' ' + df['comment']]

  df['spoiler_proba'] = [model.predict(review)[0][0]*100\
  for review in preprocessed_reviews]

  return df['spoiler_proba']


# General Styling for Webpage
CSS = """

body {
  background-size: cover;
    color: #783252;
    background-color: #EDE9E4;
}
    """

header_img_url = 'https://s3-us-west-2.amazonaws.com/flx-editorial-wordpress/wp-content/uploads/2018/03/13153742/RT_300EssentialMovies_700X250.jpg'

HEADER_CSS = f"""
#header {{
    background-image: url('{header_img_url}');
    width: 100%;
    height: 200px;
}}
.center {{
  display: flex;
  justify-content: center;
  align-items: center;
}}
#header h1 {{
  color: white;
}}
"""

HEADER_HTML = f"""
<style>
    {HEADER_CSS}
</style>

<div id='header' class='center'>
    <h1>
        SPLOCKED!
    </h1>
</div>
"""

st.write(HEADER_HTML, unsafe_allow_html=True)

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("### *Check your movie's reviews without spoilers!*")

def main():

  model = load_model(os.path.join('model_baseline'))

  with open('word_to_id.json') as json_file:
      word_dict = json.load(json_file)

  movie_title = st.text_input("Type the IMDB movie title here: ",\
   "Movie title HERE!")

  st.write('OR...')

  movie_url = st.text_input("Type the IMDB movie url here: ",\
   "https://www.imdb.com/title/tt1411697/reviews?ref_=tt_urv")

  if movie_title != "Movie title HERE!":
    try:  
      imdbID = imdb_api(movie_title)
      url = f'https://www.imdb.com/title/{imdbID}/reviews?ref_=tt_urv'
      df = get_reviews(url)
      df['spoiler_proba'] = predict(df, model, word_dict)
      st.write(df)

    except:
      st.write('Wrong name! Or, try another movie!')

  else:
    df = get_reviews(movie_url)
    df['spoiler_proba'] = predict(df, model, word_dict)
    st.write(df)

if __name__ == "__main__":
    #df = read_data()
    main()
