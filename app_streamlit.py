import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup as bsp
from splocked.predict import preprocess_review
from splocked.utils import imdb_api, get_reviews


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
  background-color: rgb(0,0,0);
}
.block-container {
  background-color: #EDE9E4;
  margin: 10px 0px;
  padding: 1rem 1rem 1rem !important;
  border-radius: 5px;
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


REVIEW_CSS = """
summary::-webkit-details-marker {
 color: #00ACF3;
 font-size: 100%;
 margin-right: 2px;
 position: absolute;
 top: 50px;
 right: 5px;
 z-index: 100;
}
summary:focus {
  outline-style: none;
}
article > details > summary {
  font-size: 28px;
  margin-top: 16px;
}
details > p {
  margin-left: 24px;
}
details details {
  margin-left: 36px;
}
details details summary {
  font-size: 16px;
}
.title {
  color: black;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  padding: 16px 80px;
  border-radius: 5px;
  margin-bottom: 15px;
  background-color: #fff;
}

.title:hover{
    margin-top: -10px;
    box-shadow: 0 4px 2px -2px rgba(0,0,0,0.8);
}

.green {
  box-shadow: 0 4px 2px -2px rgba(20,200,50,0.5);
  -webkit-transition: margin 0.2s ease-out;
  -moz-transition: margin 0.2s ease-out;
  -o-transition: margin 0.2s ease-out;
  transition: margin 0.2s ease-out;
}

.red {
  box-shadow: 0 4px 2px -2px rgba(255,0,50,0.5);
  -webkit-transition: margin 0.2s ease-out;
  -moz-transition: margin 0.2s ease-out;
  -o-transition: margin 0.2s ease-out;
  transition: margin 0.2s ease-out;
}

.relative {
  position: relative;
  z-index: 10
}
.bottom {
  position: absolute;
  top: 40px;
  right: 25px;
}
.no-margin {
  margin: 0px 0px 0px 0px;
}
.bar {
  position: absolute;
  top: 25px;
  left: 140px;
  height: 5px;
  width: 150px;
  z-index: 1000;
}

.emptybar {
  background-color: #2e3033;
  width: 100%;
  height: 100%;
  z-index:1001;
}

.filledbar {
  position: absolute;
  top: 0px;
  z-index: 1001;
  width: 0px;
  height: 100%;
  background: rgb(0,154,217);
  background: linear-gradient(90deg, rgba(0,154,217,1) 0%, rgba(217,147,0,1) 65%, rgba(255,186,0,1) 100%);
  transition: 0.6s ease-out;
}

.relative:hover .filledbar {
  width: 120px;
  transition: 0.4s ease-out;
}
"""

REVIEW_CARD = """
<div class='relative'>
  <details>
    <summary>
      <div class='title {color}'>
          <div>
            <div>
              <div class="bar">
                <div class="emptybar"></div>
                <div class="filledbar"></div>
              </div>
              <div><em>{spoiler_proba}%<em></div>
              <div>{title}</div>
            </div>
          </div>
          <div class="bottom mr-5 ">
            <p>read full review</p>
          </div>
      </div>
    </summary>
    <p>
      {comment}
    </p>
  </details>
</div>
"""

#st.write(NICE_CARD, unsafe_allow_html=True)

def main():

  model = load_model(os.path.join('gru_model'))

  with open('gru_word_to_id.json') as json_file:
      word_dict = json.load(json_file)

  movie_title = st.text_input("Type the IMDB movie title here: ",\
   "Movie title here")

  if movie_title != 'Movie title here':
      try:
          imdbID = imdb_api(movie_title)
          df = get_reviews(imdbID)
          df['spoiler_proba'] = predict(df, model, word_dict)
          reviews = ''.join([REVIEW_CARD.format(title=row['title'], comment=row['comment'], spoiler_proba=round(row['spoiler_proba'], 2), color="green" if row['spoiler_proba'] < 50 else "red") for index, row in df.iterrows()])
          REVIEW_HTML = f"""
          <style>
            {REVIEW_CSS}
          </style>
          {reviews}
          """
          st.write(REVIEW_HTML, unsafe_allow_html=True)
      except:
          st.write("Try another movie title")


if __name__ == "__main__":
    main()
