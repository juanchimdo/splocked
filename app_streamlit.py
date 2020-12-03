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
CSS = f"""

body {{
  background-size: cover;
  color: #783252;
  background-color: #156153;
}}
.block-container {{
  background-color: #EDE9E4;
  margin: 10px 0px;
  padding: 1rem 1rem 1rem !important;
  border-radius: 5px;
}}
    """

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

SPLOCKED_TITLE = """
<style>
  #splocked-title {
    font-family: 'Alfa Slab One', cursive;
    -webkit-text-stroke: 1px black;
    font-size: 100px;
    color: #E50000;
    text-align: center;
    margin: 0px 0px !important;
    padding: 0px 0px !important;
  }
</style>
<h1 id=splocked-title>SPLOCKED! </h1>
"""
st.markdown("<link rel='preconnect' href='https://fonts.gstatic.com'> <link href='https://fonts.googleapis.com/css2?family=Alfa+Slab+One&display=swap' rel='stylesheet'>", unsafe_allow_html=True)
st.write(SPLOCKED_TITLE, unsafe_allow_html=True)

# header_img_url = 'https://s3-us-west-2.amazonaws.com/flx-editorial-wordpress/wp-content/uploads/2018/03/13153742/RT_300EssentialMovies_700X250.jpg'

# HEADER_CSS = f"""
# #header {{
#     background-image: url('{header_img_url}');
#     width: 100%;
#     height: 200px;
# }}
# .center {{
#   display: flex;
#   justify-content: center;
#   align-items: center;
# }}
# #header h1 {{
#   color: white;
# }}
# """

# HEADER_HTML = f"""
# <style>
#     {HEADER_CSS}
# </style>

# <div id='header' class='center'>
#     <h1>
#         SPLOCKED!
#     </h1>
# </div>
# """

# st.write(HEADER_HTML, unsafe_allow_html=True)

# st.markdown("### *Check your movie's reviews without spoilers!*")


REVIEW_CSS = """
summary::-webkit-details-marker {
 color: #00ACF3;
 font-size: 150%;
 position: absolute;
 top: 100px;
 right: 80px;
 z-index: 100;
 border-radius: 5px;
}
.bottom {
  position: absolute;
  top: 95px;
  right: 120px;
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
  box-shadow: 0 4px 2px -2px rgba(50,50,50,0.5)
}

.green {
}

.red {
}

.relative {
  position: relative;
  z-index: 10
}
.no-margin {
  margin: 0px 0px 0px 0px;
}
.bar {
  position: absolute;
  top: 10px;
  right: 90px;
  height: 5px;
  width: 150px;
  z-index: 1000;
}

.emptybar {
  background-color: rgba(200,200,200,0.5);
  width: 100%;
  height: 100%;
  z-index:1001;
}

.filledbar {
  position: absolute;
  top: 0px;
  z-index: 1001;
  height: 100%;
  transition: 0.6s ease-out;
}

.text-align-right {
  width: 100%;
  text-align: right;
}

.text-align-left {
  width: 100;
  text-align: left;
}

.border-top {
  border-top: 1px solid rgb(100,100,100);
  width: 100%;
}
.padding-bottom {
  padding: 0px 0px 10px 0px;
}
.padding-top {
  padding: 10px 0px 0px 0px;
}
"""

REVIEW_CARD = """
<div class='relative'>
  <details>
    <summary>
      <div class='title'>
          <div class='bar'>
              <div class="emptybar"></div>
              <div class="filledbar" style="background: {color}; width: {bar_width}px"></div>
          </div>
          <div class='text-align-right'>
            <div class='padding-bottom'><strong>Spoiler-Meter {spoiler_proba}%<strong></div>
          <div>
          <div class='text-align-left'>
              <div class='padding-bottom'>{title}</div>
          </div>
          <div class="border-top padding-top">
            <div class="text-align-left">
              <p> {rating} </p>
            </div>
            <div class="bottom mr-5">
              <p>read full review</p>
            </div>
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
def define_color(number):
    if number <= 25:
        return "green"
    if number <= 50:
        return "yellow"
    if number <= 75:
        return "orange"
    if number <= 100:
        return "red"

def process_rating(rating):
  if np.isnan(rating):
    return ' '
  else:
    return f"{round(rating)}/5"

def main():

  model = load_model(os.path.join('gru_model'))

  with open('gru_word_to_id.json') as json_file:
      word_dict = json.load(json_file)

  movie_title = st.text_input("",\
   "Movie title here")

  if movie_title != 'Movie title here':
      try:
          imdbID = imdb_api(movie_title)
          df = get_reviews(imdbID)
          df['spoiler_proba'] = predict(df, model, word_dict)
          reviews = ''.join([REVIEW_CARD.format(title=row['title'], comment=row['comment'], spoiler_proba=round(row['spoiler_proba'], 2), color=define_color(row['spoiler_proba']), bar_width=(150*row['spoiler_proba'])/100, rating=process_rating(row['rating'])) for index, row in df.iterrows()])
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
