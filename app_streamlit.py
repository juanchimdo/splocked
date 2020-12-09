import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup as bsp
from splocked.predict import preprocess_review
from splocked.utils import imdb_api, get_reviews, movie_info


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
  color: rgb(70,70,70);
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
st.markdown("<link href='https://fonts.googleapis.com/css?family=Material+Icons|Material+Icons+Outlined|Material+Icons+Two+Tone|Material+Icons+Round|Material+Icons+Sharp' rel='stylesheet'>", unsafe_allow_html=True)
st.write(SPLOCKED_TITLE, unsafe_allow_html=True)
st.write("<i class='fas fa-star'></i>", unsafe_allow_html=True)
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
  box-shadow: 0 4px 2px -2px rgba(50,50,50,0.5);
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
  background-color: rgba(225,225,225,1);
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
  border-top: 1px solid rgb(200,200,200);
  width: 100%;
}
.padding-bottom {
  padding: 0px 0px 10px 0px;
}
.padding-top {
  padding: 10px 0px 0px 0px;
}
.padding-sides {
  padding: 5px 60px;
}
.yellow {
  color: yellow;
  -webkit-text-stroke: 1px rgb(50, 50, 50);
}
.grey {
  color: rgb(150, 150, 150);
  -webkit-text-stroke: 1px rgb(50, 50, 50);
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
              <div class='padding-bottom'><big>{title}</big></div>
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
    <div class="padding-sides">
      <p>{comment}</p>
    </div>
  </details>
</div>
"""

#st.write(NICE_CARD, unsafe_allow_html=True)
def define_color(number):
    if number <= 25:
        return "#17BA14"
    if number <= 50:
        return "yellow"
    if number <= 75:
        return "orange"
    if number <= 100:
        return "red"

def process_rating(rating):
  if np.isnan(rating):
    return ' '
  if rating == 1:
    return "<i class='material-icons-round yellow'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i>"
  if rating == 2:
    return "<i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i>"
  if rating == 3:
    return "<i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round grey'>star</i> <i class='material-icons-round grey'>star</i>"
  if rating == 4:
    return "<i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round grey'>star</i>"
  if rating == 5:
    return "<i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i> <i class='material-icons-round yellow'>star</i>"
MOVIE_INFO_CSS = """
.banner {
  border-radius: 5px;
  box-shadow: 0 4px 2px -2px rgba(50,50,50,0.5);
}
.flex-sa {
  display: flex;
  justify-content: space-around;
}
.text-align-center {
  text-align: center;
}
.margin-around {
  margin: 30px 60px;
}
"""

MOVIE_INFO = """
<div class="flex-sa margin-around">
  <div>
    <img class="banner" src={image_url}, width="180">
  </div>
  <div class="text-align-center">
    <h1> {movie_title} </h1>
    <p style="padding: 5px;"> {summary_txt} </p>
  </div>
</div>
"""

model = load_model(os.path.join('gru_model'))

with open('gru_word_to_id.json') as json_file:
    word_dict = json.load(json_file)

def main():

  movie_title = st.text_input("",\
   "Movie title here")
  exclude_spoilers = st.checkbox("Exclude Spoilers")

  if movie_title != 'Movie title here':
      #try:
      imdbID = imdb_api(movie_title)
      df = get_reviews(imdbID)
      df['spoiler_proba'] = predict(df, model, word_dict)
      if exclude_spoilers:
          df = df[df['spoiler_proba'] < 50]
      df = df.sort_values(by=['spoiler_proba'])
      #movie_title, summary_txt, image_url = movie_info(imdbID)
      #movie = MOVIE_INFO.format(movie_title=movie_title, summary_txt=summary_txt, image_url=image_url)
      reviews = ''.join([REVIEW_CARD.format(title=row['title'], comment=row['comment'], spoiler_proba=round(row['spoiler_proba'], 2), color=define_color(row['spoiler_proba']), bar_width=(150*row['spoiler_proba'])/100, rating=process_rating(row['rating'])) for index, row in df.iterrows()])
      MOVIE_HTML = f"""
      <style>
        {MOVIE_INFO_CSS}
      </style>
      {movie}
      """
      REVIEW_HTML = f"""
      <style>
        {REVIEW_CSS}
      </style>
      {reviews}
      """
      st.write(MOVIE_HTML, unsafe_allow_html=True)
      st.write(REVIEW_HTML, unsafe_allow_html=True)
      #except:
          #st.write("Try another movie title")


if __name__ == "__main__":
    main()
