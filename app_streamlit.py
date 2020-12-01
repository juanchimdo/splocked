import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup as bsp
from splocked.predict import preprocess_review
from splocked.utils import get_reviews


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
  padding: 5px;
  border-radius: 5px;
  margin-bottom: 10px;
}

.green {
  background-color: #99EA9A;
  border: 5px solid #78B779;
}

.red {
  background-color: #FFA7A7;
  border: 5px solid #FF7272;
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
"""

REVIEW_CARD = """\
<div class='relative'>
  <details>
    <summary>
      <div class='title {color}'>
          <div>
            <div>
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

#st.write(REVIEW_CARD, unsafe_allow_html=True)
#st.write(REVIEW_CARD, unsafe_allow_html=True)

def main():

  model = load_model(os.path.join('model_baseline'))

  with open('word_to_id.json') as json_file:
      word_dict = json.load(json_file)

  url = st.text_input("Type the IMDB movie review URL here: ",\
   "https://www.imdb.com/title/tt8134470/reviews?ref_=tt_urv")

  df = get_reviews(url)

  df['spoiler_proba'] = predict(df, model, word_dict)

  reviews = ''.join([REVIEW_CARD.format(title=row['title'], comment=row['comment'], spoiler_proba=round(row['spoiler_proba'], 2), color="green" if row['spoiler_proba'] < 50 else "red") for index, row in df.iterrows()])

  REVIEW_HTML = f"""
  <style>
    {REVIEW_CSS}
  </style>
  <div>
    {reviews}
  <div>
  """

  st.write(REVIEW_HTML, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
