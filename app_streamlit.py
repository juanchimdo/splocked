import streamlit as st
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bsp


#'https://s3-us-west-2.amazonaws.com/flx-editorial-wordpress/wp-content/uploads/2018/03/13153742/RT_300EssentialMovies_700X250.jpg'

def get_reviews(url):
    response = requests.get(url)
    soup = bsp(response.content, "html.parser")
    reviews = []
    for comment in soup.find_all("div", class_="lister-item-content"):
        titles = comment.find("a", class_="title").string.rstrip('\n').strip(' ')
        comments = comment.find_all("div", class_='text')
        for cmt in comments:
            reviews.append({'title':titles, 'comment': cmt.text})
    return pd.DataFrame(reviews)

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

  #model = load_model('\models\saved_model.pb')

  url = st.text_input("Type the IMDB movie review URL here: ",\
   "https://www.imdb.com/title/tt8134470/reviews?ref_=tt_urv")
  df = get_reviews(url)
  st.write(df)


if __name__ == "__main__":
    #df = read_data()
    main()
