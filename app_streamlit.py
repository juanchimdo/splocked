import streamlit as st
import os


st.markdown("""
<style>
body {
    color: #fff;
    background-color: #6495ED;
}

</style>
    """, unsafe_allow_html=True)


st.markdown("# **SPLOCKED !**")
st.markdown("### *Check if your friend's comment contain spoiler!\
 Or, supply an URL and find out if the comments contain any spoilers!*")


def main():


	st.text_input("Type the comment here: ", "OMG, I can't believe McDreamy died!")
	
	st.text_input("Or...type the URL here: ",\
	 "https://www.imdb.com/title/tt8134470/reviews?ref_=tt_urv")

	choice = st.radio('Are you sure you would like to know?', ('Yes', 'No'))
	if choice == 'Yes':
		st.write("SPOILER ALLERT!!!!")
	else:
		st.write("nhá, you can read it!")


if __name__ == "__main__":
    #df = read_data()
    main()
