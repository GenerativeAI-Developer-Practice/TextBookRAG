import streamlit as st
import time

st.title("RAG UI Prototype")
# about_you = st.text_area("Enter about yourself")
# birth_date = st.date_input("Enter your birthday")
# cars_count = st.slider("How many cars do you have", min_value=0, max_value=10)
# if st.button("submit"):
#     st.write(f"Good to hear about you. \n{about_you} \n{birth_date}")

with st.sidebar:
    with st.echo():
        st.write("This code will be printed to the sidebar.")

    with st.spinner("Loading..."):
        time.sleep(5)
    st.success("Done!")