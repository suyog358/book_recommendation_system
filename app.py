import pickle
import streamlit as st
import numpy as np


st.header("Book Recommendation System by Machine learning")
model = pickle.load(open("artifacts/model.pkl", 'rb'))
books_name = pickle.load(open("artifacts/books_name.pkl", 'rb'))
final_rating = pickle.load(open("artifacts/final_rating.pkl", 'rb'))
book_pivot = pickle.load(open("artifacts/book_pivot.pkl", 'rb'))


def fecth_poster(suggestion):
    books_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        books_name.append(book_pivot.index[book_id])

    for name in books_name[0]:
        ids = np.where(final_rating['Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['Image-URL-M']
        poster_url.append(url)
    return poster_url
def recommend_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    prediction = model.kneighbors(book_pivot)
    distances, suggestions = model.kneighbors(book_pivot, n_neighbors=6)

    poster_url = fecth_poster(suggestions)

    for i in range(len(suggestions)):
        books = book_pivot.index[suggestions[i]]
        for j in books:
            book_list.append(j)
    return book_list, poster_url


selected_books = st.selectbox(
    "Type or Select a book",
    books_name
)

if st.button('Show Recommendation'):
    recommendation_book, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendation_book[1])
        st.image(poster_url[1])

    with col2:
        st.text(recommendation_book[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommendation_book[3])
        st.image(poster_url[3])

    with col4:
        st.text(recommendation_book[4])
        st.image(poster_url[4])

    with col5:
        st.text(recommendation_book[5])
        st.image(poster_url[5])
