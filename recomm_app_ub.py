import pandas as pd
import numpy as np
import ast
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Configure Streamlit for full width
st.set_page_config(layout="wide")

# Load Dataimport pandas as pd

df = pd.read_csv("/home/ubuntu/Book-Recommendation/cleaned_books_data.csv", encoding="utf-8-sig")


# Standardize column names
df.columns = df.columns.str.strip()

# Handle missing values
df["Description"].fillna("No description available", inplace=True)
df["Number of Ratings"].fillna(df["Number of Ratings"].mean(), inplace=True)
df["Genre"].fillna("Unknown", inplace=True)
df["Rating"].fillna(df["Rating"].mean(), inplace=True)
df["Rating"].replace(-1.0, df["Rating"].mean(), inplace=True)
df["Listening Time"].fillna("0 hrs and 0 mins", inplace=True)

# Convert 'Listening Time' to minutes
def convert_listening_time(time_str):
    if pd.isna(time_str) or time_str.strip() == "":
        return 0  
    time_str = time_str.lower().strip()
    match = re.match(r"(\d+)\s*(?:hrs?|hours?)?\s*(?:and)?\s*(\d+)?\s*(?:mins?|minutes?)?", time_str)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    match = re.match(r"(\d+)\s*(?:hrs?|hours?)", time_str)
    if match:
        return int(match.group(1)) * 60
    match = re.match(r"(\d+)\s*(?:mins?|minutes?)", time_str)
    if match:
        return int(match.group(1))
    return 0  

df["Listening Time"] = df["Listening Time"].apply(convert_listening_time)

# Ensure 'Number of Ratings' is numeric
df["Number of Ratings"] = pd.to_numeric(df["Number of Ratings"], errors="coerce")

# Extract Publication Year (If Available)
if "Publication Year" in df.columns:
    df["Publication Year"] = pd.to_numeric(df["Publication Year"], errors="coerce")

# Vectorization & Clustering
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["Description"])
tfidf_matrix_normalized = normalize(tfidf_matrix)

# KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(tfidf_matrix_normalized)

# Streamlit UI
st.title("📚 Book Recommendation System")

# **Input Section**
st.subheader("🎯 Select Your Preferences")

col1, col2, col3 = st.columns(3)

# Select Genre
selected_genre = col1.selectbox("Select a Genre", options=[""] + sorted(df["Genre"].dropna().unique().tolist()))

# Filter Authors based on Genre selection
if selected_genre:
    filtered_authors = df[df["Genre"] == selected_genre]["Author"].dropna().unique().tolist()
else:
    filtered_authors = df["Author"].dropna().unique().tolist()

selected_author = col2.selectbox("Select an Author", options=[""] + sorted(filtered_authors))

# Filter Books based on Genre & Author selection
if selected_author:
    filtered_books = df[(df["Genre"] == selected_genre) & (df["Author"] == selected_author)]["Book Name"].dropna().unique().tolist()
else:
    filtered_books = df["Book Name"].dropna().unique().tolist()

selected_book = col3.selectbox("Select a Book", options=[""] + sorted(filtered_books))

# **Button to Show Book Details**
if st.button("📖 Show Book Details"):
    if selected_book:
        book_details = df[df["Book Name"] == selected_book]
        if not book_details.empty:
            st.subheader(f"📘 Details for: {selected_book}")
            st.write(f"**Author:** {book_details.iloc[0]['Author']}")
            st.write(f"**Description:** {book_details.iloc[0]['Description']}")
            st.write(f"**Listening Time:** {book_details.iloc[0]['Listening Time']} mins")
            st.write(f"**Number of Ratings:** {book_details.iloc[0]['Number of Ratings']}")
            st.write(f"**Price:** ₹{book_details.iloc[0]['Price']}")
            st.write(f"**Rating:** ⭐ {book_details.iloc[0]['Rating']:.1f}")
            st.write(f"**Rank:** {book_details.iloc[0]['Rank']}")
            st.write(f"**Genre:** {book_details.iloc[0]['Genre']}")
        else:
            st.warning("No details found for the selected book.")
    else:
        st.warning("Please select a book to view details.")

# **Container to Show Recommendations After Button Click**
if st.button("🔍 Get Recommendations"):
    def filter_and_recommend(df, book_name, num_recommendations=5):
        cosine_sim = cosine_similarity(tfidf_matrix_normalized)
        content_recs = df[df["Book Name"] != book_name].sample(num_recommendations)

        st.subheader("📚 Top 5 Recommendations:")
        st.dataframe(content_recs[["Book Name", "Author", "Genre", "Price", "Rating"]])

        # **Show EDA Visualizations After Button Click**
        st.subheader("📊 Book Trends & Insights")

        # 1️⃣ **Distribution of Ratings**
        fig, ax = plt.subplots()
        sns.histplot(df["Rating"], bins=20, kde=True, ax=ax)
        ax.set_title("📈 Ratings Distribution")
        st.pyplot(fig)

        # 2️⃣ **Top 10 Most Popular Genres (Excluding Unknown)**
        fig, ax = plt.subplots()
        top_genres = df[df["Genre"] != "Unknown"]["Genre"].value_counts().nlargest(10)
        sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax)
        ax.set_title("🏆 Most Popular Genres")
        st.pyplot(fig)

        # 3️⃣ **Top Authors with Most Books**
        fig, ax = plt.subplots()
        top_authors = df["Author"].value_counts().nlargest(10)
        sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax)
        ax.set_title("✍️ Top Authors with Most Books")
        st.pyplot(fig)

        # 4️⃣ **Word Cloud of Book Titles**
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Book Name"].dropna()))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("📖 Word Cloud of Book Titles")
        st.pyplot(fig)


    filter_and_recommend(df, selected_book)
