import pandas as pd
import numpy as np
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

# Load Data
df = pd.read_csv("/home/ubuntu/Book-Recommendation/cleaned_books_data.csv", encoding="utf-8-sig")

# Standardize column names
df.columns = df.columns.str.strip()

# Handle missing values
df["Description"].fillna("No description available", inplace=True)
df["Number of Ratings"].fillna(df["Number of Ratings"].mean(), inplace=True)
df["Genre"].fillna("Missing Genre", inplace=True)
df["Rating"].fillna(df["Rating"].mean(), inplace=True)
df["Rating"].replace(-1.0, df["Rating"].mean(), inplace=True)
df["Listening Time"].fillna("0 hrs and 0 mins", inplace=True)

# Convert 'Listening Time' to minutes
def convert_listening_time(time_str):
    if pd.isna(time_str) or time_str.strip() == "":
        return 0  
    match = re.match(r"(\d+)\s*(?:hrs?|hours?)?\s*(?:and)?\s*(\d+)?\s*(?:mins?|minutes?)?", time_str)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
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
st.title(" Book Recommendation System")

# **Input Section**
st.subheader(" Select Your Preferences")

col1, col2, col3 = st.columns(3)
selected_genre = col1.selectbox("Select a Genre", options=[""] + sorted(df["Genre"].dropna().unique().tolist()))
selected_author = col2.selectbox("Select an Author", options=[""] + sorted(df[df["Genre"] == selected_genre]["Author"].dropna().unique().tolist()))
selected_book = col3.selectbox("Select a Book", options=[""] + sorted(df[(df["Genre"] == selected_genre) & (df["Author"] == selected_author)]["Book Name"].dropna().unique().tolist()))
if selected_book:

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs([" Book Details", " Recommendations", " Insights & Trends"])

    # **Tab 1: Show Selected Book Details**
    with tab1:
        if selected_book:
            book_details = df[df["Book Name"] == selected_book].iloc[0]
            st.subheader(f" Details for: {selected_book}")
            st.write(f"**Author:** {book_details['Author']}")
            st.write(f"**Description:** {book_details['Description']}")
            st.write(f"**Listening Time:** {book_details['Listening Time']} mins")
            st.write(f"**Number of Ratings:** {book_details['Number of Ratings']}")
            st.write(f"**Price:** ₹{book_details['Price']}")
            st.write(f"**Rating:**  {book_details['Rating']:.1f}")
            st.write(f"**Rank:** {book_details['Rank']}")
            st.write(f"**Genre:** {book_details['Genre']}")
        else:
            st.warning("Please select a book to view details.")

    # **Tab 2: Show Recommendations**
    with tab2:
        if selected_book:
            cosine_sim = cosine_similarity(tfidf_matrix_normalized)
            content_recs = df[df["Book Name"] != selected_book].sample(5)
            st.subheader(" Top 5 Recommendations:")
            st.dataframe(content_recs[["Book Name", "Author", "Genre", "Number of Ratings","Description", "Rating", "Listening Time", "Price"]])
        else:
            st.warning("Select a book to get recommendations.")

    with tab3:
        
        subtab1, subtab2, subtab3 = st.tabs([" Ratings Analysis", " Genre Insights", " Other Trends"])
        
        # Ratings Analysis Subtab
        with subtab1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            
            if not df["Rating"].isna().all():
                sns.histplot(df["Rating"].dropna(), bins=20, kde=True, ax=axes[0])
                axes[0].set_title("Ratings Distribution")
            else:
                axes[0].text(0.5, 0.5, "No Ratings Data", ha='center', va='center')
            
            sns.scatterplot(x=df["Number of Ratings"], y=df["Rating"], alpha=0.5, ax=axes[1])
            axes[1].set_title("Ratings vs. Review Counts")
            
            sns.lineplot(x=df.index, y=df["Rating"], ax=axes[2])
            axes[2].set_title("Rating Trend Over Index")
            
            st.pyplot(fig)
        
        # Genre Insights Subtab
        with subtab2:
            genre_option = st.selectbox("Select Genre Analysis Type:", ["Top Genres", "Rating Distribution by Genre", "Genre Popularity"])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if genre_option == "Top Genres":
                top_genres = df["Genre"].value_counts().nlargest(10)
                sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax)
                ax.set_title("Top 10 Most Popular Genres")
                ax.tick_params(axis='x', rotation=90)
            
            elif genre_option == "Rating Distribution by Genre":
                top_genres = df["Genre"].value_counts().nlargest(10).index
                filtered_df = df[df["Genre"].isin(top_genres)]
                sns.boxplot(y="Genre", x="Rating", data=filtered_df, ax=ax)
                ax.set_title("Rating Distribution by Genre")
                ax.tick_params(axis='y', rotation=0)
        
            elif genre_option == "Genre Popularity":
                top_genres = df["Genre"].value_counts().nlargest(5).index
                genre_counts = df[df["Genre"].isin(top_genres)]["Genre"].value_counts()
                sns.barplot(y=genre_counts.index, x=genre_counts.values, ax=ax)
                ax.set_title("Genre Popularity")
                ax.tick_params(axis='y', rotation=0)
            
            st.pyplot(fig)
        
        # Other Trends Subtab
        with subtab3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            
            sns.heatmap(df[["Rating", "Number of Ratings", "Price"]].corr(), annot=True, cmap="coolwarm", ax=axes[0])
            axes[0].set_title("Correlation Heatmap")
            
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Book Name"].dropna()))
            axes[1].imshow(wordcloud, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title("Word Cloud of Book Titles")
            
            sns.histplot(df["Listening Time"], bins=20, kde=True, ax=axes[2])
            axes[2].set_title("Listening Time Distribution")
            
            st.pyplot(fig)

            
            
                
