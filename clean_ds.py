import pandas as pd
import numpy as np

def parse_rank_and_genre(rank_genre_str):
    if not isinstance(rank_genre_str, str) or not rank_genre_str.strip():
        return {}, [], ""
    
    categories = rank_genre_str.split(',')
    rank_dict = {}
    genre_list = []
    remarks = []
    
    for cat in categories:
        cat = cat.strip()
        remarks.append(cat)
        
        if "#" in cat and " in " in cat:
            parts = cat.split(" in ", 1)  # Split only once to avoid errors
            rank_part = parts[0].replace("#", "").strip()
            category = parts[1].strip()
            
            if rank_part.isdigit():  # Ensure rank is a valid integer
                rank_dict[category] = int(rank_part)
                genre_list.append(category)
    
    return rank_dict, genre_list, ', '.join(remarks)

# Load & Merge Data
df1 = pd.read_csv(r"C:\Users\ashwi\GUVI_Projects\Book\Audible_Catalog.csv", encoding="utf-8-sig")
df2 = pd.read_csv(r"C:\Users\ashwi\GUVI_Projects\Book\Audible_Catalog_Advanced_Features.csv", encoding="utf-8-sig")

df = pd.merge(df1, df2, on=["Book Name", "Author"], how="inner")
df.drop_duplicates(subset=["Book Name", "Author"], keep="first", inplace=True)

# Handle Number of Ratings (use 'Number of Reviews_x' as 'Number of Ratings')
if "Number of Reviews_x" in df.columns:
    df["Number of Ratings"] = df["Number of Reviews_x"]
    df.drop(columns=["Number of Reviews_x"], inplace=True)

# Handle Price Column (same logic as Rating, ensuring 'Price' is numeric)
if "Price_x" in df.columns and "Price_y" in df.columns:
    df["Price"] = df["Price_x"].fillna(df["Price_y"])
    df.drop(columns=["Price_x", "Price_y"], inplace=True)
elif "Price_x" in df.columns:
    df.rename(columns={"Price_x": "Price"}, inplace=True)
elif "Price_y" in df.columns:
    df.rename(columns={"Price_y": "Price"}, inplace=True)

# Ensure Price is numeric and fill NaN values
df["Price"] = pd.to_numeric(df.get("Price"), errors="coerce").fillna(df["Price"].mean())

# Ensure Rating Column
if "Rating_x" in df.columns and "Rating_y" in df.columns:
    df["Rating"] = df["Rating_x"].fillna(df["Rating_y"])
    df.drop(columns=["Rating_x", "Rating_y"], inplace=True)
elif "Rating_x" in df.columns:
    df.rename(columns={"Rating_x": "Rating"}, inplace=True)
elif "Rating_y" in df.columns:
    df.rename(columns={"Rating_y": "Rating"}, inplace=True)

# Ensure Rating is numeric and fill NaN values
df["Rating"] = pd.to_numeric(df.get("Rating"), errors="coerce").fillna(df["Rating"].mean())

# Split 'Ranks and Genre' column
df[['Rank', 'Genre', 'Remarks']] = df['Ranks and Genre'].apply(lambda x: pd.Series(parse_rank_and_genre(x)))
df['Rank'] = df['Rank'].apply(str)
df['Genre'] = df['Genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Drop unwanted columns
df.drop(columns=["Number of Reviews_y"], inplace=True)

# Select only the required columns
df_cleaned = df[['Book Name', 'Author', 'Description', 'Listening Time', 'Number of Ratings', 'Price', 
                 'Rating', 'Rank', 'Genre']]

# Save to CSV
df_cleaned.to_csv("cleaned_books_data.csv", index=False)

print(df_cleaned.head())
row = df_cleaned.iloc[0]

# Print each column value in a separate line
for column, value in row.items():
    print(f"{column}: {value}")