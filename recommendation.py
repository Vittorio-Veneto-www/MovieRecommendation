import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import Dict, Literal, Iterable

# Load the movie dataset
df = pd.read_csv("data/all.csv")

# It will take ~20GB memory and at least 2 minutes if the entire movie dataset is loaded
# If you wish to try the complete dataset, comment out the following 2 lines
# The following lines preserve specific movies and add a random sample of 5000 movies from the dataset
preserve_list = ["The Dark Knight Rises", "Avatar"]
df = pd.concat([df.loc[df["tmdb_title"].isin(preserve_list)], df.sample(5000, random_state=1)]).drop_duplicates().reset_index()

# Helper function to split a comma-separated string and strip whitespace
def split_str(s: str):
    return map(lambda s: s.strip(), s.split(","))

# Function to calculate the Jaccard similarity between two sets
def jaccardSim(d1: Iterable, d2: Iterable):
    s1, s2 = set(d1), set(d2)
    return len(s1 & s2) / len(s1 | s2)

# Function to calculate the Gaussian similarity between two numerical values
def gaussian(x: np.ndarray, y: float, sigma: float):
    return math.e ** (-(x - y) ** 2 / sigma ** 2 / 2)

# Compute the TF-IDF matrix for movie overviews
tfidf = TfidfVectorizer(stop_words='english')
df['tmdb_overview'] = df['tmdb_overview'].fillna('')  # Replace NaN values with empty strings
tfidf_matrix = tfidf.fit_transform(df['tmdb_overview'])

# Compute the cosine similarity matrix for movie overviews
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a series to map movie titles to their corresponding indices
indices = pd.Series(df.index, index=df['tmdb_title']).drop_duplicates()

# Function to get similarity scores based on the overview of a given movie
def get_score_overview(idx: int):
    return cosine_sim[idx]

# Load reviews data and compute TF-IDF matrix for reviews
df_reviews = pd.read_csv("data/rotten_reviews.csv", index_col="id")["reviewText"]
tfidf_matrix_reviews = TfidfVectorizer().fit_transform(df["rotten_id"].apply(lambda x: df_reviews.get(x, "")))

# Compute the cosine similarity matrix for reviews
cosine_sim_reviews = linear_kernel(tfidf_matrix_reviews, tfidf_matrix_reviews)

# Function to get similarity scores based on reviews
def get_score_reviews(idx: int):
    return cosine_sim_reviews[idx]

# Function to get similarity scores based on genres using Jaccard similarity
def get_score_genres(idx: int):
    lst = []
    genres_target = split_str(df["tmdb_genres"][idx])
    for i in range(len(df)):
        genres = split_str(df["tmdb_genres"][i])
        lst.append(jaccardSim(genres, genres_target))
    return np.array(lst)

# Function to get similarity scores based on release year, considering movies within a 5-year range as similar
def get_score_year(idx: int):
    lst = []
    year_target = int(df["tmdb_release_date"][idx][:4])
    for i in range(len(df)):
        lst.append(abs(int(df["tmdb_release_date"][i][:4]) - year_target) <= 5)
    return np.array(lst)

# Function to get similarity scores based on runtime using Gaussian similarity
def get_score_length(idx: int):
    return gaussian(df["tmdb_runtime"], df["tmdb_runtime"][idx], 20)

# Function to get similarity scores based on revenue using Gaussian similarity
def get_score_revenue(idx: int):
    return gaussian(df["tmdb_revenue"], df["tmdb_revenue"][idx], 1000000000)

# Function to get similarity scores based on IMDb score using Gaussian similarity
def get_score_imdb_score(idx: int):
    return gaussian(df["imdb_averageRating"], df["imdb_averageRating"][idx], 0.8)

# Function to get similarity scores based on Tomatometer score using Gaussian similarity
def get_score_tomatometer(idx: int):
    return gaussian(df["rotten_tomatoMeter"], df["rotten_tomatoMeter"][idx], 5)

# Fill NaN values in keywords with empty strings
df["tmdb_keywords"].fillna("", inplace=True)

# Function to get similarity scores based on keywords using Jaccard similarity
def get_score_keywords(idx: int):
    lst = []
    keywords_target = split_str(df["tmdb_keywords"][idx])
    for i in range(len(df)):
        keywords = split_str(df["tmdb_keywords"][i])
        lst.append(jaccardSim(keywords, keywords_target))
    return np.array(lst)

# Fill NaN values in production countries with empty strings
df["tmdb_production_countries"].fillna("", inplace=True)

# Function to get similarity scores based on production countries using Jaccard similarity
def get_score_country(idx: int):
    lst = []
    countries_target = split_str(df["tmdb_production_countries"][idx])
    for i in range(len(df)):
        countries = split_str(df["tmdb_production_countries"][i])
        lst.append(jaccardSim(countries, countries_target))
    return np.array(lst)

# Fill NaN values in original language with empty strings
df["rotten_originalLanguage"].fillna("", inplace=True)

# Function to get similarity scores based on language using Jaccard similarity
def get_score_language(idx: int):
    lst = []
    languages_target = split_str(df["rotten_originalLanguage"][idx])
    for i in range(len(df)):
        languages = split_str(df["rotten_originalLanguage"][i])
        lst.append(jaccardSim(languages, languages_target))
    return np.array(lst)

# Create a registry that maps factor names to their corresponding scoring functions
registry = {
    "overview": get_score_overview,
    "reviews": get_score_reviews,
    "imdb_score": get_score_imdb_score,
    "tomatometer": get_score_tomatometer,
    "keywords": get_score_keywords,
    "genres": get_score_genres,
    "country": get_score_country,
    "length": get_score_length,
    "revenue": get_score_revenue,
    "year": get_score_year,
    "language": get_score_language,
}

# Function to get recommendations based on selected factors and their weights
def get_recommendations(title, factors):
    idx = indices[title]  # Get the index of the selected movie
    # Calculate the weighted similarity scores for each factor
    sim_scores = list(enumerate(sum(map(lambda factor, weight: registry[factor](idx) * weight, factors.keys(), factors.values()))))
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first result (the movie itself) and get the top 10
    movie_indices = [i[0] for i in sim_scores]  # Extract movie indices
    return idx, df.iloc[movie_indices]  # Return the index of the input movie and the recommended movies
