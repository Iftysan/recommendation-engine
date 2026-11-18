from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data_DIR = Path(__file__).resolve().parent

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

ratings_df = pd.read_csv(Data_DIR / 'ratings.csv')
print("Ratings Data:")
print(ratings_df.head(), "\n")

movies_df = pd.read_csv(Data_DIR / 'movies.csv')
print("Movies Data:")
print(ratings_df.head(), "\n")

movies_df['year'] = movies_df.title.str.extract(r'(\(\d\d\d\d\))', expand=True)
movies_df['year'] = movies_df.year.str.extract(r'(\d\d\d\d)', expand=True)

movies_df['title'] = movies_df.title.str.replace(r'(\(\d\d\d\d\))', '', regex=True)
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print("Movies with cleaned titles and year column:")
print(movies_df.head(), "\n")

movies_df['genres'] = movies_df.genres.str.split('|')

movies_copy = movies_df.copy()
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_copy.at[index, genre] = 1

movies_copy = movies_copy.fillna(0)
print("Movies with one-hot encoded genres:")
print(movies_copy.head(), "/n")

genre_columns = movies_copy.drop(['movieId', 'title', 'genres', 'year'], axis=1)
genre_counts  = genre_columns.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(genre_counts.index, genre_counts.values, color='#4C9AFF')
plt.title('Movies Per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()