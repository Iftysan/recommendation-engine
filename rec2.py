from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
DATA_DIR = Path(__file__).resolve().parent
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
movies_df['year'] = movies_df.title.str.extract(r'\((\d{4})\)', expand=False)
movies_df['title'] = movies_df.title.str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

genre_dummies = movies_df['genres'].str.get_dummies(sep='|')
movies_copy = pd.concat([movies_df, genre_dummies], axis=1)
GENRES = list(genre_dummies.columns)

user_input = [
    {'title' : 'Jurassic Park', 'rating' : 3.0},
    {'title' : 'Zero Effect', 'rating' : 3.5},
    {'title' : 'Jumanji', 'rating' : 4.5},
    {'title' : 'Toy Story', 'rating' : 4.5}
]

movies_input = pd.DataFrame(user_input)
bad_rating = movies_input[(movies_input['rating'] < 0.5) | (movies_input['rating'] > 5.0)]
if not bad_rating.empty:
    raise ValueError(f"Ratings must be  between 0.5 and 5.0:\n{bad_rating}")
print("Your ratings:")
print(movies_input, "\n")

movies_input = pd.merge(movies_df[['movieId', 'title']], movies_input, on='title')

missing = set(t['title'] for t in user_input) - set(movies_input['title'])
if missing:
    print(f"Warning: not found in movies.csv (ignored): {sorted(missing)}\n")
    if movies_input.empty:
        raise SystemExit("None of your titles were found in movies.csv - check the spelling.")
    
print("Your ratings matched to movie IDs:")
print(movies_input, "\n")

movies_user = movies_input.merge(movies_copy[['movieId'] + GENRES], on='movieId')
UserGenreTable = movies_user[GENRES]
UserProfile = UserGenreTable.transpose().dot(movies_user['rating'])

print("Your taste profile (genre weights):")
print(UserProfile.sort_values(ascending=False), "\n")
profile_sorted = UserProfile.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(profile_sorted.index, profile_sorted.values, color='#FF6B6B')
plt.title('Your Taste Profile')
plt.xlabel("Genre")
plt.ylabel("Weight")
plt.xticks(rotation = 45, ha='right')
plt.tight_layout()
plt.show()

GenreTable = movies_copy.set_index('movieId')[GENRES]
Recommendation_df = (GenreTable * UserProfile).sum(axis=1) / UserProfile.sum()
Recommendation_df = Recommendation_df.sort_values(ascending=False)
top20 = Recommendation_df.head(20)
RecommendationTable = movies_df.set_index('movieId').loc[top20.index][['title', 'year', 'genres']]
RecommendationTable['score'] = top20.round(3)
print("Top 20 recommended:")
print(RecommendationTable.reset_index(), "\n")

top10 = RecommendationTable.head(10)
labels = top10['title'] + ' (' + top10['year'].fillna('?') + ')'

plt.figure(figsize=(9, 6))
plt.barh(labels [::-1], top10['score'][::-1], color='#51CF66')
plt.title('Top 10 picks')
plt.xlabel("Math score")
plt.tight_layout()
plt.show()

