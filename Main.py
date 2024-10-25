import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import time

start = time.time()


def recommendations(title):
    try:
        i_d = []
        indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()
        idx = indices[title]
        dis_scores = list(enumerate(model[idx]))
        dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)
        dis_scores = dis_scores[1:31]
        idn = [k[0] for k in dis_scores]
        final = df1.iloc[idn].reset_index()
        idn = [k for k in final['index']]
        for j in idn:
            if j < 15951:
                i_d.append(j)
        indices = pd.Series(movie_df.index, index=movie_df['title']).drop_duplicates()
        for y in range(1, 8):
            if idn:
                print(indices.iloc[i_d].index[y])
    except:
        print('Film not found')


# Creation of the dataframe and formatting it into a usable dataframe run this code once and store into a pickle file
# This is because the reading of the csv file takes around 200 seconds to run and therefore would be too slow.
# However, if we turn the dataframe after it's been formatted into a pickle file we can reuse this without having
# to read the csv file again.

# movies = pd.read_csv('movie.csv')
# rating = pd.read_csv('rating.csv' )
# movie_details = movies.merge(rating, on='movieId')
# movie_details.drop(columns=['timestamp'], inplace=True)
# total_ratings = movie_details.groupby(['movieId', 'genres']).sum()['rating'].reset_index()
# df = movie_details.copy()
# df.drop_duplicates(['title', 'genres'], inplace=True)
# df = df.merge(total_ratings, on='movieId')
# df.drop(columns=['userId', 'rating_x', 'genres_y'], inplace=True)
# df.rename(columns={'genres_x': 'genres',  'rating_y': 'rating'}, inplace=True)
# df['rating'] = df['rating'].astype(int)
# pd.to_pickle(df, "New_movie_dataframe.pkl")

movie_df = pd.read_pickle("New_movie_dataframe.pkl")
tfv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
x = tfv.fit_transform(movie_df['genres'])
model = sigmoid_kernel(x, x)
df1 = movie_df.copy()
ti = []
for i in df1['title']:
    ti.append(i.split(' (')[0])

df1['title'] = ti

user_profile = ['Interstellar', 'Jurassic Park']
for movies in user_profile:
    print(f'Recommendation for {movies}')
    recommendations(movies)
    print("")

end = time.time()
print(f'Time taken: {end - start}')
