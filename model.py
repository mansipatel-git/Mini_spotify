# =====================================================
# MODEL.PY (GMM + RECOMMENDATION)
# =====================================================

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("spotify_songs_real.csv")

df = df[['track_name', 'artist_name', 'tempo', 'energy',
         'danceability', 'loudness', 'valence']]

df = df.dropna()

# Features
features = df[['tempo','energy','danceability','loudness','valence']]

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Train GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X)

# Assign clusters
df['cluster'] = gmm.predict(X)

# Recommendation function
def recommend(song_name, n=5):
    song = df[df['track_name'].str.lower() == song_name.lower()]

    if song.empty:
        return None

    cluster = song.iloc[0]['cluster']

    recs = df[df['cluster'] == cluster]
    recs = recs[recs['track_name'] != song.iloc[0]['track_name']]

    return recs[['track_name','artist_name']].sample(n).values.tolist()