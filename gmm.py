# =====================================================
# MINI SPOTIFY - FROM SCRATCH (GMM + EM)
# =====================================================

import pandas as pd
import numpy as np

# -----------------------------
# STEP 1: Load Data
# -----------------------------
df = pd.read_csv("spotify_songs_real.csv")

df = df[['track_name', 'artist_name', 'tempo', 'energy',
         'danceability', 'loudness', 'valence']]

df = df.dropna()

# -----------------------------
# STEP 2: Feature Matrix
# -----------------------------
X = df[['tempo','energy','danceability','loudness','valence']].values

n, d = X.shape
K = 3   # number of clusters

# -----------------------------
# STEP 3: Manual Normalization
# -----------------------------
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X - mean) / std

# -----------------------------
# STEP 4: Initialize Parameters
# -----------------------------
np.random.seed(42)

# Random means
mu = X[np.random.choice(n, K, False)]

# Identity covariance
Sigma = [np.eye(d) for _ in range(K)]

# Equal weights
pi = np.ones(K) / K

# -----------------------------
# Gaussian Function
# -----------------------------
def gaussian(x, mean, cov):
    d = len(x)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    num = np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))
    denom = np.sqrt((2*np.pi)**d * det)

    return num / denom

# -----------------------------
# STEP 5: EM Algorithm
# -----------------------------
def EM(X, mu, Sigma, pi, iterations=20):
    n, d = X.shape
    K = len(mu)

    for _ in range(iterations):

        # -------- E-STEP --------
        gamma = np.zeros((n, K))

        for i in range(n):
            for k in range(K):
                gamma[i, k] = pi[k] * gaussian(X[i], mu[k], Sigma[k])

            gamma[i, :] /= np.sum(gamma[i, :])

        # -------- M-STEP --------
        for k in range(K):
            Nk = np.sum(gamma[:, k])

            # Update mean
            mu[k] = np.sum(gamma[:, k].reshape(-1,1) * X, axis=0) / Nk

            # Update covariance
            diff = X - mu[k]
            Sigma[k] = (gamma[:, k].reshape(-1,1) * diff).T @ diff / Nk

            # Update weights
            pi[k] = Nk / n

    return mu, Sigma, pi, gamma

# Run EM
mu, Sigma, pi, gamma = EM(X, mu, Sigma, pi)

# -----------------------------
# STEP 6: Assign Clusters
# -----------------------------
clusters = np.argmax(gamma, axis=1)
df['cluster'] = clusters

# -----------------------------
# STEP 7: Recommendation
# -----------------------------
def recommend(song_name):
    song = df[df['track_name'].str.lower() == song_name.lower()]

    if song.empty:
        return "Song not found"

    cluster = song.iloc[0]['cluster']
    recs = df[df['cluster'] == cluster]

    recs = recs[recs['track_name'] != song.iloc[0]['track_name']]

    return recs[['track_name','artist_name']].head(5)

# -----------------------------
# TEST
# -----------------------------
print("Recommendations for Kesariya:")
print(recommend("Kesariya"))