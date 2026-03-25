# =====================================================
# MINI SPOTIFY WITH UI (GMM + EM)
# =====================================================

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("spotify_songs_real.csv")

df = df[['track_name', 'artist_name', 'tempo', 'energy',
         'danceability', 'loudness', 'valence']]

df = df.dropna()

# ==============================
# FEATURE PROCESSING
# ==============================
features = df[['tempo', 'energy', 'danceability', 'loudness', 'valence']]

scaler = StandardScaler()
X = scaler.fit_transform(features)

# ==============================
# TRAIN GMM MODEL
# ==============================
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X)

df['cluster'] = gmm.predict(X)

# ==============================
# RECOMMEND FUNCTION
# ==============================
def recommend():
    song_name = entry.get()

    song = df[df['track_name'].str.lower() == song_name.lower()]

    if song.empty:
        messagebox.showerror("Error", "Song not found!")
        return

    cluster = song.iloc[0]['cluster']
    recs = df[df['cluster'] == cluster]

    recs = recs[recs['track_name'] != song.iloc[0]['track_name']]

    recs = recs[['track_name', 'artist_name']].sample(5)

    # Clear previous results
    listbox.delete(0, tk.END)

    # Show results
    for i, row in recs.iterrows():
        listbox.insert(tk.END, f"{row['track_name']} - {row['artist_name']}")

# ==============================
# UI DESIGN (TKINTER)
# ==============================
root = tk.Tk()
root.title("🎵 Mini Spotify Recommender")
root.geometry("500x400")
root.config(bg="#1DB954")  # Spotify green

# Title
title = tk.Label(root, text="🎧 MINI SPOTIFY", font=("Arial", 20, "bold"), bg="#1DB954", fg="white")
title.pack(pady=10)

# Entry box
entry = tk.Entry(root, width=40, font=("Arial", 12))
entry.pack(pady=10)

# Button
btn = tk.Button(root, text="Recommend Songs", command=recommend, bg="black", fg="white", font=("Arial", 12))
btn.pack(pady=10)

# Listbox for results
listbox = tk.Listbox(root, width=50, height=10, font=("Arial", 11))
listbox.pack(pady=10)

# Run app
root.mainloop()