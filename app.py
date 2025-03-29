import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Appliquer un style personnalisé avec un fond vert foncé et du texte blanc
st.markdown(
    """
    <style>
    body {
        background-color: #0b3d0b;
        color: white;
    }
    .stApp {
        background-color: #0b3d0b;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# 📌 Charger les données
def load_data():
    artists = pd.read_csv("datasets/artists_gp1.dat", delimiter="\t")
    user_artists = pd.read_csv("datasets/user_artists_gp1.dat", delimiter="\t")
    
    # Transformation logarithmique
    user_artists["weight"] = np.log1p(user_artists["weight"])
    
    # Convertir les IDs en string
    artists["id"] = artists["id"].astype(str)
    user_artists["userID"] = user_artists["userID"].astype(str)
    user_artists["artistID"] = user_artists["artistID"].astype(str)
    
    return artists, user_artists

# Charger les données
artists, user_artists = load_data()

# 📌 Interface utilisateur
st.title("🎶 Recommandation Musicale")

# 🔹 Demander l'ID utilisateur AVANT de l'utiliser
user_id = st.text_input("Veuillez entrer votre ID utilisateur :")

# 📌 Chansons populaires associées aux artistes
popular_songs = {
    "Lady Gaga": ["Bad Romance", "Poker Face", "Shallow"],
    "Britney Spears": ["...Baby One More Time", "Toxic", "Oops!... I Did It Again"],
    "Rihanna": ["Umbrella", "Diamonds", "We Found Love"],
    "Katy Perry": ["Firework", "Roar", "Dark Horse"],
    "The Beatles": ["Hey Jude", "Let It Be", "Come Together"],
    "Madonna": ["Like a Virgin", "Vogue", "Hung Up"],
    "Christina Aguilera": ["Genie in a Bottle", "Beautiful", "Fighter"],
    "Avril Lavigne": ["Complicated", "Sk8er Boi", "Girlfriend"],
    "Paramore": ["Misery Business", "Ain't It Fun", "Still Into You"],
    "Radiohead": ["No Surprises", "Creep", "Exit Music"]
}

# 🔹 Vérifier si l'ID utilisateur est renseigné avant d'exécuter le reste du code
if user_id:
    user_history = user_artists[user_artists["userID"] == user_id]
    
    if not user_history.empty:
        # Identifier les 10 artistes les plus écoutés
        top_10_artists = (
            user_artists.groupby("artistID")["weight"]
            .sum()
            .nlargest(11)
            .reset_index()
        )
        
        top_10_artists = top_10_artists.merge(
            artists, left_on="artistID", right_on="id"
        )[["artistID", "name", "weight"]]

        st.subheader("🎵 Top 10 artistes les plus écoutés")

        # 📌 Affichage horizontal des artistes populaires avec les chansons en infobulle
        st.write("Faites défiler vers la droite pour voir tous les artistes ⏩")
        cols = st.columns(len(top_10_artists))

        # 🔹 Création de l'affichage interactif
        for col, (index, row) in zip(cols, top_10_artists.iterrows()):
            artist_name = row['name']
            songs = popular_songs.get(artist_name, ["Chanson 1", "Chanson 2", "Chanson 3"])
            tooltip_text = f"🎶 {songs[0]}\n🎶 {songs[1]}\n🎶 {songs[2]}"

            with col:
                st.markdown(
                    f'<div title="{tooltip_text}" style="cursor: pointer; padding: 10px; text-align: center; border-radius: 10px; border: 1px solid white; background-color: #1E1E1E;">'
                    f'<strong>{index + 1} : {artist_name}</strong>'
                    '</div>',
                    unsafe_allow_html=True
                )

        # 📌 Recommandations
        user_artist_matrix = user_artists.pivot(index="userID", columns="artistID", values="weight").fillna(0)
        
        if user_id in user_artist_matrix.index:
            similarity_matrix = cosine_similarity(user_artist_matrix)
            user_index = list(user_artist_matrix.index).index(user_id)
            similar_users = np.argsort(similarity_matrix[user_index])[::-1][1:5]
            similar_user_ids = [user_artist_matrix.index[i] for i in similar_users]
            similar_users_data = user_artists[user_artists["userID"].isin(similar_user_ids)]
            
            recommended_artists = (
                similar_users_data.groupby("artistID")["weight"]
                .sum()
                .nlargest(5)
                .reset_index()
            )
            
            recommended_artists = recommended_artists.merge(
                artists, left_on="artistID", right_on="id"
            )[["artistID", "name"]]
            
            st.subheader("🎧 Recommandations pour vous")
            for index, row in recommended_artists.iterrows():
                st.markdown(f"- {row['name']}")
        else:
            st.warning(f"L'utilisateur {user_id} n'est pas dans la matrice, impossible de générer des recommandations.")
    else:
        st.warning(f"L'utilisateur {user_id} n'a pas d'historique d'écoute.")