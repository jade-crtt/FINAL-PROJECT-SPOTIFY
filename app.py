import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from sklearn.metrics.pairwise import cosine_similarity

# 🔧 Config de la page
st.set_page_config(page_title="Spotify", page_icon="🎷")

# 🎨 Styles CSS
st.markdown("""
<style>
/* 🌿 Fond général Spotify */
body, .stApp {
    background-color: #0b3d0b;
    font-family: 'Helvetica Neue', 'Arial', sans-serif;
}

/* 🖤 Sidebar */
section[data-testid="stSidebar"] {
    background-color: #121212 !important;
}

/* 🎵 Titres bien visibles */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* ✍️ Texte principal adouci */
p, span, li {
    color: #B3B3B3 !important;
    font-size: 16px;
}

/* 🧱 Forcer les titres dans les containers */
div[data-testid="stMarkdownContainer"] > h3 {
    color: #FFFFFF !important;
}
strong, b {
    color: #FFFFFF !important;
}

/* 🧭 Onglets */
div[data-testid="stTabs"] {
    background-color: #062d06;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid #1a5c1a;
    margin-bottom: 25px;
}
button[data-baseweb="tab"] {
    background-color: transparent;
    color: white;
    font-weight: bold;
    border-radius: 12px !important;
    margin: 4px;
    padding: 10px 18px;
    border: none;
    transition: all 0.2s ease-in-out;
}
button[data-baseweb="tab"]:hover {
    background-color: #1a5c1a;
    color: #1DB954;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #1DB954;
    color: black;
    font-weight: bold;
}

/* 🎤 Cartes artistes */
.artist-container {
    border: 1px solid #282828;
    border-radius: 12px;
    padding: 10px;
    transition: transform 0.3s ease;
    background-color: rgba(255, 255, 255, 0.03);
    min-height: 350px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.artist-container:hover {
    transform: scale(1.05);
    background-color: rgba(255, 255, 255, 0.05);
}
.ranking-number {
    font-size: 24px;
    font-weight: bold;
    color: #1DB954;
    margin-bottom: 10px;
}
.artist-name {
    margin-top: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    min-height: 50px;
    color: white;
}
.song-title {
    font-size: 14px;
    text-align: center;
    margin: 2px 0;
    line-height: 1.4;
}

/* 📩 Inputs */
input, textarea {
    background-color: #282828 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #1DB954 !important;
}

/* 🟩 Boutons */
button[kind="primary"] {
    background-color: #1DB954 !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# 🖼️ Logo principal
if os.path.exists("images/spotify.svg"):
    with open("images/spotify.svg", "r", encoding="utf-8") as f:
        svg_code = f.read()
    svg_code = svg_code.replace('width="', 'style="width:180px; max-width:100%;" ').replace('height="', '')
    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            {svg_code}
        </div>
    """, unsafe_allow_html=True)

# 📁 Chargement des données
@st.cache_data
def load_data():
    artists = pd.read_csv("datasets/artists_gp1.dat", delimiter="\t")
    user_artists = pd.read_csv("datasets/user_artists_gp1.dat", delimiter="\t")
    user_artists["weight"] = np.log1p(user_artists["weight"])
    artists["id"] = artists["id"].astype(str)
    user_artists["userID"] = user_artists["userID"].astype(str)
    user_artists["artistID"] = user_artists["artistID"].astype(str)
    return artists, user_artists

artists, user_artists = load_data()

# 🎵 Titres populaires
popular_songs = {
    "Lady Gaga": ["Bad Romance", "Poker Face", "Shallow"],
    "Britney Spears": ["Baby One More Time", "Toxic", "Oops! I Did It Again"],
    "Rihanna": ["Umbrella", "Diamonds", "We Found Love"],
    "Katy Perry": ["Firework", "Roar", "Dark Horse"],
    "The Beatles": ["Hey Jude", "Let It Be", "Come Together"],
    "Madonna": ["Like a Virgin", "Vogue", "Hung Up"],
    "Christina Aguilera": ["Genie in a Bottle", "Beautiful", "Fighter"],
    "Avril Lavigne": ["Complicated", "Sk8er Boi", "Girlfriend"],
    "Paramore": ["Misery Business", "Ain't It Fun", "Still Into You"],
    "Radiohead": ["No Surprises", "Creep", "Exit Music"]
}

# 📸 Récupération d'image
def get_artist_image(artist_name):
    filename = artist_name.lower().replace(" ", "_")
    for ext in ["jpeg", "jpg"]:
        path = os.path.join("images", f"{filename}.{ext}")
        if os.path.exists(path):
            return path
    return "images/placeholder.jpg"

# 🛝 Onglets
tab1, tab2 = st.tabs(["Home", "👤"])

# 🔥 Onglet HOME
with tab1:
    st.subheader("🔥 Most Popular Artists")
    st.markdown("These are the most listened-to artists at the moment, based on global user data.")
    top_10_artists = (
        user_artists.groupby("artistID")["weight"].sum().nlargest(11).reset_index()
    )
    top_10_artists = top_10_artists.merge(artists, left_on="artistID", right_on="id")[["artistID", "name", "weight"]]

    rank = 1
    for i in range(0, len(top_10_artists), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(top_10_artists):
                artist = top_10_artists.iloc[i + j]
                artist_name = artist["name"]
                img_path = get_artist_image(artist_name)
                songs = popular_songs.get(artist_name, [])

                with col:
                    image_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
                    html = f"""
                        <div class='artist-container'>
                            <div class='ranking-number'>{rank}</div>
                            <img src="data:image/jpeg;base64,{image_b64}" style="width: 100%; border-radius: 10px;"/>
                            <div class='artist-name'>{artist_name}</div>
                            {''.join([f"<p class='song-title'>{song}</p>" for song in songs if song])}
                        </div>
                    """
                    col.markdown(html, unsafe_allow_html=True)
                    rank += 1

# 👤 Onglet MY PROFILE
with tab2:
    with st.sidebar:
        st.markdown("<h2 style='color:white; margin-top: 20px;'>User Input</h2>", unsafe_allow_html=True)
        user_id = st.text_input("Enter your User ID", value="", label_visibility="visible")
        search_query = st.text_input("Search Artist", value="", placeholder="Type an artist name...")
        if search_query:
            filtered_artists = artists[artists['name'].str.lower() == search_query.strip().lower()]
            if not filtered_artists.empty:
                for _, row in filtered_artists.iterrows():
                    artist_id = row['id']
                    total_weight = user_artists[user_artists['artistID'] == artist_id]['weight'].sum()
                    st.markdown(f"**{row['name']}** - Global Listening Weight: `{total_weight:.2f}`")
                    if total_weight > 100:
                        st.caption("🎧 This artist is highly listened to globally.")
                    elif total_weight > 10:
                        st.caption("🎶 This artist has moderate popularity.")
                    else:
                        st.caption("🔍 This artist has relatively low listening activity.")
            else:
                st.info("No artist found with that exact name.")

    if user_id:
        user_history = user_artists[user_artists["userID"] == user_id]
        if not user_history.empty:
            st.subheader(f"👤 Profile of User: {user_id}")
            st.markdown("<h4 style='margin-top: 20px; color: white;'>🎵 Personalized Recommendations</h4>", unsafe_allow_html=True)
            st.markdown("These recommendations are based on your similarity with other users.")

            user_artist_matrix = user_artists.pivot(index="userID", columns="artistID", values="weight").fillna(0)

            if user_id in user_artist_matrix.index:
                similarity_matrix = cosine_similarity(user_artist_matrix)
                user_index = list(user_artist_matrix.index).index(user_id)
                similar_users = np.argsort(similarity_matrix[user_index])[::-1][1:6]
                similar_user_ids = [user_artist_matrix.index[i] for i in similar_users]

                similar_users_data = user_artists[user_artists["userID"].isin(similar_user_ids)]

                recommended_artists = (
                    similar_users_data.groupby("artistID")["weight"]
                    .sum()
                    .reset_index()
                    .sort_values(by="weight", ascending=False)
                )

                already_listened = set(user_history["artistID"])
                recommended_artists = recommended_artists[~recommended_artists["artistID"].isin(already_listened)]

                recommended_artists = recommended_artists.merge(
                    artists, left_on="artistID", right_on="id"
                )[["artistID", "name"]].head(5)

                for artist_name in recommended_artists["name"]:
                    st.markdown(f"- **{artist_name}**")
            else:
                st.warning(f"User {user_id} not found in matrix.")

            top_artists = (
                user_history.groupby("artistID")["weight"]
                .sum()
                .nlargest(10)
                .reset_index()
                .merge(artists, left_on="artistID", right_on="id")
            )

            import plotly.express as px  # assure-toi que c'est bien importé en haut du script

            st.subheader("📊 Your Top 10 Artists")
            sorted_top_artists = top_artists.sort_values(by="weight", ascending=False)
            sorted_top_artists["name"] = pd.Categorical(
                sorted_top_artists["name"],
                categories=sorted_top_artists["name"],
                ordered=True
            )

            fig = px.bar(
                sorted_top_artists,
                x="name",
                y="weight",
                title=None,
                color_discrete_sequence=["#1DB954"]  # 💚 Vert Spotify
            )
            fig.update_layout(
                plot_bgcolor="#0b3d0b",
                paper_bgcolor="#0b3d0b",
                font_color="white",
                xaxis_title=None,
                yaxis_title="Listening Weight",
                margin=dict(t=10, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"No listening history found for user {user_id}.")
    else:
        st.info("Please enter your User ID to view your profile.")
