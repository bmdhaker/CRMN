from pathlib import Path
import PIL
import pickle
from datetime import datetime
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import settings
import helper

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="CRMN Dashboard", page_icon=":bar_chart:", layout="wide")

# Define the file path with .xlsx extension
xlsx_file_path = "missions.xlsx"
sheet_name = "Sheet"  # Replace with your actual sheet name if different

# Function to save data to an Excel file
def save_data(df, file_path, sheet_name):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Initialize object counts
object_counts = {   'Drone': 0,
                    'Plongeur': 0,
                    'Roche': 0,
                    'Bateau': 0,
                    'Poisson': 0,
                    'Corail': 0,
                    'Tortue': 0,
                    'Asterie': 0,
                    'Avion': 0
                }

# Mapping of class indices to object names
class_mapping = {
                    0: 'Drone',
                    1: 'Plongeur',
                    2: 'Roche',
                    3: 'Bateau',
                    4: 'Poisson',
                    5: 'Corail',
                    6: 'Tortue',
                    7: 'Asterie',
                    8: 'Avion'
                }

# Initialisation des comptes d'objets dans la session
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = {
        'Drone': 0,
        'Plongeur': 0,
        'Roche': 0,
        'Bateau': 0,
        'Poisson': 0,
        'Corail': 0,
        'Tortue': 0,
        'Asterie': 0,
        'Avion': 0
    }

# Load Pre-trained ML Models
model_pathYOLOv9 = Path(settings.DETECTION_MODEL_YOLOv9)
model_pathYOLOv8 = Path(settings.DETECTION_MODEL_YOLOv8)
model_pathFASTER = Path(settings.DETECTION_MODEL_FASTER)

modelYOLO_V9 = helper.load_model(model_pathYOLOv9)
modelYOLO_V8 = helper.load_model(model_pathYOLOv8)
modelFASTER = helper.load_model(model_pathFASTER)

# --- USER AUTHENTICATION ---
names = ["Abdoul Aziz Baoula", "Cleeve"]
usernames = ["abdoul_aziz", "cleeve"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "CRMN_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Nom d'utilisateur / mot de passe incorrect")

if authentication_status == None:
    st.warning("Entrer votre nom d'utilisateur et mot de passe")

if authentication_status:
    # ---- READ EXCEL ----
    @st.cache_data(ttl=60)
    def get_data_from_excel():
        df = pd.read_excel(
            io="missions.xlsx",
            engine="openpyxl",
            sheet_name=sheet_name,
            #skiprows=3,
            usecols="A:Q",
            #nrows=1000,
        )
        # Assurez-vous que DateDebut est de type datetime.date
        df['DateDebut'] = pd.to_datetime(df['DateDebut'], errors='coerce').dt.date
        df['DateFin'] = pd.to_datetime(df['DateFin'], errors='coerce').dt.date
   
        return df

    df = get_data_from_excel()

    st.sidebar.title(f"Bienvenue \n {name}")
    authenticator.logout("Déconnexion", "sidebar")

    confidence = float(st.sidebar.slider(
    "Sélectionnez la confidence du modèle", 25, 100, 40)) / 100

    # Sidebar navigation
    st.sidebar.header("Navigation")
    navigation = st.sidebar.radio(
        "",
        ('Accueil', 'Image', 'Vidéo')
    )

    # Sidebar model selection
    st.sidebar.header("Sélectionnez le modèle")
    model_selection = st.sidebar.selectbox(
        "Choisissez le modèle pour la détection d'objets",
        ("YOLOv9", "YOLOv8", "Faster R-CNN")
    )

    # Main content based on navigation selection
    if navigation == 'Accueil':
        # Add content for the main page if needed
        st.sidebar.header("Filtrez ici:")
        drone = st.sidebar.multiselect(
            "Sélectionnez le drone:",
            options=df["TypeDrone"].unique(),
            default=df["TypeDrone"].unique()
        )

        mission_type = st.sidebar.multiselect(
            "Sélectionnez le type de mission:",
            options=df["MissionType"].unique(),
            default=df["MissionType"].unique(),
        )

        saison = st.sidebar.multiselect(
            "Sélectionnez la saison:",
            options=df["Saison"].unique(),
            default=df["Saison"].unique()
        )

        df_selection = df.query(
            "TypeDrone == @drone & MissionType == @mission_type & Saison == @saison"
        )

        # ---- MAINPAGE ----
        st.title(":bar_chart: CRMN Dashboard")
        st.markdown("##")

        # Trier par dateDebut de la plus récente à la plus ancienne
        df_sorted = df.sort_values(by="DateDebut", ascending=False)

        # Afficher les premières lignes triées
        st.write("Aperçu de la table des missions:")
        st.dataframe(df_sorted.head().reset_index(drop=True))

        st.markdown("---")

        # Graphique : Mission par Type de Drone
        mission_by_typedrone = (
            df_selection.groupby(by=["TypeDrone"]).sum(numeric_only=True)[["Total"]].sort_values(by="Total")
        )
        fig_typedrone_mission = px.bar(
            mission_by_typedrone,
            x="Total",
            y=mission_by_typedrone.index,
            orientation="h",
            title="<b>Temps de Mission par Type de Drone </b>",
            color_discrete_sequence=["#0083B8"] * len(mission_by_typedrone),
            template="plotly_white",
        )
        fig_typedrone_mission.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )

        st.plotly_chart(fig_typedrone_mission, use_container_width=True)

        # TOP KPI's
        total = int(df_selection["Total"].sum())
        average_rating = round(df_selection["Rating"].mean(), 1)
        star_rating = ":star:" * int(round(average_rating))
        max_rating = round(df_selection["Rating"].max(), 2)
        max_star_rating = ":star:" * int(round(max_rating))
        min_rating = round(df_selection["Rating"].min(), 2)
        min_star_rating = ":star:" * int(round(min_rating))

        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.subheader("Note min :")
            st.subheader(f"{min_rating} {min_star_rating}")
        with middle_column:
            st.subheader("Note max :")
            st.subheader(f"{max_rating} {max_star_rating}")
        with right_column:
            st.subheader("Note en moyenne :")
            st.subheader(f"{average_rating} {star_rating}")

        st.markdown("""---""")

        # Mission par Type de Drone [BAR CHART]
        mission_by_typedrone = (
            df_selection.groupby(by=["TypeDrone"]).sum(numeric_only=True)[["Total"]].sort_values(by="Total")
        )
        fig_typedrone_mission = px.bar(
            mission_by_typedrone,
            x="Total",
            y=mission_by_typedrone.index,
            orientation="h",
            title="<b>Mission par Type de Drone </b>",
            color_discrete_sequence=["#0083B8"] * len(mission_by_typedrone),
            template="plotly_white",
        )
        fig_typedrone_mission.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_typedrone_mission, use_container_width=True)

        st.markdown("---")

        # Mission par Saison [BAR CHART]
        mission_by_saison = (
            df_selection.groupby(by=["Saison"]).sum(numeric_only=True)[["Total"]].sort_values(by="Total")
        )
        fig_saison_mission = px.bar(
            mission_by_saison,
            x="Total",
            y=mission_by_saison.index,
            orientation="h",
            title="<b>Mission par Saison</b>",
            color_discrete_sequence=["#0083B8"] * len(mission_by_saison),
            template="plotly_white",
        )
        fig_saison_mission.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_saison_mission, use_container_width=True)

    # ---- IMAGE DETECTION ----
    if navigation == 'Image':
        st.title("Détection d'objets sur Image")

        uploaded_image = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Image téléversée", use_column_width=True)

            # Convert to numpy array for model processing
            image_np = np.array(image)

            # Load the selected model
            if model_selection == "YOLOv9":
                model = modelYOLO_V9
            elif model_selection == "YOLOv8":
                model = modelYOLO_V8
            else:  # Faster R-CNN
                model = modelFASTER

            # Perform detection
            results = helper.detect_objects(image_np, model, confidence)
            st.write("Résultats de la détection :")
            st.json(results)

            # Process results and display
            for obj in results:
                label = obj['label']
                st.session_state.object_counts[class_mapping[label]] += 1

    # ---- VIDEO DETECTION ----
    if navigation == 'Vidéo':
        st.title("Détection d'objets sur Vidéo")

        uploaded_video = st.file_uploader("Choisir une vidéo", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            # Load the selected model
            if model_selection == "YOLOv9":
                model = modelYOLO_V9
            elif model_selection == "YOLOv8":
                model = modelYOLO_V8
            else:  # Faster R-CNN
                model = modelFASTER

            # Perform detection on video
            results = helper.detect_objects_video(uploaded_video, model, confidence)
            st.write("Résultats de la détection :")
            st.json(results)

            # Process results and display
            for obj in results:
                label = obj['label']
                st.session_state.object_counts[class_mapping[label]] += 1

    # ---- TABLEAU DES DONNEES DETECTEES ----
    st.write("Tableau des objets détectés:")
    st.write(pd.DataFrame(st.session_state.object_counts.items(), columns=['Objet', 'Nombre']))

    # ---- ENREGISTRER LES DONNEES ----
    # Assuming you want to save the object counts to an Excel file
    df_counts = pd.DataFrame(st.session_state.object_counts.items(), columns=['Objet', 'Nombre'])
    save_data(df_counts, xlsx_file_path, "ObjectCounts")

    # Display the saved file
    st.write(f"Les données ont été sauvegardées dans {xlsx_file_path}.")
