

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

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
 



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="CRMN Dashboard", page_icon=":bar_chart:", layout="wide")

 

# Define the file path with .xlsx extension
xlsx_file_path = "missions.xlsx"
sheet_name = "Sheet"  # Replace with your actual sheet name if different

# Function to save data to an Excel file
def save_data(df, file_path, sheet_name):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Fonction pour adapter le modèle au nombre de classes
def modify_model(model, num_classes):
    # Obtenir le nombre de caractéristiques d'entrée pour la tête de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Remplacer la tête de classification
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)

    return model

# Charger le modèle
@st.cache_resource
def load_model():
    # Charger le modèle pré-entraîné sans les poids
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # Adapter le modèle au nombre de classes de votre modèle sauvegardé
    model = modify_model(model, num_classes=9)  # Nombre de classes dans votre modèle sauvegardé
    model_path = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/weights/FRCNN-V0-5epochs.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

modelFast = load_model()

# Définir les transformations pour le prétraitement
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Fonction pour effectuer les inférences
def predict(image):
    image_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension de lot
    with torch.no_grad():
        predictions = modelFast(image_tensor)
    return predictions

# Fonction pour dessiner les boîtes de délimitation
def draw_boxes(image, boxes, labels, scores, threshold):
    image_np = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box
            image_np = cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            image_np = cv2.putText(image_np, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return Image.fromarray(image_np)

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

# Load Pre-trained ML Model
try:
    model_pathYOLOv9 = Path(settings.DETECTION_MODEL_YOLOv9)
    model_pathYOLOv8 = Path(settings.DETECTION_MODEL_YOLOv8)
    model_pathFASTER = Path(settings.DETECTION_MODEL_FASTER)

    modelYOLO_V9 = helper.load_model(model_pathYOLOv9)
    modelYOLO_V8 = helper.load_model(model_pathYOLOv8)
    modelFASTER = helper.load_model(model_pathFASTER)

except Exception as ex:
    st.error(f"Impossible de charger le modèle YOLO V9. Vérifiez le chemin spécifié: {model_pathYOLOv9}")
    st.error(f"Impossible de charger le modèle YOLO V8. Vérifiez le chemin spécifié: {model_pathYOLOv8}")
    st.error(f"Impossible de charger le modèle FASTER RCNN. Vérifiez le chemin spécifié: {model_pathFASTER}")
    st.error(ex)

# ---- READ EXCEL ----
@st.cache_data(ttl=60)
def get_data_from_excel():
    df = pd.read_excel(
        io=xlsx_file_path,
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


model_pathYOLOv9 = Path(settings.DETECTION_MODEL_YOLOv9)
model_pathYOLOv8 = Path(settings.DETECTION_MODEL_YOLOv8)
model_pathFASTER = Path(settings.DETECTION_MODEL_FASTER)

#model_path = Path(settings.DETECTION_MODEL)


confidence = float(st.sidebar.slider(
    "Selectionnez la confidence du model", 25, 100, 40)) / 100


# Sidebar navigation
st.sidebar.header("Navigation")
navigation = st.sidebar.radio(
    "",
    ('Acceuil', 'Image', 'Video')
)


    # Sidebar model selection
st.sidebar.header("Sélectionnez le modèle")
model_selection = st.sidebar.selectbox(
    "Choisissez le modèle pour la détection d'objets",
    ("YOLOv9", "YOLOv8", "Faster R-CNN")
)


# Main content based on navigation selection
if navigation == 'Acceuil':
      
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
            "TypeDrone == @drone & MissionType ==@mission_type & Saison == @saison"
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
            xaxis=(dict(showgrid=False))
        )

 

        # Compter le nombre d'occurrences de chaque type de mission
        mission_type_counts = df["MissionType"].value_counts().reset_index()
        mission_type_counts.columns = ["MissionType", "Count"]

        # Créer un graphique à secteurs pour afficher la répartition par type de mission
        fig_mission_type_counts = px.pie(
            mission_type_counts, values="Count", names="MissionType", title="<b>Répartition par Type de Mission</b>",
            color_discrete_sequence=px.colors.qualitative.Set1, template="plotly_white"
        )
        #st.plotly_chart(fig_mission_type_sales, use_container_width=True)

        # Compter le nombre d'occurrences de chaque saison
        saison_counts = df["Saison"].value_counts().reset_index()
        saison_counts.columns = ["Saison", "Count"]

        # Créer un graphique à secteurs pour afficher la répartition par saison
        fig_saison_counts = px.pie(
            saison_counts, values="Count", names="Saison", title="<b>Répartition par Saison</b>",
            color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_white"
        )
        #st.plotly_chart(fig_mission_type_saison, use_container_width=True)


        left_column, right_column = st.columns(2)
        left_column.plotly_chart(fig_mission_type_counts, use_container_width=True)
        right_column.plotly_chart(fig_saison_counts, use_container_width=True)




        # Calculer les totaux par type d'objet
        total_objects = df_selection[["Drone", "Plongeur", "Roche", "Bateau", "Poisson", "Corail", "Tortue", "Asterie", "Avion"]].sum()

        # Créer le graphique à barres avec Plotly Express
        fig_objects = px.bar(
            x=total_objects.values,
            y=total_objects.index,
            orientation="h",
            title="<b>Total par type d'objet dans les missions sélectionnées</b>",
            labels={"x": "Total", "y": "Type d'objet"},
            color_discrete_sequence=["#E694FF"] * len(total_objects),  # Utilisation de la couleur préférée
            template="plotly_white",
        )

        # Mise à jour du layout du graphique
        fig_objects.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",  # Fond transparent
            xaxis=dict(showgrid=False),  # Pas de grille sur l'axe x
        )
 
 
        # Widgets pour la sélection des colonnes
        columns_of_interest = st.multiselect('Sélectionner les colonnes à visualiser', df.columns.tolist())

        # Filtrer le DataFrame avec les colonnes sélectionnées
        df_filtered = df[columns_of_interest]

        # Convertir la colonne 'DateFin' en datetime pour faciliter le groupement
        if 'DateFin' in df_filtered.columns:
            df_filtered['DateFin'] = pd.to_datetime(df_filtered['DateFin'])

            # Sélection unique des dates de fin
            unique_dates = df_filtered['DateFin'].unique()

            # Grouper les données par 'DateFin' et sommer les détections
            df_grouped = df_filtered.groupby('DateFin').sum()

            # Tracer les graphiques
            fig, ax = plt.subplots(figsize=(10, 6))

            for column in columns_of_interest[1:]:  # Exclure 'DateFin' de la boucle de traçage
                ax.plot(df_grouped.index, df_grouped[column], marker='o', label=column)

            ax.set_title('Nombre de détections en fonction de la date de fin')
            ax.set_xlabel('Date de fin')
            ax.set_ylabel('Nombre de détections')
            ax.legend()
            ax.grid(True)

            # Formatter les étiquettes de l'axe des abscisses (DateFin)
            date_format = DateFormatter('%d-%m-%Y')
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
        else:
            st.warning('Sélectionnez au moins la colonne "DateFin" et une autre colonne pour visualiser les données.')


        


        # ---- HIDE STREAMLIT STYLE ----
        hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_st_style, unsafe_allow_html=True)
elif navigation == 'Image':
    st.subheader("Détection d'objets sur une image")
    source_img = st.file_uploader("Choisir une image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    if source_img is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        #default_image = PIL.Image.open(default_image_path)
        #st.image(default_image, caption="Image par défaut", use_column_width=True)
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Image chargée", use_column_width=True)
        #st.session_state.pop('detected_objects', None)

        # Load the selected model
        if model_selection == "YOLOv9":
            model = modelYOLO_V9
        elif model_selection == "YOLOv8":
            model = modelYOLO_V8
        elif model_selection == "Faster R-CNN":  # Faster R-CNN
            model = modelFASTER

        if st.button('Détecter les objets') and (model_selection == "YOLOv9" or model_selection == "YOLOv8")  :
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
            res = model.predict(uploaded_image, conf=confidence)
            st.session_state.detected_objects = {
                'image': uploaded_image,
                'results': res
            }
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            #st.image(res_plotted, caption='Image avec objets détectés', use_column_width=True)

            with st.expander("Résultats de la détection"):
                for box in boxes:
                    label_index = box.cls.item()
                    label = class_mapping.get(label_index, 'Unknown')
                    #st.write(f"Objets détectées: {label} avec confidence: {box.conf.item()}")  # Display each label

                    #if label in object_counts:
                        #object_counts[label] += 1

                    # Mettre à jour le compteur d'objets
                    if label in st.session_state.object_counts:
                        st.session_state.object_counts[label] += 1

            detected_objects = st.session_state.detected_objects
            uploaded_image = detected_objects['image']
            res = detected_objects['results']


        else :

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
                
                # Prédictions
            res = predict(uploaded_image)

            st.session_state.detected_objects = {
                'image': uploaded_image,
                'results': res
            }
                # Extraire les boîtes de délimitation, labels et scores
            boxes = res[0]['boxes'].cpu().numpy()
            labels = res[0]['labels'].cpu().numpy()
            scores = res[0]['scores'].cpu().numpy()

                # Assumer que vous avez un mapping des IDs des classes vers les noms
            id_to_class = {0: 'drone', 1: 'plongeur', 2: 'roche', 3: 'bateau', 4: 'poisson', 5: 'corail', 6: 'tortue', 7: 'asterie', 8: 'avion'}
            labels = [id_to_class[int(label)] for label in labels]

                # Dessiner les boîtes de délimitation sur l'image avec le seuil de confiance
            image_with_boxes = draw_boxes(uploaded_image, boxes, labels, scores, confidence)
            st.image(image_with_boxes, caption='Image avec objets détectés')
                
            # Mettre à jour le compteur d'objets
            for label in labels:
                if label in st.session_state.object_counts:
                    st.session_state.object_counts[label] += 1

            st.write("Historique des objets détectés:")
            for obj, count in st.session_state.object_counts.items():
                st.write(f"{obj}: {count}")


        if 'detected_objects' in st.session_state and 'object_counts' in st.session_state:

            detected_objects = st.session_state.detected_objects
            uploaded_image = detected_objects['image']
            res = detected_objects['results']

            if model_selection == "YOLOv9" or model_selection == "YOLOv8":
                # Display the image with detected objects
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Image avec objets détectés', use_column_width=True)

                # Display detection results
                with st.expander("Résultats de la détection"):
                    for box in res[0].boxes:
                        label_index = box.cls.item()
                        label = class_mapping.get(label_index, 'Unknown')
                        st.write(f"Objets détectés: {label}, avec confidence: {box.conf.item()}")

            counts = st.session_state.object_counts
            if model_selection == "Faster R-CNN":
                # Afficher les compteurs des objets détectés
                if any(value > 0 for value in st.session_state.object_counts.values()):
                    st.write("## Compteurs des objets détectés")
                    st.session_state.object_counts_df = pd.DataFrame(list(st.session_state.object_counts.items()), columns=['Objet', 'Nombre'])
                    st.table(st.session_state.object_counts_df)

                    # Formulaire pour ajouter de nouvelles données
                    st.title("Ajouter de nouvelles données")
                    with st.form("mission_form"):
                        total = st.number_input("Temps Total (Minutes)", min_value=0)
                        rating = st.number_input("Note", min_value=0.0, max_value=5.0, step=0.1)
                        time = st.time_input("Heure", datetime.now().time())
                        date_debut = st.date_input("DateDebut", datetime.now().date()).strftime("%Y-%m-%d")
                        date_fin = st.date_input("DateFin", datetime.now().date()).strftime("%Y-%m-%d")
                        mission_type = st.selectbox("MissionType", df["MissionType"].unique())
                        saison = st.selectbox("Saison", df["Saison"].unique())
                        type_drone = st.selectbox("TypeDrone", df["TypeDrone"].unique())
                        submitted = st.form_submit_button("Ajouter")

                    # Sauvegarde des nouvelles données
                    if submitted:
                        new_data = {
                            'Total': total,
                            'Rating': rating,
                            'Time': time.strftime("%H:%M:%S"),
                            'DateDebut': date_debut,
                            'DateFin': date_fin,
                            'MissionType': mission_type,
                            'Saison': saison,
                            'TypeDrone': type_drone,
                            'Drone': st.session_state.object_counts['Drone'],
                            'Plongeur': st.session_state.object_counts['Plongeur'],
                            'Roche': st.session_state.object_counts['Roche'],
                            'Bateau': st.session_state.object_counts['Bateau'],
                            'Poisson': st.session_state.object_counts['Poisson'],
                            'Corail': st.session_state.object_counts['Corail'],
                            'Tortue': st.session_state.object_counts['Tortue'],
                            'Asterie': st.session_state.object_counts['Asterie'],
                            'Avion': st.session_state.object_counts['Avion']
                        }

                        new_df = pd.DataFrame([new_data])
                        df = pd.concat([df, new_df], ignore_index=True)
                        save_data(df, xlsx_file_path, sheet_name)
                        st.cache_data.clear()
                        st.success("Nouvelle mission ajoutée avec succès!")

                        # Rediriger vers la page d'accueil
                        # st.experimental_rerun()
            if model_selection == "YOLOv9" or model_selection == "YOLOv8":
                if any(value > 0 for value in counts.values()):
                        
    
                    # Display the counts in a table
                    st.write("## Compteurs des objets détectés")
                    st.session_state.object_counts_df = pd.DataFrame(list(st.session_state.object_counts.items()), columns=['Objet', 'Nombre'])
                    st.table(st.session_state.object_counts_df)


                    # Formulaire pour ajouter de nouvelles données
                    st.title("Ajouter de nouvelles données")
                    with st.form("mission_form"):
                        total = st.number_input("Temps Total (Minutes)", min_value=0)
                        rating = st.number_input("Note", min_value=0.0, max_value=5.0, step=0.1)
                        time = st.time_input("Heure", datetime.now().time())
                        date_debut = st.date_input("DateDebut", datetime.now().date()).strftime("%Y-%m-%d")
                        date_fin = st.date_input("DateFin", datetime.now().date()).strftime("%Y-%m-%d")
                        mission_type = st.selectbox("MissionType", df["MissionType"].unique())
                        saison = st.selectbox("Saison", df["Saison"].unique())
                        type_drone = st.selectbox("TypeDrone", df["TypeDrone"].unique())
                        submitted = st.form_submit_button("Ajouter")

                    # Sauvegarde des nouvelles données
                    if submitted:
                        new_data = {
                            'Total': total,
                            'Rating': rating,
                            'Time': time.strftime("%H:%M:%S"),
                            'DateDebut': date_debut,
                            'DateFin': date_fin,
                            'MissionType': mission_type,
                            'Saison': saison,
                            'TypeDrone': type_drone,
                            'Drone': st.session_state.object_counts['Drone'],
                            'Plongeur': st.session_state.object_counts['Plongeur'],
                            'Roche': st.session_state.object_counts['Roche'],
                            'Bateau': st.session_state.object_counts['Bateau'],
                            'Poisson': st.session_state.object_counts['Poisson'],
                            'Corail': st.session_state.object_counts['Corail'],
                            'Tortue': st.session_state.object_counts['Tortue'],
                            'Asterie': st.session_state.object_counts['Asterie'],
                            'Avion': st.session_state.object_counts['Avion']
                        }

                        new_df = pd.DataFrame([new_data])
                        df = pd.concat([df, new_df], ignore_index=True)
                        save_data(df, xlsx_file_path, sheet_name)
                        st.cache_data.clear()
                        st.success("Nouvelle mission ajoutée avec succès!")
                            
                        # Rediriger vers la page d'accueil
                        #st.experimental_rerun()

    
            else:
                st.write("Aucun objet n'a été détecté.")
        else :
            st.write("")


elif navigation == 'Video':

    # Load the selected model
    if model_selection == "YOLOv9":
            model = modelYOLO_V9
    elif model_selection == "YOLOv8":
            model = modelYOLO_V8
    elif model_selection == "Faster R-CNN": 
            model = modelFASTER        
    st.subheader("Détection d'objets sur une vidéo")
    # Call function for video detection if implemented in helper module
    helper.play_stored_video(confidence, model)

else:
    st.error("SVP, sélectionnez une option de navigation valide !") 



 