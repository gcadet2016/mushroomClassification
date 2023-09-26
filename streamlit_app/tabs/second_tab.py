import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import config
import matplotlib.pyplot as plt
import seaborn as sns

title = "Exploration de données"
sidebar_name = "Exploration de données"


def run():

    st.title(title)

    st.markdown(
        """
        ## Identification des champignons
        Le champignon possède un label (nom). Ce label est unique pour chaque espèce, nous l'utiliserons donc comme variable cible.  
        Initialement, la variable cible contient 11999 espèces enregistrées:  
        Les histogrammes ci-dessous représentent la répartition des espèces observées.
        En examinant la distribution initiale de la variable cible, nous constatons que certaines espèces sont très bien représentées, tandis que d'autres sont sous-représentées.  
        Par exemple, une des espèces est représentée par 2 images sur un total de 647 615.
        
        """
    )
    # todo delete
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
    #st.line_chart(chart_data)
    #

    #=====================================================
    # Analyse des labels de champignons dans l'echantillon
    #=====================================================
    infos_images  = pd.read_csv(config.INFOS_IMAGES_PATH, low_memory=False)
    # Représentation de la distribution de la variable :


    fig = plt.figure(figsize=(12, 20))
    sns.countplot(data=infos_images, y="label", order=infos_images["label"].value_counts().index)
    plt.xlabel("Nombre d'images")
    plt.ylabel("Espèce de champignon")
    plt.title("Répartition des espèces de champignons dans l'échantillon")
    st.pyplot(fig)

    st.markdown(
            """
            ## Top 10 des champignons les plus représentés
            Nous sommes dans un cas de distribution déséquilibrée.  
            Pour des raisons techniques (temps de traitements des données), nous allons nous concentrer sur les 10 espèces les plus représentées.
            Néanmoins si nous devions travailler sur l'ensemble des espèces, nous devrions utiliser des techniques de rééquilibrage des données (ex. oversampling, undersampling, SMOTE, etc.).
            """
        )

    # Création du top 10 des espèces les plus représentées.

    # Analyse des labels de champignons dans l'echantillon
    top10 = pd.read_csv(config.TOP10_PATH, low_memory=False)
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(data=top10, x="label", order=top10["label"].value_counts().index)
    plt.xlabel("Espèce de champignon")
    plt.xticks(rotation=70)
    plt.ylabel("Nombre d'images")
    plt.title("TOP 10 des espèces les plus représentées")
    st.pyplot(fig)

    st.markdown(
            """
            ### Répartition proportinonelle à l'ensemble du jeu de données
            """
        )

    comptage_valeurs = infos_images['label'].value_counts()
    top10_label = comptage_valeurs.head(10)

    # Reste des valeurs présentes :
    reste_hors_top10 = comptage_valeurs.iloc[10:].sum()

    # Création d'une series contenant le top 10 et le reste.
    top10_et_reste = top10_label.append(pd.Series([reste_hors_top10], index=['Autres']))

    # Calcul des pourcentages pour chaque valeur (espèce) :

    total = infos_images['label'].count()
    pourcentages_valeurs_top10 = (top10_label / total) * 100    # uniquement le top 10
    pourcentages_valeurs_tout = (top10_et_reste / total) * 100  # Toutes les espèces (top 10 + le reste)

    # Construction du graphique :
    fig = plt.figure(figsize=(8,5))
    #plt.grid(True, linestyle = '--')

    plt.title('Distribution de \'label\'')
    plt.xlabel('Espèces')
    plt.xticks(rotation = 70)
    plt.ylabel('Pourcentage d\'apparition')
    plt.bar(pourcentages_valeurs_tout.index, pourcentages_valeurs_tout, color = 'g')


    # Affichage les pourcentages au-dessus des barres
    for i, value in enumerate(pourcentages_valeurs_tout):
        plt.text(i, value + 1, f"{value:.2f}%", ha='center', va='bottom')


    plt.annotate(' TOP 10 ', xy=(9, 10), xytext=(2.5, 60), arrowprops={'facecolor':'blue'})
    plt.annotate(' ', xy=(0, 10), xytext=(2.5, 60), arrowprops={'facecolor':'blue'});

    st.pyplot(fig)

    st.markdown(
        """
        ## Dimensions des images
        l'analyse de la distribution des dimensions des images nous permet de constater que :
            - Les images sont généralement de largeur 320 pixels   
        -	Les images sont généralement de hauteur 240 px avec également une partie assez importante d’entre elles de hauteurs 220 px et 320 px  
        -	Le total de pixels en hauteur x largeur (sans les canaux RGB), correspond en majorité à 75 000 px, une partie importante d’entre elles comporte 70 000 px.
        """
    )
    import cv2
    def extract_features(url_img):

        '''
        Extrait les features des images, les renvoient sous forme d'un DataFrame contenant les largeurs, hauteurs, et moyennes RGB
            - url_img : Chemin des images
        '''

        img = cv2.imread(url_img)
        hauteur, largeur, canal = img.shape
        features = {
            'largeur': largeur,
            'hauteur': hauteur,
            'moyenne_rouge': np.mean(img[:,:,2]),
            'moyenne_vert': np.mean(img[:,:,1]),
            'moyenne_bleu': np.mean(img[:,:,0])}
        
        return features
    
    # Extraction des features des images du top 10
    liste_features = []

    for index, row in top10.iterrows():
        filepath = row['image_url']
        features = extract_features(filepath)
        liste_features.append(features)

    features_top10 = pd.DataFrame(liste_features)
    features_top10['moyenne_couleurs'] = (features_top10['moyenne_rouge']\
                                            + features_top10['moyenne_vert']\
                                            + features_top10['moyenne_bleu']) / 3

    features_top10['label'] = top10['label']
    features_top10.head()


    fig = plt.figure(figsize=(16,12))
    sns.catplot(features_top10, kind='boxen')
    plt.grid(False)
    plt.title('Box Plot des features')
    plt.xticks(rotation = 45);

    st.pyplot(fig)

    
    st.markdown(
        """
        ## Test 3

        You can also display images using [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

        ```python
        import streamlit as st
        from PIL import Image

        st.image(Image.open("assets/sample-image.jpg"))

        ```

        """
    )

    st.image(Image.open("assets/sample-image.jpg"))
