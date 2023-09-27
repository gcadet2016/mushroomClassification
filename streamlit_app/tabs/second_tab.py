import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Second tab"
sidebar_name = "Second Tab"


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
    plt.ylabel("Espèces de champignon")
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
    plt.xlabel("Espèces de champignon")
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

        ## Test

        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse gravida urna vel tincidunt vestibulum. Nunc malesuada molestie odio, vel tincidunt arcu fringilla hendrerit. Sed leo velit, elementum nec ipsum id, sagittis tempus leo. Quisque viverra ipsum arcu, et ullamcorper arcu volutpat maximus. Donec volutpat porttitor mi in tincidunt. Ut sodales commodo magna, eu volutpat lacus sodales in. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam interdum libero non leo iaculis bibendum. Suspendisse in leo posuere risus viverra suscipit.

        Nunc eu tortor dolor. Etiam molestie id enim ut convallis. Pellentesque aliquet malesuada ipsum eget commodo. Ut at eros elit. Quisque non blandit magna. Aliquam porta, turpis ac maximus varius, risus elit sagittis leo, eu interdum lorem leo sit amet sapien. Nam vestibulum cursus magna, a dapibus augue pellentesque sed. Integer tincidunt scelerisque urna non viverra. Sed faucibus leo augue, ac suscipit orci cursus sed. Mauris sit amet consectetur nisi.
        """
    )

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))

    st.line_chart(chart_data)

    st.markdown(
        """
        ## Test 2

        Proin malesuada diam blandit orci auctor, ac auctor lacus porttitor. Aenean id faucibus tortor. Morbi ac odio leo. Proin consequat facilisis magna eu elementum. Proin arcu sapien, venenatis placerat blandit vitae, pharetra ac ipsum. Proin interdum purus non eros condimentum, sit amet luctus quam iaculis. Quisque vitae sapien felis. Vivamus ut tortor accumsan, dictum mi a, semper libero. Morbi sed fermentum ligula, quis varius quam. Suspendisse rutrum, sapien at scelerisque vestibulum, ipsum nibh fermentum odio, vel pellentesque arcu erat at sapien. Maecenas aliquam eget metus ut interdum.
        
        ```python

        def my_awesome_function(a, b):
            return a + b
        ```

        Sed lacinia suscipit turpis sit amet gravida. Etiam quis purus in magna elementum malesuada. Nullam fermentum, sapien a maximus pharetra, mauris tortor maximus velit, a tempus dolor elit ut lectus. Cras ut nulla eget dolor malesuada congue. Quisque placerat, nulla in pharetra dapibus, nunc ligula semper massa, eu euismod dui risus non metus. Curabitur pretium lorem vel luctus dictum. Maecenas a dui in odio congue interdum. Sed massa est, rutrum eu risus et, pharetra pulvinar lorem.
        """
    )

    st.area_chart(chart_data)

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
