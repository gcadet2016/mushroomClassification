import streamlit as st


title = "Classification de champignon"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Le projet porte sur la classification d'espèces de champignons à partir d'images.  
        Objectif : développer une application capable d'identifier une espèce de champignon à partir d'une photo. 
        Le modèle devra être entrainé et testé sur un jeu de données (ensembles de photos). Par la suite il pourra être utilisé pour identifier des champignons à partir de photos prises par des utilisateurs.
        L'algorithme utilise la technologie 'computer vision' pour s'entrainer et pour classifier les images.  

        ## Contexte et analyse des données
        Pour atteindre les objectifs de notre projet, nous utilisons la base de données [Mushroom Observer](https://mushroomobserver.org/).  
        Initialement, cette base contient plus de 600 000 images de champignons qui nous permettront d'entraîner le modèle et de le tester.  
        Il reste neanmoins possible d'enrichir la base à l'aide de nouvelles images :
          - Images sourcées d'internet (ex. Google Images),  
          - Images générées par un prompt IA (Ex. Midjourney),  
          - Contributions personnelles (photographies).   
        Nous possedons également des datasets (fichiers .csv) associés aux images qui necessitent un ensemble de pré-traitements. 
        Ces informations complémentaires proviennent également de la base de données de Mushroom Observer.

        ## Remarques
        Il n'y a pas d'objectif de précision fixé à terme.  
        Nous limiterons volontairement l'étude à dix classes de champignons pour des raisons techniques (temps des traitements informatique) et pour respecter le délai imposé sur le projet.
        Ces dix espèces correspondent aux espèces les plus représentées du jeu de données.

        ## Prérequis
        Les images ne sont pas dans la repository GitHub. Il faut les télécharger et dezipper dans le dossier '../../images'.  
        - images dans le dossier '../../images'  
        - Sans les images, certaines pages ne fonctionneront pas.
        """
    )
